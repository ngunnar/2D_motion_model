import pandas as pd
import tensorflow as tf
import os
import numpy as np
import random
import skimage.draw
import cv2

trace_error = ['0X2507255D8DC30B4E', 
               '0X280B7441A7E287B2',
               '0X3269691452F4F42A',
               '0X35A5E9C9075E56EE',
               '0X366AD377E4A81FBE',
               '0X367085DDC2E90657',
               '0X3A79044052AC3197',
               '0X3EB0FC2695B0AB5F',
               '0X4154F112065C857B',
               '0X43DE853BD6E0C849',
               '0X46ACC9C2CF9CFB1E',
               '0X4B3A70F6BD40224B',
               '0X4BBA9C8FB485C9AB', #
               '0X4DEAED0369D251DE',
               '0X4E9F08061D109568',
               '0X500FC4E8716B0A8F',
               '0X53BD50EB0C43D30D',
               '0X59D5F41F45601E03',
               '0X62120814160BA377',
               '0X642E639A8CDE539B',
               '0X67E8F2D130F1A55',
               '0X67F8AC58B0BAA98',
               '0X6E02E0F24F63EFD7',
               '0X6E5824E76BEB3ECA',
               '0X766B7B0ABDB07CD5',
               '0X76CF3E993E9964A9',
               '0X7F33CFDB31ADCD6A',
               '0XB8513240ED5E94'
               ]

def loadvideo(filename: str):
    """Loads a video from a file.
    Args:
        filename (str): filename of video
    Returns:
        A np.ndarray with dimensions (frames, height, width). The
        values will be uint8's ranging from 0 to 255.
    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_width, frame_height), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        v[count] = frame

    return v

def read_trace(t, video):
    x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    x = np.concatenate((x1[1:], np.flip(x2[1:])))
    y = np.concatenate((y1[1:], np.flip(y2[1:])))

    r, c = skimage.draw.polygon(np.rint(y).astype('int'), np.rint(x).astype('int'), (video.shape[1], video.shape[2]))
    mask = np.zeros((video.shape[1], video.shape[2]), np.float32)
    mask[r, c] = 1
    return mask

def train_generator(df, root, length, normalization_layer):
    def gen():
        for index, row in df.iterrows():
            video = loadvideo(os.path.join(root, 'Videos',row['FileName']+'.avi'))
            if video.shape[0] < length:
                continue
            start_frame = random.randint(0, video.shape[0]-length)
            v = video[start_frame:start_frame+length,...]
            v = normalization_layer(v)
            mask = tf.zeros((length,), dtype=tf.bool)
            yield {"input_video": v, 'input_ref': v[0,...], "input_mask": mask}
    return gen

def test_generator(df, root, normalization_layer):
    def gen():
        for index, row in df.iterrows():            
            video = loadvideo(os.path.join(root, 'Videos',row['FileName']+'.avi'))
            trace_low = read_trace(np.asarray(row['TraceLow']).T, video)
            trace_high = read_trace(np.asarray(row['TraceHigh']).T, video)
            frame_low = int(row['FrameLow'])
            frame_high = int(row['FrameHigh'])
            
            video = normalization_layer(video)
            
            yield {'input_video': video[0:,...],
                   'trace_low': trace_low,
                   'trace_high': trace_high,
                   'frame_low': frame_low,
                   'frame_high': frame_high,
                   'file': row['FileName']
                   }
    return gen

def parse_csv(root):
    # Load fileList and volumeTracing
    fileslist = pd.read_csv(os.path.join(root, 'FileList.csv'))
    volumeTracing = pd.read_csv(os.path.join(root, 'VolumeTracings.csv'))

    grouped_df = volumeTracing.groupby(['FileName', 'Frame']).agg(lambda x: list(x)).reset_index()
    df = pd.DataFrame({'FileName': grouped_df['FileName'],
                        'Trace': list(zip(grouped_df['X1'], grouped_df['Y1'], grouped_df['X2'], grouped_df['Y2'])),
                        'Frame': grouped_df['Frame']
                        })
    
    df = df.groupby('FileName').agg({'Trace': ['first', 'last'], 'Frame': ['first', 'last']}).reset_index()
    df.columns = ['FileName', 'TraceLow', 'TraceHigh', 'FrameLow', 'FrameHigh']
    df['FileName'] = df['FileName'].str.replace('.avi', '')

    # Merge with fileList based on fileName
    df = pd.merge(fileslist, df, on='FileName')
    df['FrameDifference'] = df['FrameHigh'] - df['FrameLow']

    threshold_col = 'FrameDifference'
    #threshold = df['NumberOfFrames'].quantile(0.95)
    threshold = df[threshold_col].quantile(0.95)
    
    # Update Split # TODO
    df['Split'] = ['VAL' if nf > threshold else 'TRAIN' for nf in df[threshold_col]]

    # Split into train and val
    val_df = df[df['Split'] == 'VAL']
    train_df = df[df['Split'] == 'TRAIN']

    val_df = val_df[~val_df['FileName'].isin(trace_error)]
    return train_df, val_df


def get_dataset(root, length, shape):
    train_df, val_df = parse_csv(root)

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_dataset = tf.data.Dataset.from_generator(train_generator(train_df, root, length, normalization_layer),
                                                   output_types = ({"input_video": tf.float32, "input_ref": tf.float32, "input_mask": tf.bool}),
                                                   output_shapes = ({"input_video": (length, *shape), "input_ref": shape, "input_mask": (length)}))

    test_dataset = tf.data.Dataset.from_generator(train_generator(val_df, root, length, normalization_layer),
                                                   output_types = ({"input_video": tf.float32, "input_ref": tf.float32, "input_mask": tf.bool}),
                                                   output_shapes = ({"input_video": (length, *shape), "input_ref": shape, "input_mask": (length)}))
    
    len_train = (train_df['NumberOfFrames'] // length).sum()
    return train_dataset, test_dataset, len_train


def get_eval_data(root, shape, get_train=False):
    train_df, val_df = parse_csv(root)
    #val_df = train_df
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    test_dataset = tf.data.Dataset.from_generator(test_generator(val_df, root, normalization_layer),
                                                  output_types = ({"input_video": tf.float32, "trace_low": tf.float32, 'trace_high': tf.float32, 'frame_low': tf.int16, 'frame_high': tf.int16, 'file': tf.string}),
                                                  output_shapes = ({"input_video": (None, *shape), "trace_low": shape, "trace_high": shape, 'frame_low':(), 'frame_high':(), 'file': ()}))
    if get_train:
        train_dataset = tf.data.Dataset.from_generator(test_generator(train_df, root, normalization_layer),
                                                  output_types = ({"input_video": tf.float32, "trace_low": tf.float32, 'trace_high': tf.float32, 'frame_low': tf.int16, 'frame_high': tf.int16, 'file': tf.string}),
                                                  output_shapes = ({"input_video": (None, *shape), "trace_low": shape, "trace_high": shape, 'frame_low':(), 'frame_high':(), 'file': ()}))
        return test_dataset, train_dataset
    
    
    return test_dataset




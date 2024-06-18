import tensorflow as tf
import os


def get_data(root):
    train_dataset = tf.data.Dataset.load(os.path.join(root, 'ACDC_train'))
    test_dataset = tf.data.Dataset.load(os.path.join(root, 'ACDC_test'))
    return train_dataset, test_dataset

def get_dataset(root):
    train_dataset, test_dataset = get_data(root)
    train_dataset = train_dataset.map(lambda x: {'input_video': x['input_video'],
                                                 'input_ref': x['input_ref'],
                                                 'input_mask': tf.zeros(x['input_video'].shape[0], dtype=tf.bool)})
    
    test_dataset = test_dataset.map(lambda x: {'input_video': x['input_video'],
                                                 'input_ref': x['input_ref'],
                                                 'input_mask': tf.zeros(x['input_video'].shape[0], dtype=tf.bool)})
    
    return train_dataset, test_dataset, len(train_dataset)


def get_eval_data(root):
    train_dataset, test_dataset = get_data(root)
    return train_dataset, test_dataset
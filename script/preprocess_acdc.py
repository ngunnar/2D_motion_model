from collections import defaultdict
import SimpleITK as sitk
from scipy.ndimage import zoom
import os
import glob
import tqdm
import numpy as np
import tensorflow as tf
import argparse

def normalize_img(img):
    min_val = np.min(img)
    max_val = np.max(img)
    return (img - min_val)/(max_val - min_val)

def get_data(patients):
    data = []

    for patient in patients:
        files = glob.glob(os.path.join(patient, '*.nii.gz'))

        img_4D = sitk.ReadImage(files[0])        
        img4D_np = sitk.GetArrayFromImage(img_4D)

        img4D_np = normalize_img(zoom(img4D_np, (35/img4D_np.shape[0], 1, img_4D.GetSpacing()[0]/1.5, img_4D.GetSpacing()[1]/1.5), order=1))
        T_org = int(files[-1].split('_')[-2][5:]) - 1
        T = int(round(T_org*35/img4D_np.shape[0]))
        patient = os.path.basename(files[0]).split('_')[0]
        gt_frame1 = sitk.GetArrayFromImage(sitk.ReadImage(files[2]))
        gt_frame1 = zoom(gt_frame1, (1, img_4D.GetSpacing()[0]/1.5, img_4D.GetSpacing()[1]/1.5), order=1)
        gt_frameT = sitk.GetArrayFromImage(sitk.ReadImage(files[-1]))
        gt_frameT = zoom(gt_frameT, (1, img_4D.GetSpacing()[0]/1.5, img_4D.GetSpacing()[1]/1.5), order=1)

        nonzero_coords = np.argwhere(np.sum(gt_frame1>0, axis=(0))>0)
        centroid = nonzero_coords.mean(axis=0).astype('int')
        img4D_np = img4D_np[:,:,centroid[0]-128//2:centroid[0]+128//2,centroid[1]-128//2:centroid[1]+128//2]
        gt_frame1 = gt_frame1[:,centroid[0]-128//2:centroid[0]+128//2,centroid[1]-128//2:centroid[1]+128//2]
        gt_frameT = gt_frameT[:,centroid[0]-128//2:centroid[0]+128//2,centroid[1]-128//2:centroid[1]+128//2]

        for z in tqdm.tqdm(range(img4D_np.shape[1])):
            if (np.alltrue([np.sum(gt_frame1[z,...]==i) > 80 for i in range(3)])) and (np.alltrue([np.sum(gt_frameT[z,...]==i) > 80 for i in range(3)])):
                d = defaultdict() 
                d["T"] = T
                d["z"] = z
                d["input_ref"] = img4D_np[0,z,...] 
                d["input_video"] = img4D_np[:,z,...] 
                d['contour_0'] = gt_frame1[z,...]
                d['contour_T'] = gt_frameT[z,...]
                d['patient'] = patient
                data.append(d)
    return data

def data_generator(data):
    def generator():
        for item in data:
            yield item
    return generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('-root', '--root', type=str, help='path to ACDC data')
    args = parser.parse_args()
    root = args.root
    training_patients = glob.glob(os.path.join(root, 'training/patient*'))
    testing_patients = glob.glob(os.path.join(root, 'testing/patient*'))
    
    output_signature = {
        'T': tf.TensorSpec(shape=(), dtype=tf.int32),
        'z': tf.TensorSpec(shape=(), dtype=tf.int32),
        'input_ref': tf.TensorSpec(shape=(128,128), dtype=tf.float32),
        'input_video': tf.TensorSpec(shape=(35,128,128), dtype=tf.float32),
        'contour_0': tf.TensorSpec(shape=(128,128), dtype=tf.int32),
        'contour_T': tf.TensorSpec(shape=(128,128), dtype=tf.int32),
        'patient': tf.TensorSpec(shape=(), dtype=tf.string)
    }


    training_data = get_data(training_patients)
    training_dataset = tf.data.Dataset.from_generator(data_generator(training_data), output_signature=output_signature)
    training_dataset.save(os.path.join(root, 'ACDC_train'))

    testing_data = get_data(testing_patients)    
    testing_dataset = tf.data.Dataset.from_generator(data_generator(testing_data), output_signature=output_signature)
    testing_dataset.save(os.path.join(root, 'ACDC_test'))


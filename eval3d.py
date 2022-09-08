import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tqdm

import numpy as np
import SimpleITK as sitk
import tensorflow as tf


# Set dataset directory
data_root = '/vol/biodata/data/BTCV/Abdomen/IsotropicData/Training'
files = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010',
         '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029', '0030',
         '0031', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0039', '0040']
# train_images = [os.path.join(data_root, f'img/img{file}.nii.gz') for file in files[:25]]
# train_labels = [os.path.join(data_root, f'label/label{file}.nii.gz') for file in files[:25]]
test_images = [os.path.join(data_root, f'img/img{file}.nii.gz') for file in files[25:]]
test_labels = [os.path.join(data_root, f'label/label{file}.nii.gz') for file in files[25:]]

# # # Set dataset directory
# data_root = '/vol/biodata/data/Pancreas-CT/isotropic'
# files = ['0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011',
#          '0012', '0013', '0014', '0016', '0017', '0018', '0019', '0020', '0021', '0022',
#          '0024', '0026', '0027', '0028', '0029', '0030', '0031', '0032', '0033', '0034',
#          '0035', '0038', '0039', '0040', '0041', '0042', '0043', '0044', '0045', '0046',
#          '0047', '0048']
# # train_images = [os.path.join(data_root, f'image/PANCREAS_{file}.nii.gz') for file in files[:35]]
# # train_labels = [os.path.join(data_root, f'label/label{file}.nii.gz') for file in files[:35]]
# test_images = [os.path.join(data_root, f'image/PANCREAS_{file}.nii.gz') for file in files[35:]]
# test_labels = [os.path.join(data_root, f'label/label{file}.nii.gz') for file in files[35:]]


label_classes = {
    1: 'Spleen',
    2: 'Right Kidney',  # Not in Pancreas-CT
    3: 'Left Kidney',
    4: 'Gallbladder',
    5: 'Esophagus',
    6: 'Liver',
    7: 'Stomach',
    8: 'Aorta',  # Not in Pancreas-CT
    9: 'Inferior Vena Cava',  # Not in Pancreas-CT
    10: 'Portal and Splenic Vein',  # Not in Pancreas-CT
    11: 'Pancreas',
    12: 'Right Adrenal Gland',  # Not in Pancreas-CT
    13: 'left Adrenal Gland',  # Not in Pancreas-CT
    14: 'Duodenum',
}


a=1


# This code is based on
# https://gist.github.com/aewhite/14db960f9e832bce4041bf185cdc9615
def extract_volume_patches(images, patch_size=128, stride=120):
    return tf.extract_volume_patches(
        input=images,
        ksizes=[1, patch_size, patch_size, patch_size, 1],
        strides=[1, stride, stride, stride, 1],
        padding='SAME')


@tf.function
def extract_patches_inverse(shape, patches):
    _x = tf.zeros(shape)
    _y = extract_volume_patches(_x)
    grad = tf.gradients(_y, _x)[0]
    return tf.gradients(_y, _x, grad_ys=patches)[0] / grad


# Code adapted from "Generalized dice loss for multi-class segmentation"
# https://github.com/keras-team/keras/issues/9395#issuecomment-370971561
def dice_coef(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient function; ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersect = tf.keras.backend.sum(y_true_f * y_pred_f, axis=-1)
    denom = tf.keras.backend.sum(y_true_f + y_pred_f, axis=-1)
    return tf.keras.backend.mean((2. * intersect / (denom + smooth)))


# Load Segmentation Model
model = tf.keras.models.load_model('checkpoints/3dresunet_btcv.tf', compile=False)  # BTCV model
# model = tf.keras.models.load_model('checkpoints/3dresunet_fine2.tf', compile=False)  # Pancreas-CT model


# Run Evaluation
for image_file, label_file in zip(test_images, test_labels):

    print(f'--- Running eval on {os.path.basename(image_file)} ---')

    image_itk = image_itk_orig = sitk.ReadImage(image_file)
    image_itk = sitk.IntensityWindowing(image_itk, -500., 500)
    image_arr = sitk.GetArrayFromImage(image_itk).astype('uint8')
    image_arr = tf.image.convert_image_dtype(image_arr, tf.float32)[None, ..., None]

    image_shape = image_arr.shape
    patches = extract_volume_patches(image_arr, 128, 120)
    patches_shape = patches.shape
    patches = tf.reshape(patches, shape=(-1, 128, 128, 128, 1))

    y_pred_all = []
    for patch in tqdm.tqdm(patches):
        y_pred_all.append(model(patch[None, ...]))
    y_pred_all = np.concatenate(y_pred_all, axis=0)

    logits = []
    for c in tqdm.tqdm(range(model.output_shape[-1])):
        t = y_pred_all[..., [c]]
        t = tf.reshape(t, shape=patches_shape)
        t = extract_patches_inverse(image_shape, t)
        logits.append(t)
    logits = tf.concat(logits, axis=-1)[0]

    label_itk = sitk.ReadImage(label_file)
    label_arr = sitk.GetArrayFromImage(label_itk).astype('uint8')
    label_arr = np.eye(model.output_shape[-1])[label_arr]

    for i in range(1, model.output_shape[-1]):
        print(f'{label_classes[i]}: {dice_coef(label_arr[..., i], logits[..., i]):02f}')

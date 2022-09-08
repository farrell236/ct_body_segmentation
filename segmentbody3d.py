import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tqdm
import argparse
import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from scipy import ndimage
from typing import Iterable


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


def resample(itk_image: sitk.Image,
             new_spacing: Iterable[float],
             method: int = sitk.sitkLinear,
             outside_val: float = 0
             ) -> sitk.Image:
    shape = itk_image.GetSize()
    spacing = itk_image.GetSpacing()
    output_shape = tuple(int(round(s * os / ns)) for s, os, ns in zip(shape, spacing, new_spacing))
    return sitk.Resample(
        itk_image,
        output_shape,
        sitk.Transform(),
        method,
        itk_image.GetOrigin(),
        new_spacing,
        itk_image.GetDirection(),
        outside_val,
        sitk.sitkFloat32,
    )


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Body Organ Segmentation for CT Images')
    parser.add_argument('-i', '--input_fn', help='Input CT Volume')
    parser.add_argument('-o', '--output_fn', help='Output Segmentation')
    parser.add_argument('-m', '--model_fn', help='Trained Model')
    parser.add_argument('-minW', '--min_window', help='Intensity Window minimum value', default=-500.)
    parser.add_argument('-maxW', '--max_window', help='Intensity Window maximum value', default=500.)
    parser.add_argument('-iso', '--isotropic', help='Input Volume is Isotropic', action='store_true', default=False)
    parser.add_argument('-p', '--patch_size', help='3D Patch Division (patch)', default=128)
    parser.add_argument('-s', '--stride', help='3D Patch Division (stride)', default=120)
    parser.add_argument('-v', '--verbose', help='Verbose Output', action='store_true', default=False)
    parser.add_argument('-b', '--batch_size', help='Inference Batch Size', default=4)
    args = vars(parser.parse_args())

    ##### Debug Overrides
    # args['input_fn'] = '/vol/biodata/data/BTCV/Abdomen/IsotropicData/Testing/img/img0069.nii.gz'
    # args['input_fn'] = '/vol/biodata/data/Pancreas-CT/isotropic/image/PANCREAS_0042.nii.gz'
    # args['input_fn'] = '/vol/biodata/data/Pancreas-CT/isotropic/image/PANCREAS_0047.nii.gz'
    # args['input_fn'] = '/vol/biodata/data/Pancreas-CT/Pancreas-CT-nifti/PANCREAS_0047.nii.gz'
    # args['model_fn'] = 'checkpoints/3dresunet_fine2.tf'
    # args['isotropic'] = True
    # args['verbose'] = True

    # Load ITK image
    image_itk = image_itk_orig = sitk.ReadImage(args['input_fn'])

    # Resample image to isotropic (if required)
    if not args['isotropic']:
        if args['verbose']: print(f'Resampling image to isotropic...')
        image_itk = resample(image_itk, new_spacing=(1., 1., 1.,))

    image_itk = sitk.IntensityWindowing(image_itk, args['min_window'], args['max_window'])
    image_arr = sitk.GetArrayFromImage(image_itk).astype('uint8')
    image_arr = tf.image.convert_image_dtype(image_arr, tf.float32)[None, ..., None]

    # Load trained body segmentation model
    model = tf.keras.models.load_model(args['model_fn'], compile=False)
    n_classes = model.output_shape[-1]
    if args['verbose']: print(f'Loaded Segmentation Model: {args["model_fn"]}')

    # Divide input volume into sub-patches
    image_shape = image_arr.shape
    patches = extract_volume_patches(image_arr, args['patch_size'], args['stride'])
    patches_shape = patches.shape
    patches = tf.reshape(patches, shape=(-1, args['patch_size'], args['patch_size'], args['patch_size'], 1))

    # Run model on sub-patches
    if args['verbose']: print(f'Running Model Inference...')
    y_pred = model.predict(patches, verbose=args['verbose'], batch_size=args['batch_size'])

    # Combine sub-patches to original volume dimension
    if args['verbose']: print(f'Reconstructing Volume...')
    logits = []
    for c in tqdm.tqdm(range(n_classes) if args['verbose'] else range(n_classes)):
        t = y_pred[..., [c]]
        t = tf.reshape(t, shape=patches_shape)
        t = extract_patches_inverse(image_shape, t)
        logits.append(t)
    logits = tf.concat(logits, axis=-1)
    logits = tf.argmax(logits, axis=-1)
    logits = logits.numpy()[0]

    # # Post-process and clean predicted segmentations
    # cleaned_mask = np.zeros_like(logits)
    # for label in range(1, n_classes):
    #     if args['verbose']: print(f'Cleaning Label: {label_classes[label]}')
    #     binary_img = logits == label
    #     label_im, nb_labels = ndimage.label(binary_img)
    #     sizes = ndimage.sum(binary_img, label_im, range(nb_labels + 1))
    #     mask = sizes > max(sizes) - 1
    #     cleaned_mask += mask[label_im] * label

    # Resample mask to original spacing (if required) and copy image header
    pred_itk = sitk.GetImageFromArray(logits.astype('int16'))
    if not args['isotropic']:
        if args['verbose']: print(f'Resampling mask to: {image_itk_orig.GetSpacing()}...')
        pred_itk = resample(pred_itk, new_spacing=image_itk_orig.GetSpacing(), method=sitk.sitkNearestNeighbor)
    pred_itk.CopyInformation(image_itk_orig)

    # Save predicted segmentation mask to disk
    if args['output_fn'] is None:
        fn, ext = os.path.basename(args['input_fn']).split('.', 1)
        sitk.WriteImage(pred_itk, f'{fn}_mask.{ext}')
        if args['verbose']: print(f'Segmentation mask saved to: {fn}_mask.{ext}')
    else:
        sitk.WriteImage(pred_itk, args['output_fn'])
        if args['verbose']: print(f'Segmentation mask saved to: {args["output_fn"]}')

import glob
import os
import tqdm
import SimpleITK as sitk

from typing import Iterable

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


# data_root = '/vol/biodata/data/BTCV/Abdomen/RawData/Training'
# files = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010',
#          '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029', '0030',
#          '0031', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0039', '0040']
# train_images = [os.path.join(data_root, f'img/img{file}.nii.gz') for file in files]
# train_labels = [os.path.join(data_root, f'label/label{file}.nii.gz') for file in files]
#
# for file in tqdm.tqdm(train_images):
#     image_itk = sitk.ReadImage(file)
#     image_itk = resample(image_itk, new_spacing=(1., 1., 1.,))
#     sitk.WriteImage(image_itk, f'resampled/{os.path.basename(file)}')
#
# for file in tqdm.tqdm(train_labels):
#     image_itk = sitk.ReadImage(file)
#     image_itk = resample(image_itk, new_spacing=(1., 1., 1.,), method=sitk.sitkNearestNeighbor)
#     sitk.WriteImage(image_itk, f'resampled/{os.path.basename(file)}')

a=1

# files = glob.glob('/vol/biodata/data/BTCV/Abdomen/RawData/Testing/img/*')
# for file in tqdm.tqdm(files):
#     image_itk = sitk.ReadImage(file)
#     image_itk = resample(image_itk, new_spacing=(1., 1., 1.,))
#     sitk.WriteImage(image_itk, f'resampled/{os.path.basename(file)}')

a=1

files = glob.glob('/vol/biodata/data/Pancreas-CT/TCIA_multiorgan_labels/*')
# files = glob.glob('/vol/biodata/data/BTCV/label_btcv_multiorgan/*')
for file in tqdm.tqdm(files):
    image_itk = sitk.ReadImage(file)
    image_itk = resample(image_itk, new_spacing=(1., 1., 1.,), method=sitk.sitkNearestNeighbor)
    sitk.WriteImage(image_itk, f'resampled/{os.path.basename(file)}')

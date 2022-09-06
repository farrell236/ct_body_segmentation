import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import SimpleITK as sitk
import tensorflow as tf

import wandb
from wandb.keras import WandbCallback

wandb.init(project="BodySegmenter", entity="farrell236")


# Training Parameters
epochs = 2000
batch_size = 4
patch_size = 128
learning_rate = 1e-4
n_classes = 14

# # Set dataset directory
# data_root = '/vol/biodata/data/BTCV/Abdomen/IsotropicData/Training'
# files = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010',
#          '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029', '0030',
#          '0031', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0039', '0040']
# train_images = [os.path.join(data_root, f'img/img{file}.nii.gz') for file in files[:25]]
# train_labels = [os.path.join(data_root, f'label/label{file}.nii.gz') for file in files[:25]]
# test_images = [os.path.join(data_root, f'img/img{file}.nii.gz') for file in files[25:]]
# test_labels = [os.path.join(data_root, f'label/label{file}.nii.gz') for file in files[25:]]

# Set dataset directory
data_root = '/vol/biodata/data/Pancreas-CT/isotropic'
files = ['0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011',
         '0012', '0013', '0014', '0016', '0017', '0018', '0019', '0020', '0021', '0022',
         '0024', '0026', '0027', '0028', '0029', '0030', '0031', '0032', '0033', '0034',
         '0035', '0038', '0039', '0040', '0041', '0042', '0043', '0044', '0045', '0046',
         '0047', '0048']
train_images = [os.path.join(data_root, f'image/PANCREAS_{file}.nii.gz') for file in files[:35]]
train_labels = [os.path.join(data_root, f'label/label{file}.nii.gz') for file in files[:35]]
test_images = [os.path.join(data_root, f'image/PANCREAS_{file}.nii.gz') for file in files[35:]]
test_labels = [os.path.join(data_root, f'label/label{file}.nii.gz') for file in files[35:]]

a=1

def load_itk_volume(image_fn, label_fn):

    image_itk = sitk.ReadImage(bytes.decode(image_fn.numpy(), 'utf-8'))
    # image_itk = sitk.ReadImage(image_fn)
    image_itk = sitk.IntensityWindowing(image_itk, -500., 500.)
    image_arr = sitk.GetArrayFromImage(image_itk).astype('uint8')[..., None]
    image_arr = tf.image.convert_image_dtype(image_arr, tf.float32)

    label_itk = sitk.ReadImage(bytes.decode(label_fn.numpy(), 'utf-8'))
    # label_itk = sitk.ReadImage(label_fn)
    label_arr = sitk.GetArrayFromImage(label_itk).astype('float32')[..., None]

    return image_arr, label_arr


def random_crop(image, label):
    stacked_image = tf.stack([image, label], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, patch_size, patch_size, patch_size, 1])
    return cropped_image[0], cropped_image[1]


def extract_patches(image, label):
    stacked_image = tf.stack([image, label], axis=0)
    patches = tf.extract_volume_patches(
        input=stacked_image,
        ksizes=[1, patch_size, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, patch_size, 1],
        padding='SAME')  # [2, n_patch_d, n_patch_h, n_patch_w, n_patch_voxels]
    patches = tf.reshape(patches, [2, -1, patch_size ** 3])  # [2, n_patchs, n_patch_voxels]
    patches = tf.reshape(patches, [2, -1, patch_size, patch_size, patch_size, 1])  # [2, n_patchs, 128, 128, 128, 1]
    return patches[0], patches[1]


def ensure_shape(image, label):
    image = tf.ensure_shape(image, (None, None, None, None))
    label = tf.ensure_shape(label, (None, None, None, None))
    return image, label


a=1


train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.map(map_func=lambda x, y: tf.py_function(
    func=load_itk_volume, inp=[x, y], Tout=[tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(map_func=random_crop, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(map_func=ensure_shape, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

valid_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
valid_dataset = valid_dataset.map(map_func=lambda x, y: tf.py_function(
    func=load_itk_volume, inp=[x, y], Tout=[tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.map(map_func=extract_patches, num_parallel_calls=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
valid_dataset = valid_dataset.map(map_func=ensure_shape, num_parallel_calls=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(batch_size)


a=1


# Code adapted from "Generalized dice loss for multi-class segmentation"
# https://github.com/keras-team/keras/issues/9395#issuecomment-370971561
def dice_coef(y_true, y_pred, num_classes=15, smooth=1e-7):
    '''
    Dice coefficient function; ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = tf.keras.backend.flatten(tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)[..., 1:])
    y_pred_f = tf.keras.backend.flatten(y_pred[..., 1:])
    intersect = tf.keras.backend.sum(y_true_f * y_pred_f, axis=-1)
    denom = tf.keras.backend.sum(y_true_f + y_pred_f, axis=-1)
    return tf.keras.backend.mean((2. * intersect / (denom + smooth)))


def dice_coef_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef(y_true, y_pred)


def combined_loss(y_true, y_pred, alpha=0.5):
    sBCE = tf.keras.losses.SparseCategoricalCrossentropy()
    return (1 - alpha) * sBCE(y_true, y_pred) + alpha * dice_coef_loss(y_true, y_pred)



# from residual_unet import get_network
#
# widths = [16, 32, 48, 64]
# block_depth = 4
# model = get_network((None, None, None, 1), widths, block_depth, n_classes=15)
model = tf.keras.models.load_model('checkpoints/3dresunet.tf', compile=False)




a=1


# Train with combined Binary Crossentropy and Dice Loss
csv_logger = tf.keras.callbacks.CSVLogger(f'logs/training_dice.log')
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'checkpoints/3dresunet_fine2.tf',
    monitor='val_dice_coef', mode='max', verbose=1,
    save_best_only=True)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=combined_loss,
    metrics=[dice_coef])
model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=epochs,
    callbacks=[checkpoint, csv_logger, WandbCallback()])

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import SimpleITK as sitk
import tensorflow as tf

from DeeplabV3Plus import DeeplabV3Plus


# Training Parameters
epochs = 1000
batch_size = 16
buffer_size = 4000
learning_rate = 1e-4

# Set dataset directory
data_root = '/mnt/nas_houbb/users/Benjamin/data/BTCV/RawData/Training'
files = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010',
         '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029', '0030',
         '0031', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0039', '0040']
train_images = [os.path.join(data_root, f'img/img{file}.nii.gz') for file in files[:25]]
train_labels = [os.path.join(data_root, f'label/label{file}.nii.gz') for file in files[:25]]
test_images = [os.path.join(data_root, f'img/img{file}.nii.gz') for file in files[25:]]
test_labels = [os.path.join(data_root, f'label/label{file}.nii.gz') for file in files[25:]]

a=1

def load_itk_volume(image_fn, label_fn):

    image_itk = sitk.ReadImage(bytes.decode(image_fn.numpy(), 'utf-8'))
    # image_itk = sitk.ReadImage(image_fn)
    image_itk = sitk.IntensityWindowing(image_itk, -1000., 400.)
    image_arr = sitk.GetArrayFromImage(image_itk).astype('uint8')[..., None]
    image_arr = tf.image.convert_image_dtype(image_arr, tf.float32)

    label_itk = sitk.ReadImage(bytes.decode(label_fn.numpy(), 'utf-8'))
    # label_itk = sitk.ReadImage(label_fn)
    label_arr = sitk.GetArrayFromImage(label_itk).astype('uint8')[..., None]

    return image_arr, label_arr


def ensure_shape(image, label):
    image = tf.ensure_shape(image, (None, None, None))
    label = tf.ensure_shape(label, (None, None, None))
    return image, label


a=1


train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.map(map_func=lambda x, y: tf.py_function(
    func=load_itk_volume, inp=[x, y], Tout=[tf.float32, tf.uint8]), num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
train_dataset = train_dataset.map(map_func=ensure_shape, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=buffer_size)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

valid_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
valid_dataset = valid_dataset.map(map_func=lambda x, y: tf.py_function(
    func=load_itk_volume, inp=[x, y], Tout=[tf.float32, tf.uint8]), num_parallel_calls=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
valid_dataset = valid_dataset.map(map_func=ensure_shape, num_parallel_calls=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(batch_size)


a=1


# Code adapted from "Generalized dice loss for multi-class segmentation"
# https://github.com/keras-team/keras/issues/9395#issuecomment-370971561
def dice_coef(y_true, y_pred, num_classes=14, smooth=1e-7):
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


a=1


# Define DeepLabV3+ Model
model = DeeplabV3Plus(image_size=(512, 512, 1), num_classes=14)
# model = tf.keras.models.load_model('checkpoints/DeeplabV3Plus.tf')


a=1


# Train with combined Binary Crossentropy and Dice Loss
csv_logger = tf.keras.callbacks.CSVLogger(f'logs/training_2d.log')
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'checkpoints/DeeplabV3Plus.tf',
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
    callbacks=[checkpoint, csv_logger])

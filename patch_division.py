# This code is based on
# https://gist.github.com/aewhite/14db960f9e832bce4041bf185cdc9615

import numpy as np
import tensorflow as tf


def extract_volume_patches(images, patch_size=128, stride=110):
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


if __name__ == "__main__":

    # load initial image
    image = np.random.rand(1, 138, 442, 442, 1).astype('float32')

    a=1

    # Extract patches using "extract_image_patches()"
    patches = extract_volume_patches(image)

    patches_shape = patches.shape

    patches_1 = tf.reshape(patches, shape=(-1, 128, 128, 128, 1))
    patches_2 = tf.reshape(patches_1, shape=patches_shape)

    images_reconstructed = extract_patches_inverse(image.shape, patches_2)  # Reconstruct Image

    a=1



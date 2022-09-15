import tensorflow as tf


class windowLevel(tf.keras.layers.Layer):
    def __init__(self, name='windowLevelLayer'):
        # initialize the layer
        super().__init__(name=name)
        # create the constant bias initializer with initial window setting
        self.output_bias = tf.keras.initializers.Constant([-500., 500.])

    def build(self, input_shape):
        # get the batch size, height, width and channel size of the input
        # (B, H, W, C) = input_shape

        # define the windowing network
        self.regressorNet = tf.keras.Sequential([
            tf.keras.applications.MobileNet(
                include_top=False, weights=None,
                input_shape=(None, None, 1), pooling='avg'),
            tf.keras.layers.Dense(
                units=2,
                kernel_initializer='zeros',
                bias_initializer=self.output_bias)
        ])

    def clip_and_normalize(self, inputs):
        (image, win) = inputs
        # image = tfp.math.clip_by_value_preserve_gradient(image, win[0], win[1])
        image = tf.clip_by_value(image, win[0], win[1])
        image = (image - win[0]) / (win[1] - win[0])
        return image

    def call(self, x):
        # estimate the window levels
        windowLevel = self.regressorNet(x)

        # apply clip_and_normalize function on every element in batch
        x = tf.map_fn(self.clip_and_normalize, (x, windowLevel), dtype=tf.float32)

        # return the windowed image
        return x

from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, BatchNormalization, Add, LeakyReLU
from keras.models import Model
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from functools import partial
import keras
from keras import layers

from keras import backend as K


# keras.mixed_precision.set_global_policy("mixed_bfloat16")

# def MDE(true: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
#    """Mean euclidean Distance Error. Similar to RMSE."""
#    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(true - pred), axis=-1)))


def mean_euclidean_distance_error(y_true, y_pred):
    """
    Computes the Mean Euclidean Distance Error between y_true and y_pred.

    Args:
    - y_true: True labels, a TensorFlow/Theano tensor of the same shape as y_pred.
    - y_pred: Predictions, a TensorFlow/Theano tensor.

    Returns:
    - The mean Euclidean distance error between y_true and y_pred.
    """
    # Calculate the Euclidean distance
    distance = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

    # Return the mean distance
    return K.mean(distance)


# def residual_block(x, channels, down_sample=False):
#     strides = [2, 1] if down_sample else [1, 1]

#     KERNEL_SIZE = (3, 3)
#     # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
#     INIT_SCHEME = "he_normal"

#     if down_sample:
#         # perform down sampling using stride of 2, according to [1].
#         shortcut = Conv2D(channels, kernel_size=(1, 1), strides=2, padding="same", kernel_initializer=INIT_SCHEME)(x)
#         shortcut = BatchNormalization()(shortcut)
#     else:
#         shortcut = x

#     x = layers.Conv2D(channels, kernel_size=KERNEL_SIZE, strides=strides[0], padding="same", kernel_initializer=INIT_SCHEME)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     x = layers.Conv2D(channels, kernel_size=KERNEL_SIZE, strides=strides[1], padding="same", kernel_initializer=INIT_SCHEME)(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.Add()([x, shortcut])
#     x = layers.ReLU()(x)

#     return x


class ResnetBlock(Model):
    """A standard resnet block."""

    def __init__(self, channels: int, down_sample=False):
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(
            self.__channels,
            strides=self.__strides[0],
            kernel_size=KERNEL_SIZE,
            padding="same",
            kernel_initializer=INIT_SCHEME,
        )
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(
            self.__channels,
            strides=self.__strides[1],
            kernel_size=KERNEL_SIZE,
            padding="same",
            kernel_initializer=INIT_SCHEME,
        )
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same"
            )
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


def create_network():
    # return __Pirnat1G()
    model = keras.Sequential()

    model.add(layers.Input(shape=(2, 16, 924)))
    model.add(layers.Permute((2, 3, 1)))

    model.add(layers.Conv2D(32, kernel_size=(1, 7), strides=(1, 3), padding="same", kernel_initializer="he_normal"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(1, 4)))

    model.add(ResnetBlock(32))
    model.add(ResnetBlock(32))

    model.add(ResnetBlock(64, down_sample=True))
    model.add(ResnetBlock(64))

    model.add(ResnetBlock(128, down_sample=True))
    model.add(ResnetBlock(128))

    model.add(ResnetBlock(256, down_sample=True))
    model.add(ResnetBlock(256))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Flatten())

    model.add(layers.Dense(1000, activation=LeakyReLU(alpha=0.001)))
    model.add(layers.Dense(2))

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, name="sgd"),
        loss=mean_euclidean_distance_error,
    )

    return model


Pirnat1G = partial(
    KerasRegressor,
    model=create_network,
    random_state=42,
    batch_size=32,
    epochs=100,
    callbacks=[EarlyStopping(monitor="val_loss", patience=10)],  # , ReduceLROnPlateau(patience=4)],
    validation_split=0.2,
)

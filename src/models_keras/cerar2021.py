from tensorflow import keras
from keras import backend as K, layers, optimizers, callbacks
from .pirnat2022 import mean_euclidean_distance_error

from scikeras.wrappers import KerasRegressor
from functools import partial


def _resnet_block(input, filters, kernel=3, strides=1, activation="relu"):
    common = dict(use_bias=False, padding="same", kernel_initializer="he_normal")

    if isinstance(strides, int):
        strides = (strides, strides)

    if K.int_shape(input)[-1] != filters:
        shortcut = input
        shortcut = layers.Conv2D(filters, 1, strides=strides, **common)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = layers.MaxPooling2D(1, strides=strides)(input) if strides != (1, 1) else input

    x = input
    x = layers.Conv2D(filters, kernel, strides=strides, **common)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv2D(filters, kernel, **common)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.Activation(activation)(x)
    return x


def resnet_block(x, filters, kernel, stride1=1, blocks=3):
    x = _resnet_block(x, filters, kernel, strides=stride1)
    for _ in range(1, blocks):
        x = _resnet_block(x, filters, kernel, strides=1)
    return x


def build_model():
    # conv_kwargs = dict(padding='same', kernel_initializer='he_normal', use_bias=False)
    fc_kwargs = dict(kernel_initializer="he_normal", use_bias=True)
    bn_kwargs = dict(axis=-1, epsilon=1.001e-5)

    x = inputs = layers.Input(shape=(2, 16, 924), name="h")
    x = layers.Permute((2, 3, 1))(x)

    filters = 16
    growth_rate = 1.5

    for _ in range(4):
        x = resnet_block(x, filters, (1, 7), stride1=(1, 3), blocks=3)
        filters *= growth_rate

    x = layers.Flatten()(x)

    x = layers.Dense(1024, **fc_kwargs)(x)
    x = layers.BatchNormalization(**bn_kwargs)(x)
    x = layers.Activation("relu")(x)

    x = layers.Dense(2)(x)

    model = keras.Model(inputs, x)
    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9),
        loss=mean_euclidean_distance_error,
    )

    return model


Cerar2021 = partial(
    KerasRegressor,
    model=build_model,
    random_state=42,
    batch_size=32,
    epochs=500,
    callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=10)],  # , ReduceLROnPlateau(patience=4)],
    validation_split=0.2,
)

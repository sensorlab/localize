import autokeras as ak
import tensorflow as tf

from .utils import utils


# Custom Block class generator
class GenericCustomBlock(ak.Block):
    """
    A subclass of the autokerass block used to create custom blocks from
    a layer config dictionary.
    """

    def __init__(self, layer_config: dict, **kwargs):
        super().__init__(**kwargs)
        self.layer_config = layer_config

    def build(self, hp, inputs=None):
        """Dynamically build the layers specified in the layer_config."""
        x = inputs[0] if isinstance(inputs, list) else inputs

        for cnf in self.layer_config:
            layer_module = cnf.get("module", "keras.layers")  # Default to keras.layers if module not specified
            layer_class = cnf.get("class")
            layer_args = utils.parse_args(cnf.get("args", {}), hp)

            # Dynamically get the layer class and instantiate it with parsed args
            layer = utils.get_class(layer_module, layer_class)(**layer_args)

            # Apply the layer to the input
            x = layer(x)

        return x


class SlidingWindowBlock(ak.Block):
    def __init__(self, lookback=10, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback

    def build(self, hp, inputs=None):
        inp = inputs[0] if isinstance(inputs, list) else inputs

        # Lambda layer to apply sliding window transformation
        def sliding_window_reshape(x):
            # Add a new axis to represent timesteps
            return tf.keras.backend.tile(tf.expand_dims(x, axis=1), [1, self.lookback, 1])

        # Lambda layer with explicit output shape
        x = tf.keras.layers.Lambda(sliding_window_reshape, output_shape=(self.lookback, inp.shape[-1]))(inp)

        return x


def mean_euclidean_distance_error(y_true, y_pred):
    """Calculates the mean Euclidean distance between y_true and y_pred."""
    y_true = tf.cast(y_true, "float64")
    y_pred = tf.cast(y_pred, "float64")
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1)))

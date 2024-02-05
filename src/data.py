import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, Iterable


class ReshapeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sample_shape: Iterable[int] = None, copy=True) -> None:
        super().__init__()

        self.new_shape = sample_shape
        self.make_copy = copy

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame], copy=None):
        n_samples = len(X)

        if hasattr(X, "iloc"):
            # This is either Pandas or Polars DataFrame
            X = X.to_numpy()

        if isinstance(X, np.ndarray):
            if copy:
                X = X.copy()

            print((n_samples, *self.new_shape))

            if self.new_shape:
                X = X.reshape((n_samples, *self.new_shape))

            return X

        else:
            raise NotImplementedError

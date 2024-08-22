from typing import Iterable, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# class FlattenAndCombine(BaseEstimator, TransformerMixin):
#     def __init__(self, keys: Iterable[str]):
#         self.keys = keys

#     def fit(self, X, y=None, **fit_params):
#         return self

#     def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
#         # Input is expected to be dict
#         data = [X[key].reshape(shape=(len(X[key]), -1)) for key in self.keys]

#         pass

#         return super().fit_transform(X, y, **fit_params)


class ExtractByKeyword(BaseEstimator, TransformerMixin):
    def __init__(self, keyname: str):
        super().__init__()
        self.keyname = keyname

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X: Union[dict, np.ndarray, pd.DataFrame], copy=None):
        return X[self.keyname]


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

def replace_extreme_outliers(series: pd.Series, multiplier: float = 3.0) -> pd.Series:
    """
    Replaces extreme outliers in a pandas Series with interpolated values.
    
    Parameters:
    series (pd.Series): The input data series.
    multiplier (float): The multiplier for the IQR to define extreme outliers (default is 3.0).
    
    Returns:
    pd.Series: A series with extreme outliers replaced by interpolated values.
    """
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (series < lower_bound) | (series > upper_bound)
    
    series_cleaned = series.copy()
    
    # Replace outliers with NaN for interpolation
    series_cleaned[outliers] = np.nan
    series_cleaned = series_cleaned.interpolate(method='linear')
    
    # In case of any remaining NaNs at the beginning or end, forward/backward fill them
    series_cleaned = series_cleaned.bfill().ffill()
    
    return series_cleaned
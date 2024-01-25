"""
The `src` package is the root of this project.
"""
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
import json

from sklearn import model_selection
import numpy as np

# Project source code directory
SRC_PATH = Path(__file__).resolve().parents[0]

# Project root path
PROJECT_PATH = Path(__file__).resolve().parents[1]

# Model directory path
MODEL_PATH = PROJECT_PATH / "models"

# Reports directory path
REPORTS_PATH = PROJECT_PATH / "reports"

# Data paths
RAW_DATA_PATH = PROJECT_PATH / "data" / "raw"
INTERIM_DATA_PATH = PROJECT_PATH / "data" / "interim"
PROCESSED_DATA_PATH = PROJECT_PATH / "data" / "processed"


def load_params(params_path: Path) -> dict:
    with open(params_path) as fp:
        params = yaml.safe_load(fp)
        return params


def save_params(obj: object, params_path: Path):
    match params_path.suffix:
        case ".json":
            with open(params_path, mode="w") as fp:
                json.dump(obj, fp, sort_keys=True, indent=2)

        case ".pkl" | ".joblib":
            joblib.dump(obj, params_path)

        case ".yaml" | ".yml":
            with open(params_path, mode="w") as fp:
                yaml.safe_dump(obj, fp)

        case _:
            raise NotImplementedError


def load_data(path: Path) -> Any:
    match path.suffix:
        case ".pq" | ".parquet":
            return pd.read_parquet(path)
        case ".pkl" | ".joblib":
            return joblib.load(path)
        case _:
            raise NotImplementedError


def safe_indexing(X: Any, indices, *, axis=0):
    # Simplified version of

    if X is None:
        return X

    if hasattr(X, "iloc"):  # Most likely Pandas DataFrame
        return X.iloc[indices]

    if hasattr(X, "shape"):  # Most likely Numpy array
        return X[indices]

    raise NotImplementedError(f'Safe indexing not implemented for "{type(X)}"')


class PredefinedSplit(model_selection.BaseCrossValidator):
    """Simple cross-validator for predefined train-test splits."""

    def __init__(self, indices_pairs: list[tuple[np.ndarray, np.ndarray]]):
        self.idx_pairs = indices_pairs

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator"""
        return len(self.idx_pairs)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        for train_idx, test_idx in self.idx_pairs:
            yield train_idx, test_idx

"""
The `src` package is the root of this project.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn import model_selection


# Project source code directory
SRC_PATH = Path(__file__).resolve().parents[0]

# Project root path
PROJECT_PATH = Path(__file__).resolve().parents[1]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert arrays to lists

        elif np.issubdtype(obj, np.integer):
            return int(obj)  # Convert np integers to Python int

        elif np.issubdtype(obj, np.floating):
            return float(obj)  # Convert np floats to Python float

        elif np.issubdtype(obj, np.bool_):
            return bool(obj)  # Convert np bools to Python bool

        return json.JSONEncoder.default(self, obj)


def load_params(params_path: Path) -> dict:
    with open(params_path) as fp:
        params = yaml.safe_load(fp)
        return params


def save_params(obj: object, params_path: Path):
    match params_path.suffix:
        case ".json":
            with open(params_path, mode="w") as fp:
                json.dump(obj, fp, sort_keys=True, indent=2, cls=NumpyEncoder)

        case ".pkl" | ".joblib":
            joblib.dump(obj, params_path)

        case ".yaml" | ".yml":
            with open(params_path, mode="w") as fp:
                yaml.safe_dump(obj, fp)

        case _:
            raise NotImplementedError


def load_data(path: Path, mmap: bool = True) -> Any:
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

    # if isinstance(X, dict):
    #     print("Safe index on dicts")
    #     #indexed_dict = {key: value[indices] for key, value in X.items()}
    #     # Initialize an empty list to store the indexed dictionaries
    #     indexed_samples = []
    #     for idx in indices:
    #         # For each index, create a new dictionary where each key is a feature name
    #         # and its value is the feature value for the current index.
    #         sample_dict = {key: X[key][idx] for key in X}
    #         indexed_samples.append(sample_dict)
    #     return indexed_samples

    # subsets = {key: value[indices] for key, value in X.items()}
    # X = [for ]
    # return X

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


def empty_directory(directory_path: Path):
    """
    Remove all contents of a directory without deleting the directory itself.

    Args:
        directory_path (Path or str): The path to the directory you want to empty.
    """
    for item in directory_path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()  # Remove file or symlink
        elif item.is_dir():
            shutil.rmtree(item)  # Remove directory and its contents

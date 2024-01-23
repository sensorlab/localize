from pathlib import Path
from typing import Literal

import click
import joblib
import xgboost
from sklearn import ensemble, linear_model, model_selection, multioutput, neighbors
import numpy as np

from src import load_data, load_params


class CustomCrossValidation(model_selection.BaseCrossValidator):
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


@click.command()
@click.option(
    "--input",
    "--data",
    "input_dataset_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--split-indices",
    "split_indices_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
)
@click.option(
    "--algorithm",
    "--use",
    "algorithm",
    type=click.Choice(["linear", "rforest", "xgb", "xgbrf", "knn"], case_sensitive=False),
    required=True,
    # show_default=True,
    # help="Method to harmonize values",
)
@click.option(
    "--out-model",
    "--output-model",
    "--output",
    "model_output_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=True,
    # help="Input data, parquet",
)
# @click.option(
#     "--out-best-params",
#     "--output-best-params",
#     "best_params_output_path",
#     type=click.Path(dir_okay=False, writable=True, path_type=Path),
#     required=True,
#     # help="Input data, parquet",
# )
@click.option(
    "--params",
    "--model-params",
    "params_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
def cli(
    input_dataset_path: Path,
    split_indices_path: Path,
    model_output_path: Path,
    # best_params_output_path: Path,
    params_path: Path,
    algorithm: Literal["linear", "rforest", "xgb", "xgbrf", "knn"],
):
    params = load_params(params_path)
    random_state = params["seed"]
    mparams = params["models"]["regression"]

    # Load data
    features, targets = load_data(input_dataset_path)
    cv_indices = load_data(split_indices_path)
    cv = CustomCrossValidation(cv_indices)

    # Number of available CPU cores
    n_jobs = joblib.cpu_count(only_physical_cores=False)

    # Stratified KFold for evaluation
    match algorithm:
        case "linear":
            estimator = linear_model.LinearRegression()
        case "rforest":
            estimator = ensemble.RandomForestRegressor(random_state=random_state)
        case "xgb":
            estimator = xgboost.XGBRegressor(random_state=random_state)
        case "xgbrf":
            estimator = xgboost.XGBRFRegressor(random_state=random_state)
        case "knn":
            estimator = neighbors.KNeighborsRegressor()
        case _:
            raise NotImplementedError

    # Prepare hyper-parameters:
    hparams = mparams.get(algorithm, {})

    # Estimator expects `estimator__` prefix
    hparams = {f"estimator__{k}": v for k, v in hparams.items()}

    # Scoring metrics
    # scoring = {"mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error"}

    gridsearch = model_selection.GridSearchCV(
        estimator=multioutput.MultiOutputRegressor(estimator),
        param_grid=hparams,
        refit="mse",
        cv=cv,
    )

    with joblib.parallel_backend("loky", n_jobs=n_jobs):
        gridsearch.fit(features, targets)

    # Save best model
    best_model = gridsearch.best_estimator_
    joblib.dump(best_model, model_output_path)


if __name__ == "__main__":
    cli()

from pathlib import Path
from typing import Literal

import click
import joblib
import xgboost
from sklearn import ensemble, linear_model, model_selection, multioutput, neighbors

from src import load_data, load_params


class CustomCrossValidation(model_selection.BaseCrossValidator):
    """Simple passthrough cross validator for datasets with predefined splits."""

    def __init__(self, indices_pairs):
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
    show_default=True,
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
    params_path: Path,
    algorithm: Literal["linear", "rforest", "xgboost", "knn"],
):
    params = load_params(params_path)
    random_state = params["seed"]
    mparams = params["models"]["regression"]

    # Load data
    features, targets = load_data(input_dataset_path)
    cv_indices = load_data(split_indices_path)
    cv = CustomCrossValidation(cv_indices)

    # Number of available CPU cores
    n_jobs = joblib.cpu_count(only_physical_cores=True)

    # Stratified KFold for evaluation
    match algorithm:
        case "linear":
            estimator = linear_model.LinearRegression(n_jobs=n_jobs)
        case "rforest":
            estimator = ensemble.RandomForestRegressor(n_jobs=n_jobs, random_state=random_state)
        case "xgb":
            estimator = xgboost.XGBRegressor(n_jobs=n_jobs, random_state=random_state)
        case "xgbrf":
            estimator = xgboost.XGBRFRegressor(n_jobs=n_jobs, random_state=random_state)
        case "knn":
            estimator = neighbors.KNeighborsRegressor(n_jobs=n_jobs)
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
        # scoring=scoring,
        refit="mse",
        n_jobs=n_jobs,
        cv=cv,
    )

    gridsearch.fit(features, targets)

    # Save best parameters
    # best_idx = gridsearch.best_index_
    # results = gridsearch.cv_results_

    # scores = {
    #    "best_params": gridsearch.best_params_,
    #    "mse": -results["mean_test_mse"][best_idx],
    #    "mae": -results["mean_test_mae"][best_idx],
    # }

    # if metrics_output_path:
    #    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    #    with open(metrics_output_path, mode="w") as fp:
    #        json.dump(scores, fp)

    # Save best model
    best_model = gridsearch.best_estimator_
    joblib.dump(best_model, model_output_path)

    # TODO: save best parameters

    # outputs = []
    # for train_idx, test_idx in cv.split(features, targets):
    #     X_train = safe_indexing(features, train_idx)
    #     y_train = safe_indexing(targets, train_idx)

    #     X_test = safe_indexing(features, test_idx)
    #     y_test = safe_indexing(targets, test_idx)

    #     m = best_model.fit(X_train, y_train)
    #     y_pred = m.predict(X_test)

    #     outputs.append((y_test, y_pred))

    # joblib.dump(tuple(outputs), model_output_path)

    # model_output_path.parent.mkdir(parents=True, exist_ok=True)
    # joblib.dump(best_model, model_output_path)

    # Train best model on all data

    # Performance evaluation
    # pipeline = make_pipeline(params, algorithm)


if __name__ == "__main__":
    cli()

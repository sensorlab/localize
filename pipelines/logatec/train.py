from pathlib import Path

import click
import joblib
import xgboost
from sklearn import ensemble, linear_model, model_selection, multioutput, neighbors
import time

# try:
#     from sklearnex import patch_sklearn
#     patch_sklearn()
# except ImportError:
#     pass

from src import load_data, load_params, save_params, PredefinedSplit, safe_indexing


def tune_model_hyperparameters(model, hparams: dict, data, indice):
    pass


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
    type=click.Choice(
        ["LinearRegression", "RandomForestRegressor", "XGBRegressor", "XGBRFRegressor", "KNeighborsRegressor"],
        case_sensitive=False,
    ),
    required=True,
    # show_default=True,
    # help="Method to harmonize values",
)
@click.option(
    "--out-model",
    "--output-model",
    "--output",
    "--model-output",
    "model_output_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    # required=True,
    # help="Input data, parquet",
)
@click.option(
    # "--out-best-params",
    "--output-best-params",
    # "--best-params-output",
    "best_params_output_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    # required=True,
    # help="Input data, parquet",
)
# @click.option(
#     "--params",
#     "--model-params",
#     "params_path",
#     type=click.Path(exists=True, dir_okay=False, path_type=Path),
#     required=True,
# )
@click.option(
    "--output-report",
    "output_report_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def cli(
    input_dataset_path: Path,
    split_indices_path: Path,
    model_output_path: Path,
    best_params_output_path: Path,
    output_report_path: Path,
    # params_path: Path,
    algorithm: str,
):
    params = load_params("./params.yaml")
    random_state = params["seed"]
    mparams = params["models"]["regression"]

    # Load data
    features, targets = load_data(input_dataset_path)
    split_data = load_data(split_indices_path)
    cv_indices = split_data["indices"]
    cv = PredefinedSplit(cv_indices)

    # Number of available CPU cores
    n_jobs = -1  # joblib.cpu_count(only_physical_cores=True)

    # Stratified KFold for evaluation
    match algorithm:
        case "LinearRegression":
            estimator = linear_model.LinearRegression()
        case "RandomForestRegressor":
            estimator = ensemble.RandomForestRegressor(random_state=random_state)
        case "XGBRegressor":
            estimator = xgboost.XGBRegressor(random_state=random_state)
        case "XGBRFRegressor":
            estimator = xgboost.XGBRFRegressor(random_state=random_state)
        case "KNeighborsRegressor":
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

    model = gridsearch.best_estimator_

    reports = {
        "predictions": {
            "y_true": [],
            "y_pred": [],
        },
        "model_metadata": {
            "algorithm": algorithm,
            "hyperparameters": gridsearch.best_params_,
            "train_time": [],
            "predict_time": [],
        },
        "split_metadata": {
            "split_type": split_data["metadata"]["split_type"],
        },
    }

    for train_indices, test_indices in cv.split(features, targets):
        X_train = safe_indexing(features, train_indices)
        y_train = safe_indexing(targets, train_indices)

        X_test = safe_indexing(features, test_indices)
        y_test = safe_indexing(targets, test_indices)

        time_start = time.perf_counter()
        with joblib.parallel_backend("loky", n_jobs=n_jobs):
            model.fit(X_train, y_train)
        time_end = time.perf_counter()

        reports["model_metadata"]["train_time"].append(time_end - time_start)

        time_start = time.perf_counter()
        with joblib.parallel_backend("loky", n_jobs=n_jobs):
            y_pred = model.predict(X_test)
        time_end = time.perf_counter()

        reports["model_metadata"]["predict_time"].append(time_end - time_start)

        reports["predictions"]["y_true"].append(y_test)
        reports["predictions"]["y_pred"].append(y_pred)

    # reports = []

    # for (train_indices, test_indices), split_metadata in zip(cv.split(features, targets), split_data["metadata"]):
    #     X_train = safe_indexing(features, train_indices)
    #     y_train = safe_indexing(targets, train_indices)

    #     X_test = safe_indexing(features, test_indices)
    #     y_test = safe_indexing(targets, test_indices)

    #     time_start: float = time.perf_counter()
    #     with joblib.parallel_backend("threading", n_jobs=n_jobs):
    #         model.fit(X_train, y_train)
    #     time_end: float = time.perf_counter()

    #     y_pred = model.predict(X_test)

    #     report = {
    #         "predictions": {
    #             "y_true": y_test,
    #             "y_pred": y_pred,
    #         },
    #         "model_metadata": {
    #             "algorithm": algorithm,
    #             "hyperparameters": gridsearch.best_params_,
    #             "training_time": time_end - time_start
    #         },
    #         "split_metadata": split_metadata,
    #         # "train_metrics": {},
    #         # "additional_info": {},
    #     }

    #     reports.append(report)

    if output_report_path:
        joblib.dump(reports, output_report_path)

    # Save best model
    if model_output_path:
        joblib.dump(gridsearch.best_estimator_, model_output_path)

    if best_params_output_path:
        save_params(gridsearch.best_params_, best_params_output_path)


if __name__ == "__main__":
    cli()

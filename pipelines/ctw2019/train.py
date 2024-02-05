from pathlib import Path

import click
import joblib
import xgboost
from sklearn import (
    ensemble,
    linear_model,
    model_selection,
    multioutput,
    neighbors,
    pipeline,
    preprocessing,
    decomposition,
)
import time

import numpy as np


from src import load_data, load_params, save_params, PredefinedSplit, safe_indexing


def arnold2019localization():
    """The model is based on:
    Arnold et. al: On Deep Learning-based Massive MIMO Indoor User Localization
    """
    pass


def arnold2019sounding():
    """The model is based on:
    Arnold et. al: Novel Massive MIMO Channel Sounding Data applied to Deep Learning-based Indoor Positioning
    """
    pass


def cerar2021paper():
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
        ["LinearRegression", "RandomForestRegressor", "XGBRegressor", "KNeighborsRegressor"],
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
    algorithm: str,
):
    params = load_params("./params.yaml")
    random_state = params["seed"]
    mparams = params["models"].get("regression", {})

    # Load data
    features, targets = load_data(input_dataset_path)

    # Modifications to dataset for machine learning
    h, snr = features["h"], features["snr"]
    h_flat = h.reshape(h.shape[0], -1)
    features = np.concatenate((h_flat, snr), axis=1)
    assert features.shape == (h.shape[0], 16 * 924 * 2 + 16)
    ######

    split_data = load_data(split_indices_path)
    cv_indices = split_data["indices"]
    cv = PredefinedSplit(cv_indices)

    # runner parameters
    n_jobs = joblib.cpu_count(only_physical_cores=False)
    backend = "threading"

    # Pick the algorithm to evaluate
    match algorithm:
        case "LinearRegression":
            estimator = linear_model.LinearRegression(n_jobs=-1)

        case "RandomForestRegressor":
            estimator = ensemble.RandomForestRegressor(random_state=random_state, n_jobs=-1)

        case "XGBRegressor":
            estimator = xgboost.XGBRegressor(random_state=random_state, n_jobs=-1)

        case "KNeighborsRegressor":
            estimator = neighbors.KNeighborsRegressor(n_jobs=-1)

        case "arnold2019localization":
            pass

        case _:
            raise NotImplementedError(f"Algorithm '{algorithm}' not implemented.")

    estimator = pipeline.Pipeline(
        [
            ("scale", preprocessing.StandardScaler()),
            ("pca", decomposition.PCA(n_components=0.7)),
            ("regressor", multioutput.MultiOutputRegressor(estimator, n_jobs=-1)),
        ]
    )

    # Prepare hyper-parameters:
    hparams = mparams.get(algorithm, {})

    # (Optional) hyper-parameter tuning
    if hparams:
        # Estimator expects `estimator__` prefix
        hparams = {f"regressor__estimator__{k}": v for k, v in hparams.items()}

        gridsearch = model_selection.GridSearchCV(
            estimator=estimator,
            param_grid=hparams,
            refit="mse",
            cv=cv,
            n_jobs=-1,
        )

        with joblib.parallel_backend(backend, n_jobs=n_jobs):
            gridsearch.fit(features, targets)

        best_model = gridsearch.best_estimator_
        best_params = gridsearch.best_params_
    else:
        best_model = estimator
        best_params = hparams

    # Prepare empty report
    reports = {
        "predictions": {
            "y_true": [],
            "y_pred": [],
        },
        "model_metadata": {
            "algorithm": algorithm,
            "hyperparameters": best_params,
            "train_time": [],
            "predict_time": [],
        },
        "split_metadata": split_data["metadata"],
        # "split_metadata": {
        #    "split_type": split_data["metadata"]["split_type"],
        # },
    }

    for train_indices, test_indices in cv.split(features, targets):
        X_train = safe_indexing(features, train_indices)
        y_train = safe_indexing(targets, train_indices)

        X_test = safe_indexing(features, test_indices)
        y_test = safe_indexing(targets, test_indices)

        time_start = time.perf_counter()
        with joblib.parallel_backend(backend, n_jobs=n_jobs):
            best_model.fit(X_train, y_train)
        time_end = time.perf_counter()

        reports["model_metadata"]["train_time"].append(time_end - time_start)

        time_start = time.perf_counter()
        with joblib.parallel_backend(backend, n_jobs=n_jobs):
            y_pred = best_model.predict(X_test)
        time_end = time.perf_counter()

        reports["model_metadata"]["predict_time"].append(time_end - time_start)

        reports["predictions"]["y_true"].append(y_test)
        reports["predictions"]["y_pred"].append(y_pred)

    if output_report_path:
        joblib.dump(reports, output_report_path)

    if best_params_output_path:
        save_params(best_params, best_params_output_path)


if __name__ == "__main__":
    cli()

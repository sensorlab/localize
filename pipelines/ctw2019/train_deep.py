from pathlib import Path

import click
import joblib
import time

import lightning as L
import numpy as np


from src import load_params, PredefinedSplit, safe_indexing


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
    "architecture",
    type=click.Choice(
        ["arnold2019sounding", "arnold2018deep"],
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
    architecture: str,
):
    params = load_params("./params.yaml")
    random_state = params["seed"]
    # mparams = params["models"].get("regression", {})

    # Load data
    features, targets = joblib.load(input_dataset_path)
    # features = features["h"]
    # features = np.transpose(features, (0, 3, 1, 2))  # BHWC --> BCHW

    split_data = joblib.load(split_indices_path)
    cv_indices = split_data["indices"]
    cv = PredefinedSplit(cv_indices)

    # runner parameters
    n_jobs = joblib.cpu_count(only_physical_cores=False)
    backend = "threading"

    match architecture:
        case "arnold2019sounding":
            from src.models.arnold2019sounding import Arnold2019SoundingModel
            from src.wrappers import LightningRegressorWrapper

            net = Arnold2019SoundingModel(random_state=random_state)
            L.seed_everything(random_state)
            estimator = LightningRegressorWrapper(net, batch_size=64, max_epochs=100, random_state=random_state)

            # BHWC --> BCHW
            features = np.transpose(features["h"], (0, 3, 1, 2))
            assert features.shape == (features.shape[0], 2, 16, 924)

        case "arnold2018deep":
            from src.models.arnold2018deep import Arnold2018DeepModel
            from src.wrappers import LightningRegressorWrapper

            net = Arnold2018DeepModel(in_channels=16 * 924 * 2 + 16, random_state=random_state)
            L.seed_everything(random_state)
            estimator = LightningRegressorWrapper(net, batch_size=64, max_epochs=100, random_state=random_state)

            h, snr = features["h"], features["snr"]
            h_flat = h.reshape(h.shape[0], -1)
            features = np.concatenate((h_flat, snr), axis=1)
            assert features.shape == (h.shape[0], 16 * 924 * 2 + 16)

        case _:
            raise NotImplementedError()

    best_model = estimator

    # Prepare empty report
    reports = {
        "predictions": {
            "y_true": [],
            "y_pred": [],
        },
        "model_metadata": {
            "algorithm": architecture,
            "hyperparameters": {
                "batch_size": 64,
                "max_epochs": 100,
            },
            "train_time": [],
            "predict_time": [],
        },
        "split_metadata": split_data["metadata"],
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

    # if best_params_output_path:
    #    save_params(best_params, best_params_output_path)


if __name__ == "__main__":
    cli()

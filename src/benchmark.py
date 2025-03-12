#!/usr/bin/env python3

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:
    pass

# check if running on MacOS
import platform

import click
import joblib
import numpy as np
import torch


if platform.system() == "Darwin":
    # check if M1/M2
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


import os
from pathlib import Path

import keras
import tensorflow as tf
import yaml
from skorch.helper import SliceDict

from src import PredefinedSplit
from src.automl.automl_manager import AutoMLManager
from src.gridsearch.gridsearch_manager import GridSearchManager
from src.metrics import MetricsHandler


torch.set_float32_matmul_precision("high")
# Just to make sure that everything stays deterministic
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"

tf.config.experimental.enable_op_determinism()
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

keras.utils.set_random_seed(42)

tmp_dir_path = "../../tmp/"


def load_yaml_config(path: str | Path) -> dict:
    with open(path, mode="r") as file:
        config = yaml.safe_load(file)
    return config


def get_hyperparameters(model_name, models_config):
    return models_config[model_name].get("hyperparameters", {})


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
    "--indices",
    "split_indices_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
)
@click.option(
    "--algorithm",
    "--use",
    "--model",
    "model_name",
    type=str,
    required=True,
    # show_default=True,
    # help="Select model",
)
@click.option("--optimizer", "optimizer_name", type=str, required=True)
# @click.option(
#     "--out-model",
#     "--output-model",
#     "--output",
#     "--model-output",
#     "model_output_path",
#     type=click.Path(dir_okay=False, writable=True, path_type=Path),
#     # required=True,
#     # help="Input data, parquet",
# )
# @click.option(
#    # "--out-best-params",
#    "--output-best-params",
#    # "--best-params-output",
#    "best_params_output_path",
#    type=click.Path(dir_okay=False, writable=True, path_type=Path),
#    # required=True,
#    # help="Input data, parquet",
# )
# @click.option(
#     "--params",
#     "--model-params",
#     "params_path",
#     type=click.Path(exists=True, dir_okay=False, path_type=Path),
#     required=True,
# )
@click.option(
    # "--output-report",
    "--output-results",
    "results_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def cli(
    input_dataset_path: Path,
    split_indices_path: Path,
    # model_output_path: Path,
    # best_params_output_path: Path,
    results_path: Path,
    model_name: str,
    optimizer_name: str,
):
    # load dataset
    features, targets = joblib.load(input_dataset_path, mmap_mode=None)

    # https://github.com/skorch-dev/skorch/blob/master/docs/user/FAQ.rst#how-do-i-use-sklearn-gridseachcv-when-my-data-is-in-a-dataset
    if isinstance(features, dict):
        features = SliceDict(**features)

    # load split indices
    splits = joblib.load(split_indices_path, mmap_mode=None)

    # load indices into custom CrossValidator
    cv_indices: list[tuple[np.ndarray, np.ndarray]] = splits["indices"]
    cv = PredefinedSplit(cv_indices)

    # Path to your YAML configuration file
    config_path = "./params.yaml"
    # Load the configuration
    config = load_yaml_config(config_path)

    metrics_handler = MetricsHandler(config["evaluation"]["metrics"])
    store_top_num_models = config["evaluation"].get("save_top_models", 0.1)

    if optimizer_name == "gridsearch":
        gridsearch_config = config["gridsearch"][model_name]
        hyperparameters = get_hyperparameters(model_name, config["gridsearch"])

        estimator, is_skorch_module = GridSearchManager.construct_model(gridsearch_config)

        grid_search_parameters = {
            "estimator": estimator,
            "param_grid": hyperparameters,
            "cv": cv,
        }

        # Fix: Skorch and GridSearch don't play well with parallel CVs
        grid_search = GridSearchManager(
            tmp_dir_path=tmp_dir_path,
            model_save_dir_path=Path(str(results_path).replace(".pkl", "")),
            single_thread=is_skorch_module,
            scorers=metrics_handler,
            save_models=True,
        )

        grid_search.search(features, targets, search_parameters=grid_search_parameters)

        reports = grid_search.generate_report(store_top_num_models=store_top_num_models)

        # grid_search.cleanup_tmp()

    elif optimizer_name == "automl":
        auto_ml = AutoMLManager(
            config=config["automl"][model_name],
            tmp_dir_path=tmp_dir_path,
            model_save_dir_path=Path(str(results_path).replace(".pkl", "")),
            project_name=model_name,
        )

        auto_ml.search(features, targets, test_size=config["split"].get("test_size", 0.2))
        reports = auto_ml.generate_report(store_top_num_models=store_top_num_models, cv=cv, metrics=metrics_handler)

        # auto_ml.cleanup_tmp()

    reports["model_data"]["metadata"] = {"algorithm": model_name}
    reports["split_data"]["metadata"] = splits["metadata"]

    if results_path:
        joblib.dump(reports, results_path)

    # if best_params_output_path:
    #    save_params(best_hparams, best_params_output_path)


if __name__ == "__main__":
    cli()

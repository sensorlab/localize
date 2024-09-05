#!/usr/bin/env python3

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:
    pass

import importlib
import inspect
import time
from pathlib import Path

import click
import joblib
import numpy as np


import torch

# check if running on MacOS
import platform

if platform.system() == 'Darwin':
    #check if M1/M2
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


import yaml
from sklearn.base import BaseEstimator
from src.gridsearch.gridsearch_manager import GridSearchManager
from sklearn.pipeline import Pipeline
from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.helper import SliceDict

from src import PredefinedSplit, safe_indexing, save_params

from data import root_mean_squared_error, euclidean_distance, median_euclidean_distance, mean_percentage_error
from sklearn import metrics

torch.set_float32_matmul_precision("high")


# List of workarounds
quirks = {
    # Fix #1: Skorch and GridSearch don't play well with parallel CVs
    "is_skorch_module": False,
    # Fix #2: GridSearchCV doesn't save trained models
    "store_models": False # CONSTANT
}


def load_yaml_config(path: str | Path) -> dict:
    with open(path, mode="r") as file:
        config = yaml.safe_load(file)
    return config


def construct_model_or_pipeline(model_config: dict) -> BaseEstimator:
    if "pipeline" in model_config:
        steps = []
        for step in model_config["pipeline"]:
            step_estimator = construct_model_or_pipeline(step)
            steps.append((step["step"], step_estimator))
        return Pipeline(steps)
    else:
        module = importlib.import_module(model_config["module"])
        ModelClass = getattr(module, model_config["class"])
        parameters = model_config.get("parameters", {})

        # Check if ModelClass has n_jobs parameter. If so, set it to number of cores (not threads)
        if "n_jobs" in inspect.signature(ModelClass).parameters:
            parameters["n_jobs"] = joblib.cpu_count(only_physical_cores=True)

        # TODO: should `optimizer` be part of parameter or on same model as declaration
        if "optimizer" in parameters:
            optimizer_module = importlib.import_module(parameters["optimizer"]["module"])
            OptimizerClass = getattr(optimizer_module, parameters["optimizer"]["class"])
            parameters["optimizer"] = OptimizerClass

        # TODO: should `optimizer` be part of parameter or on same model as declaration
        if "callbacks" in parameters:
            callbacks = [construct_model_or_pipeline(callback) for callback in parameters["callbacks"]]
            parameters["callbacks"] = callbacks

        if "estimator" in model_config:  # Handling nested models
            # Check if the current model is a Skorch model and requires a PyTorch model class instead of instance
            if ModelClass in [NeuralNetRegressor, NeuralNetClassifier]:
                quirks["is_skorch_module"] = True
                # Import the PyTorch model class without instantiating it
                inner_module = importlib.import_module(model_config["estimator"]["module"])
                InnerModelClass = getattr(inner_module, model_config["estimator"]["class"])
                # Pass the PyTorch model class directly, without instantiation
                return ModelClass(InnerModelClass, **parameters)

            inner_model = construct_model_or_pipeline(model_config["estimator"])
            return ModelClass(inner_model, **parameters)

        return ModelClass(**parameters)


def get_hyperparameters(model_name, models_config):
    return models_config[model_name].get("hyperparameters", {'n_jobs': [4]})


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
    # "--output-report",
    "--output-results",
    "results_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def cli(
    input_dataset_path: Path,
    split_indices_path: Path,
    # model_output_path: Path,
    best_params_output_path: Path,
    results_path: Path,
    model_name: str,
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
    model_config = config["models"][model_name]

    # construct pipeline from YAML file and obtain hyperparameters
    model_or_pipeline = construct_model_or_pipeline(model_config)
    hyperparameters = get_hyperparameters(model_name, config["models"])
    
    grid_search_parameters = {"estimator": model_or_pipeline,
                              "param_grid":hyperparameters,
                              "cv":cv}

    scorers= {'rmse': {'scorer': root_mean_squared_error, 'greater_is_better': False},
              'euclidean': {'scorer': euclidean_distance, 'greater_is_better': False},
              'r_squared': {'scorer': metrics.r2_score, 'greater_is_better': True},
              'mae': {'scorer': metrics.mean_absolute_error, 'greater_is_better': False},
              'mede': {'scorer': median_euclidean_distance, 'greater_is_better': False},
              'mpe': {'scorer': mean_percentage_error, 'greater_is_better': False},
             }
    if True:
        grid_search = GridSearchManager(tmp_dir_path = "../../tmp/",
                                        model_save_dir_path = Path(str(results_path).replace(".pkl", "")),
                                        single_thread=quirks["is_skorch_module"], 
                                        scorers = scorers)
        grid_search.search(features, targets, search_parameters = grid_search_parameters)

        reports = grid_search.generate_report(store_top_num_models = 0.1)

        reports["model_data"]["model_metadata"] = {"algorithm": model_name}
        reports["split_data"]["split_metadata"] = splits["metadata"]
        
        grid_search.cleanup_tmp()
        
    else:
        best_hparams = hyperparameters
        results = {}
        for key in scorers:
            results[key] = []
        
        for train_indices, test_indices in cv.split(features, targets):
            X_train = safe_indexing(features, train_indices)
            y_train = safe_indexing(targets, train_indices)

            X_test = safe_indexing(features, test_indices)
            y_test = safe_indexing(targets, test_indices)

            time_start = time.perf_counter()
            # with joblib.parallel_config(backend=backend, n_jobs=n_jobs, verbose=11):
            model_or_pipeline.fit(X_train, y_train)
            time_end = time.perf_counter()

            time_start = time.perf_counter()
            # with joblib.parallel_config(backend=backend, n_jobs=n_jobs, verbose=11):
            y_pred = model_or_pipeline.predict(X_test)
            time_end = time.perf_counter()
            
            for key, scorer in scorers.items():
                results[key].append(scorer["scorer"](y_test, y_pred))
            
            y_pred = y_pred.astype(np.float32)
        for key in scorers:
            results[f"mean_test_{key}"] = np.mean(results[key])
            results[f"std_test_{key}"] = np.std(results[key])
        
#=======================================
#     # general train parameters
#     # n_jobs = joblib.cpu_count()
#     gridsearch_jobs = 4
#     # backend = "threading"

#     if quirks["is_skorch_module"]:
#         gridsearch_jobs = 1

#     print(model_or_pipeline)
#     print(f"{hyperparameters=}")

#     if hyperparameters:
#         grid_search = GridSearchCV(
#             estimator=model_or_pipeline,
#             param_grid=hyperparameters,
#             scoring="neg_mean_squared_error",
#             n_jobs=gridsearch_jobs,
#             cv=cv,
#             verbose=2,
#             error_score="raise",
#         )

#         # with joblib.parallel_config(backend=backend, n_jobs=n_jobs, verbose=11):
#         grid_search.fit(features, targets)

#         model_or_pipeline = grid_search.best_estimator_
#         best_hparams = grid_search.best_params_

#     else:
#         best_hparams = hyperparameters

#     print(f"{best_hparams=}")

    # Prepare template for the final report
    
#     for train_indices, test_indices in cv.split(features, targets):
#         X_train = safe_indexing(features, train_indices)
#         y_train = safe_indexing(targets, train_indices)

#         X_test = safe_indexing(features, test_indices)
#         y_test = safe_indexing(targets, test_indices)

#         time_start = time.perf_counter()
#         # with joblib.parallel_config(backend=backend, n_jobs=n_jobs, verbose=11):
#         model_or_pipeline.fit(X_train, y_train)
#         time_end = time.perf_counter()

#         reports["model_metadata"]["train_time"].append(time_end - time_start)

#         time_start = time.perf_counter()
#         # with joblib.parallel_config(backend=backend, n_jobs=n_jobs, verbose=11):
#         y_pred = model_or_pipeline.predict(X_test)
#         time_end = time.perf_counter()

#         reports["model_metadata"]["predict_time"].append(time_end - time_start)

#         reports["predictions"]["y_true"].append(y_test)

#         y_pred = y_pred.astype(np.float32)
#         reports["predictions"]["y_pred"].append(y_pred)

    

    if results_path:
        joblib.dump(reports, results_path)

    if best_params_output_path:
        save_params(best_hparams, best_params_output_path)


if __name__ == "__main__":
    cli()

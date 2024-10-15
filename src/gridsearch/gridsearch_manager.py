from sklearn.model_selection import GridSearchCV
from .modified_gridsearch import GridSearchCVWithStoredModels
from sklearn import metrics
import numpy as np
import pandas as pd
import os
from pathlib import Path
import joblib
from tabulate import tabulate
import humanize
import shutil
import glob
from tqdm import tqdm


from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from skorch import NeuralNetClassifier, NeuralNetRegressor

import importlib
import inspect

from typing import Any, Union, Dict, Callable

from src import empty_directory
from src.metrics import MetricsHandler

class GridSearchManager:
    def __init__(self, scorers: MetricsHandler, tmp_dir_path: Path, model_save_dir_path : Path, save_models: bool = True, show_progress_bar : bool = True, single_thread: bool = False, verbose: int = 2):

        # Default grid search settings
        self._search_default_params = {
            "verbose" : 1,
            "n_jobs" : 4,
            "error_score" : "raise",
            "refit": False
        }

        self._search_forced_params = {
            "scoring" : scorers.to_GridsearchCV(),
        }
        self._scorers =  scorers.metrics_names

        # Forces single thread
        if single_thread:
            self._search_forced_params["n_jobs"] = 1

        # Set verbosity level for grid search
        self.verbose = verbose
        if self.verbose < 2:
            self._search_default_params["verbose"] = 0

        self._store_models = save_models
        self.tmp_dir_path = tmp_dir_path
        self.model_save_dir_path = model_save_dir_path
        self._show_progress_bar = show_progress_bar


    def log(self, text: str, level: int = 1) -> None:
        """
        Logs messages based on the verbosity level.

        Args:
        - text (str): Text to print if the verbosity level meets the required level.
        - level (int): Required level to print the text.

        """
        # Only here so that it's easier to change in the future
        if self.verbose >= level:
            print(text)

    @classmethod
    def construct_model(cls, model_config) -> BaseEstimator:
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

        # Check if ModelClass has n_jobs parameter. If so, set it to number of cores (not threads)
        if "n_jobs" in inspect.signature(ModelClass).parameters:
            parameters["n_jobs"] = joblib.cpu_count(only_physical_cores=True)

        return ModelClass(**parameters)

    def _order_by_mean_rank(self, results_df: pd.DataFrame, scoring: Union[dict, None] = None) -> pd.DataFrame:
        """
        Orders the candidates in the results DataFrame by their mean rank, and then by individual scoring columns.

        Args:
        - results_df (pd.DataFrame): DataFrame containing the results of GridSearchCV.
        - scoring (dict | None, optional): Dictionary of scoring metrics used in GridSearchCV.
            If not given uses self._scorers. Defaults to None.

        Returns:
        - pd.DataFrame: DataFrame sorted by mean rank and individual rank columns.

        """
        if scoring is None:
            scoring = self._scorers

        rank_columns = [f'rank_test_{key}' for key in scoring]

        # Calculate the mean rank across all scoring metrics
        results_df['mean_rank'] = results_df[rank_columns].mean(axis=1)

        # Sort by mean rank, and then by individual rank columns
        sort_columns = ['mean_rank'] + rank_columns
        sorted_df = results_df.sort_values(by=sort_columns)

        # Save original index and then reset it
        sorted_df['candidate_idx'] = sorted_df.index
        sorted_df.reset_index(drop=True, inplace=True)

        return sorted_df


    def _correct_scores(self, df:pd.DataFrame, scores: dict) -> pd.DataFrame:
        """
        Scores returned by GridSearchCV are negated if a lower score is better; this function
        corrects that by checking if greater is better for each scoring metric and applying corrections.

        Args:
        - df (pd.DataFrame): DataFrame containing the results of GridSearchCV.
        - scores (dict) dict of scorer_name: greater_is_better

        Returns:
        - pd.DataFrame: DataFrame with corrected scores. If a higher score is better, the score is
            returned as is; otherwise, it is negated.
        """
        for scorer_name, greater_is_better in scores.items():
            # Identify all columns that contain the scorer name (e.g., split0_test_rmse, mean_test_rmse, etc.)
            relevant_columns = [col for col in df.columns if (scorer_name in col and 'std_' not in col)]

            for col in relevant_columns:
                # If a higher score is better, no need to change anything; otherwise, negate the values
                if not greater_is_better:
                    df[col] = -df[col]

        return df


    def search(self, features: Union[pd.DataFrame, np.ndarray],
               targets: Union[pd.DataFrame, np.ndarray],
               search_parameters: dict[str, Any]) -> None:
        """
        Performs the grid search and saves the relevant data.

        Args:
        - search_parameters (dict): The parameters that are passed to GridSearchCV.
        - feature (pd.DataFrame | np.ndarray): Input data for training
        - targets (pd.DataFame | np.ndarray): Output data for validation
        """


        # Merge provided parameters with default and forced settings
        parameters = {**self._search_default_params, **search_parameters, **self._search_forced_params}

        assert "param_grid" in parameters and parameters["param_grid"], "'param_grid' must exist and not be empty"

        self.log("\n")
        self.log(f"Estimator: {parameters['estimator']}")
        self.log(f"Hyperparameters: {parameters['param_grid']}\n")

        # # Choose the appropriate GridSearchCV class based on whether models should be stored
        if self._store_models:
            self.grid_search = GridSearchCVWithStoredModels(**parameters)
            custom_params = {
                "tmp_dir_path": self.tmp_dir_path,
                "show_progress_bar": self._show_progress_bar
            }
        else :
            self.grid_search = GridSearchCV(**parameters)
            custom_params = {}

        # Perform grid search
        self.grid_search.fit(features, targets, **custom_params)

        # Store the results and sort them by mean rank
        self.results = pd.DataFrame(self.grid_search.cv_results_)
        self.results = self._order_by_mean_rank(self.results)
        self.results = self._correct_scores(self.results, self._scorers)


    def predict(self, X: Union[np.ndarray, pd.Series], index: Union[None, int] = None) -> np.ndarray: #DEPRECATED
        """
        Predicts the output using the best model by default, or a specified model if `index` is provided.

        Args:
        - X (np.ndarray | pd.Series): Input data for prediction.
        - index (None | int, optional): Index of the model to use for prediction.
            If None, uses the best model. Default to None.

        Returns:
        - np.ndarray: Array of predicted values based on the selected model.
        """

        if index is not None and self._store_models:
            # Validate the index
            if not (0 <= index < len(self.models)):
                raise ValueError(f"Invalid index {index}. It should be between 0 and {len(self.models) - 1}.")

            model = self.models[index]
            self.log(f"Using model at index {index} for prediction.", level=3)
        else:
            # Default to using the best model
            model = self.best_estimator_
            self.log("Using the best model for prediction.", level=3)

        y_pred = model.predict(X)

        return y_pred


    def generate_report(self, store_top_num_models: Union[int, float] = 0.1) -> dict:
        """
        Generates a report of the top models based on the evaluation results.
        The function loads model data, moves the selected top models to a specified directory,
        and generates a report table. The table includes ranking, scores, fit time, score time,
        model size, and model parameters.

        Args:
        - store_top_num_models (int | float): Specifies the number or fraction of top models to store.
            Models are stored to model_save_dir_path. Defaults to 0.1.
          - If an integer is provided, it represents the exact number of top models.
          - If a float between 0 and 1 is provided, it represents the fraction of the total candidates.

        Returns:
        - dict: A dictionary with all the report data
        """

        # Defining report template
        self.report = {
            "model_data": {
                "reports": [],
                "metadata": None
            },
            "split_data" : {
                "splits": [],
                "metadata": None
            },
            "optimizer_data": {
                "metadata": {
                    "algorithm": "gridsearch"
                },
                "additional_data":  {
                    "results": self.results
                }
            }
        }

        # Load general information
        general_info = joblib.load(os.path.join(self.tmp_dir_path, f"general.pkl"))
        n_splits = general_info["n_splits"]
        n_candidates = general_info["n_candidates"]

        # Determine the number of top models to save
        if isinstance(store_top_num_models, int) or (isinstance(store_top_num_models, float) and store_top_num_models.is_integer()):
            save_top_n = min(store_top_num_models, n_candidates)
        elif isinstance(store_top_num_models, float) and 0 < store_top_num_models < 1:
            save_top_n = int(n_candidates * store_top_num_models)
        else:
            save_top_n = 1

        save_top_n = max(1, min(save_top_n, n_candidates))

        # Get the top models from the results
        top_n_df = self.results.head(save_top_n).copy()

        # Prepare to log the report
        self.log("\n")
        table_data = []

        self.model_save_dir_path.mkdir(parents=True, exist_ok=True)
        empty_directory(self.model_save_dir_path)

        # Iterate through the top models and prepare the report data
        for idx, row in top_n_df.iterrows():
            candidate_idx = row['candidate_idx']

            # Load and move top models for each split
            model_data_per_split = []
            for split_idx in range(n_splits):
                model_file = os.path.join(self.tmp_dir_path, f"est-{candidate_idx}-{split_idx}.pkl")

                if os.path.exists(model_file):
                    model_size = os.path.getsize(model_file)

                    # Move the model file to the save directory
                    destination_file = os.path.join(self.model_save_dir_path, f"est-{candidate_idx}-{split_idx}.pkl")
                    shutil.move(model_file, destination_file)

                    predictions_file = os.path.join(self.tmp_dir_path, f"pred-{candidate_idx}-{split_idx}.pkl")
                    if os.path.exists(predictions_file):
                        predictions_data = joblib.load(predictions_file)
                        y_true = predictions_data["y_true"]
                        y_pred = predictions_data["y_pred"]
                    else:
                        y_true = None
                        y_pred = None

                        self.log(f"Warning: File '{predictions_file}' does not exist.", level = 0)

                    model_data_per_split.append({"model_path": destination_file, "y_true": y_true, "y_pred": y_pred, "model_size":model_size})
                else:
                    self.log(f"Warning: File '{model_file}' does not exist.", level = 0)
                    models.append(None)  # Handle missing files

            # Prepare the model report
            model_report = {
                "model_data_per_split": model_data_per_split,
                "scores": {
                    key: {
                        "mean": row[f'mean_test_{key}'],
                        "std": row[f'std_test_{key}']
                    }
                    for key in self._scorers
                },
                "params": row['params'],
                "fit_time": {"mean": row['mean_fit_time'], "std": row['std_fit_time']},
                "score_time": {"mean": row['mean_score_time'], "std": row['std_score_time']}
            }
            self.report["model_data"]["reports"].append(model_report)


            total_model_size = sum([dat["model_size"] for dat in model_report["model_data_per_split"]])
            # Prepare the row for the logged report table
            table_row = [
                idx + 1,
                *[f"{model_report['scores'][key]['mean']:.4f} ± {model_report['scores'][key]['std']:.2f}" for key in self._scorers],
                f"{model_report['fit_time']['mean']:.1f} ± {model_report['fit_time']['std']:.0f} sec",
                f"{model_report['score_time']['mean']:.1f} ± {model_report['score_time']['std']:.0f} sec",
                f"{humanize.naturalsize(total_model_size, binary=True)} ({n_splits}x{humanize.naturalsize(model_size, binary=True)})",
                f"{model_report['params']}"
            ]
            table_data.append(table_row)

        # Define the headers for the report table
        headers = [
            "Rank",
            *[key for key in self._scorers],
            "Fit Time",
            "Score Time",
            "Model Size",
            "Params"
        ]

        self.log(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Load and append split data to the report
        for split_idx in range(n_splits):
            split_file = f"{self.tmp_dir_path}split-{split_idx}"
            if os.path.exists(split_file):
                split = joblib.load(split_file)
                self.report["split_data"]["splits"].append(split)
            else:
                self.log(f"Warning: File '{split_file}' does not exist.", level = 0)
                self.report["split_data"]["splits"].append(None)
        return self.report

    def cleanup_tmp(self):
        """
        Cleans up the temporary model storage directory by deleting all files
        except for the '.gitignore' file. Logs the number of files erased.
        """

        # Use glob to list all files in the temporary directory
        files = glob.glob(os.path.join(self.tmp_dir_path, '*'))

        # Log the start of the cleanup process
        self.log("\nCleaning up temporary model storage directory")

        num_erased = 0
        is_gitignore = False  # Flag to track if .gitignore is present

        # Initialize the progress bar with total iterations
        progress_bar = tqdm(total=len(files), desc="Deleting tmp files", unit="files", dynamic_ncols=True, smoothing=0.1)

        # Iterate over all files in the directory
        for file in files:
            # Check if the file is not .gitignore
            if os.path.basename(file) != '.gitignore':
                try:
                    # Attempt to remove the file
                    os.remove(file)
                    num_erased+=1

                except Exception as e:
                    print(f"Could not delete {file}: {e}")
            else:
                is_gitignore = True # Mark that .gitignore is present
            progress_bar.update(1)

        progress_bar.close()

        # Log the number of files erased, excluding .gitignore if present
        self.log (f"Erased {num_erased} out of {len(files) - int(is_gitignore)} files") #don't count git ignore







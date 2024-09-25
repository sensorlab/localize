import os
from pathlib import Path
import gc
import time
import shutil
from typing import Union, Generator

import numpy as np
import pandas as pd
import keras
from keras_tuner import Tuner, HyperParameters
from keras.models import Model
from keras.optimizers import Optimizer
from sklearn.model_selection import train_test_split


from .automl_configparser import AutoKerasConfigParser
from .utils import utils
from src import PredefinedSplit, safe_indexing, empty_directory
from src.metrics import MetricsHandler

##
import matplotlib.pyplot as plt
##

class AutoMLManager:
    """
    A class to manage all interactions with autokeras. Responsible for configuration,
    training, evaluation, and generating the report dict.
    """
    def __init__(self, config: dict, tmp_dir_path: Path, model_save_dir_path : Path, project_name: str):
        """
        Initializes the AutoMLManager instance.

        Args:
        - config (dict): The configuration dictionary for AutoML.
        - tmp_dir_path (Path): The directory path for storing temporary files.
        - model_save_dir_path (Path): The directory path for saving the trained models.
        - project_name (str): The name of the project.
        """
        self.config = config.copy()
        self.model_save_dir_path = model_save_dir_path
        self.tmp_dir_path = Path(os.path.join(tmp_dir_path, project_name))
        self.setup_config(project_name)
        self.setup_auto_model()


    def setup_config(self, project_name: str):
        """
        Ensures all important settings are present and then constructs the inputs,
        outputs, and blocks using the config parser.

        Args:
        - project_name (str): name of the project, used by autokeras.
        """

        self.config["settings"]["directory"] = self.tmp_dir_path
        self.config["settings"].setdefault("project_name", project_name)
        self.config_parser = AutoKerasConfigParser(self.config)
        self.seed = self.config["settings"].get("seed", 42)

        # Fit config is used during fitting, but must still be preprocessed.
        self.fit_config = utils.parse_args(self.config.get("fit_settings", {}), self.config_parser.hp)
        self.history = keras.callbacks.History()
        self.fit_config.setdefault("callbacks", []).append(self.history)


    def setup_auto_model(self):
        """
        Sets up the automodel using, and retrieves data needed later on.
        """
        # The configs that determine how data is given to each input.
        self.input_configs = self.config_parser.input_configs

        # Used to allow any number of outputs, names are needed to reference
        # each output when, e.g. passing metrics.
        self.num_outputs = len(self.config_parser.outputs)
        self.output_names = self.config_parser.output_names

        self.auto_model = self.config_parser.build_model()


    def prepare_data(self, features: Union[pd.DataFrame, dict], targets: pd.DataFrame, test_size: float):
        """
        Prepares the data to be inputed to the AutoModel.

        Args:
        - feature (pd.DataFrame | dict): Input data for training
        - targets (pd.DataFame): Output data for validation
        - test_size (float): size of the validation split
        """
        self.preped_features = self._prepare_features(features)

        # Splits the numpy array into a list of 1D arrays, one fore each input,
        # because that is the format expected by the AutoModel
        self.preped_targets = np.hsplit(utils.to_numpy(targets), self.num_outputs)

        self._split_data(test_size)


    def _prepare_features(self, features: Union[pd.DataFrame, dict]) -> list[np.ndarray]:
        """
        Prepares the input data for each model input based on the configs.

        Args:
        - features (pd.DataFrame | dict): input data to be processed.
        """
        if isinstance(features, dict):
            #If features is a dict, then 'key' must be provided for each input, features[key] are then processed as normal
            return [self._process_features(features[cnf['key']], cnf) for cnf in self.input_configs]
        # processes each features for each input.
        return [self._process_features(features, cnf) for cnf in self.input_configs]


    def _process_features(self, features: Union[np.ndarray, pd.DataFrame], config: Union[dict, list]):
        """
        Processes the input data, for a model input given it's config. Handles filtering of columns and
        converting to np.ndarray.

        args:
        - features (np.ndarray, pd.datafrane): input data to be filtered and converted.
        - config (dict | list): configuration on how to filter the data.
        """
        if config is None or isinstance(config, list):
            # When config is a list, return the columns with the given index/name.
            return utils.to_numpy(features[config] if config else features)
        if isinstance(features, pd.DataFrame):
            config.pop("key", None) # If it was a dict that has already been dealt with.
            return self._process_dataframe(features, config)
        return features


    def _process_dataframe(self, df: pd.DataFrame, config: dict):
        """
        Handles filtering of columns in a pd.DataFrame.

        Args:
        - df (pd.DataFrame): a pandas dataframe that will be filtered.
        - config (dict): a dictionary of function_name: substring/s
        """
        # TODO: rewrite this to just use regex for filtering

        selected_cols = df.columns

        # A list of supported functions for filtering.
        # Callable(name, search_str) -> bool, True to include the column
        funcs = {
            "startswith": lambda name, search_str: name.startswith(search_str),
            "endswith": lambda name, search_str: name.endswith(search_str),
            "contains": lambda name, search_str: search_str in name
        }

        """
        Multiple filters can be applied. Each filter can either be given a
        substring or a list of substrings. When multiple filters are present
        a column must match all of them (and). When multiple given a list of
        substrings a collumn must match any of them (or)
        """
        for key, value in config.items():
            if key not in funcs:
                raise KeyError(f"Invalid filter for column selection. Got {key} expected {', '.join(funcs.keys())}.")
            selected_cols = utils.select_columns(selected_cols, funcs[key], value)
        return df[selected_cols].to_numpy()


    def _split_data(self, test_size: float):
        """
        Creates a train test split for used when fitting automodel, as using
        corss-validation would mean rerunning the entire search multiple times
        (AutoKeras has no native support for cv)

        Args:
        - test_size (float): size of the validation split.
        """
        n_samples = self.preped_features[0].shape[0]

        val_split = self.fit_config.get("validation_split", None)

        final_test_size = val_split if val_split else test_size
        shuffle = self.fit_config.get("shuffle", True)

        # features and targets are list[np.ndarray], we make a temporary
        # array just to guarantee that everything is split correctly
        indices = np.arange(n_samples)
        train_indices, val_indices = train_test_split(
            indices, test_size=final_test_size, random_state=self.seed, shuffle=shuffle
        )

        self.X_train_list = [features[train_indices] for features in self.preped_features]
        self.X_val_list = [features[val_indices] for features in self.preped_features]
        self.y_train_list = [target[train_indices] for target in self.preped_targets]
        self.y_val_list = [target[val_indices] for target in self.preped_targets]


    def search(self, features: Union[pd.DataFrame, dict], targets: pd.DataFrame, test_size: float):
        """
        Prepares features and targets for the AutoModel, and then performs the search/fit
        on the AutoModel.

        Args:
        - feature (pd.DataFrame | dict): Input data for training
        - targets (pd.DataFame): Output data for validation
        - test_size (float): size of the validation split
        """
        self.prepare_data(features, targets, test_size)

        self.auto_model.fit(
            self.X_train_list,
            self.y_train_list,
            validation_data=(self.X_val_list, self.y_val_list),
            **self.fit_config
        )


    def generate_report(
        self, cv: PredefinedSplit,
        metrics: MetricsHandler,
        store_top_num_models: Union[int, float] = 0.1,
        eval_top_num_models: Union[int, float] = 1.0
    ) -> dict:
        """
        Collect the data for the report by retraining the models for each cv split, and
        then and saves the best models:

        Args:
        - cv (PredefinedSplit): a split that returns the indicies for each split.
        - metrics (MetricsHandler): handles the function used to eval the metrics.
        - store_top_num_models (int | float): Specifies the number or fraction of top models to store.
            Models are stored to model_save_dir_path. Defaults to 0.1.
          - If an integer is provided, it save that many of the top models.
          - If a float between 0 and 1 is provided it saves that fraction of the top models.
        - evaluate_top_num_models (int|float): Same as store_top_num_models, except that is sets how
          many modles to evaluate (retrain on each cv split).

        Returns:
        - dict: the report.
        """

        self.report = self._initialize_report()
        save_top_n = self._get_n_to_save(store_top_num_models)
        eval_top_n = self._get_n_to_save(eval_top_num_models) #works the same
        self._prepare_model_save_directory()

        for idx, (model, hyperparameters) in enumerate(self._get_best_models(save_top_n)):
            print(f"\nProcessing model {idx + 1}")

            optimizer = model.optimizer
            del model
            self._process_model(idx, optimizer, hyperparameters, cv, metrics)

        return self.report


    def _initialize_report(self):
        """
        Initializes the report structure.
        """
        return {
            "model_data": {"model_reports": [], "model_metadata": None},
            "split_data": {"splits": [], "split_metadata": None}
        }


    def _get_n_to_save(self, store_top_num_models: Union[int, float]) -> int:
        """
        Calculates the number of models to save, while making sure the value is in the range [1, n_models].
        When int saves that number of models, when float saves that fraction of the models.
        Reused for eval_top_n_to_save.

        Args:
        - store_top_num_models (int | float): number or fratcion of models to store.

        Returns:
        - int: the number of models to store
        """
        n_candidates = len(self.auto_model.tuner.oracle.trials)
        if isinstance(store_top_num_models, int) or (isinstance(store_top_num_models, float) and store_top_num_models.is_integer()):
            save_top_n = min(int(store_top_num_models), n_candidates)
        elif isinstance(store_top_num_models, float) and 0.0 < store_top_num_models <= 1.:
            save_top_n = int(n_candidates * store_top_num_models)
        else:
            save_top_n = 1
        return max(1, min(save_top_n, n_candidates))


    def _prepare_model_save_directory(self):
        """
        Makes sure that the directory exsists and that it's empty.
        """
        self.model_save_dir_path.mkdir(parents=True, exist_ok=True)
        empty_directory(self.model_save_dir_path)


    def _get_best_models(self, num_models: int) -> Generator[tuple[Model, HyperParameters], None, None]:
        """
        Loads the best models in order of their ranking.

        Args:
        - num_models (int): the number of models to load.

        Returns:
        - Generator[tuple[Model, HyperParameters], None, None]: returns a generator to avoid loading all the models at once,
          - the first output is the loaded moddel,
          - the second output is the hyperparameters that were used to generate that model.
        """
        top_trials = self.auto_model.tuner.oracle.get_best_trials(num_models)
        for trial in top_trials:
            model = self.auto_model.tuner.load_model(trial)
            yield model, trial.hyperparameters


    def _process_model(self, idx: int, optimizer: Optimizer, hyperparameters: HyperParameters, cv: PredefinedSplit, metrics: MetricsHandler):
        """
        Retrains and evaluated the model for each cross validation split and collects the relevant data,
        while ensuring an equal starting point for each split.

        Args:
        - idx (int): index (ranking) of the model.
        - optimizer (Optimizer): optimizer from the pretrained model, to use for generating new optimizers for the new model.
        - hyperparameters (HyperParameters): hyperparameters to recreate the model.
        - cv (PredefinedSplit): a predefined split that returns indicies.
        - metrics (MetricsHandler): handles getting the functions used for calculating metrics.
        """
        scores = {name: [] for name in metrics.metrics_names}
        model_data_per_split, train_times, predict_times, splits = [], [], [], []

        for split_idx, (train_indices, test_indices) in enumerate(cv.split(self.preped_features[0], self.preped_targets[0])):
            print(f"\tProcessing split {split_idx + 1} out of {cv.get_n_splits()}")

            # Ensure the starting point is the same
            model = self._reset_training(optimizer, hyperparameters)

            save_path = os.path.join(self.model_save_dir_path, f"model-{idx}-{split_idx}.keras")
            model.save(save_path)
            model_size = os.path.getsize(save_path)

            X_train, X_test, y_train, y_test = self._prepare_split_data(train_indices, test_indices)

            train_time = self._train_model(model, X_train, y_train, X_test, y_test)
            train_times.append(train_time)

            y_pred, predict_time = self._predict(model, X_test)
            predict_times.append(predict_time)

            model_data_per_split.append({
                "model_path": save_path,
                "y_pred": y_pred,
                "y_true": y_test,
                "model_size": 0
            })

            self._update_scores(scores, metrics, y_test, y_pred)
            splits.append(test_indices)

            del model
            keras.backend.clear_session()
            gc.collect()

        ###
        self._add_model_report(model_data_per_split, scores, hyperparameters, train_times, predict_times)
        self.report["split_data"]["splits"] = splits


    def _reset_training(self, optimizer: Optimizer, hyperparameters: HyperParameters) -> Model:
        """
        Ensures each training run has the same starting point by setting the seed and build a new model.

        Args:
        - optimizer (Optimizer): used to rebuild a new optimizer.
        - hyperparameters (HyperParameters): use to rebuild an untrained model.

        Returns:
        - Model: an untrained model
        """
        keras.utils.set_random_seed(self.seed)
        model = self.auto_model.tuner.hypermodel.build(hyperparameters)
        new_optimizer = type(optimizer).from_config(optimizer.get_config())
        model.compile(optimizer=new_optimizer, loss='mse')
        return model


    def _prepare_split_data(self, train_indices, test_indices):
        """
        Handles getting the indicies for training and testing, and returning the data
        in a format supported by AutoModel.

        Args:
        - train_indicies: indicies of the dataset to be used for training
        - test_indicies: indicies of the dataset to be used for testing

        Returns:
        - tuple[list[np.ndarray]] - returns X and y for training and validation
          - X_train, X_test, y_train, y_test
        """
        X_train = [safe_indexing(x, train_indices) for x in self.preped_features]
        X_test = [safe_indexing(x, test_indices) for x in self.preped_features]
        y_train = [safe_indexing(y, train_indices) for y in self.preped_targets]
        y_test = [safe_indexing(y, test_indices) for y in self.preped_targets]
        return X_train, X_test, y_train, y_test


    def _train_model(self, model, X_train, y_train, X_test, y_test):
        """
        Trains the model and measures the time it took.

        Args:
        - model: the untrained model that is then trained.
        - X_train, X_test: the input data for the model for training and testing.
        - y_trian, y_test: the output data for the model for training and testing.

        Returns:
        - The time it took to train the model-
        """
        start_time = time.perf_counter()
        model.fit(X_train, y_train, validation_data=(X_test, y_test), **{**self.fit_config, "verbose": 2}) # Verbosity 2 is less than 1 (0<2<1)
        return time.perf_counter() - start_time


    def _predict(self, model: Model, X_test) -> tuple[np.ndarray, float]:
        """
        Predicts using the trained model and measures the time it took.

        Args:
        - model: the trianed model
        - X_test: the input data for the model for which to predict the values.

        Returns:
        - tuple[np.ndarray, float]
        """
        start_time = time.perf_counter()
        print(X_test[0].shape)
        y_pred = model.predict(X_test, batch_size=self.fit_config.get("batch_size", 32))
        predict_time = time.perf_counter() - start_time
        return np.squeeze(np.stack(y_pred, axis=0), axis=-1), predict_time


    def _update_scores(self, scores: list, metrics: MetricsHandler, y_test, y_pred):
        """
        Calculates each metrics.

        Args:
        - scores (list): a list of the scores
        - metrics (MetricsHandler): provides the functions to calculate the metrics.
        - y_test: true values.
        - y_pred: predicted values
        """
        y_test = np.squeeze(np.stack(y_test, axis=0), axis=-1)
        for name, func in metrics.metrics.items():
            scores[name].append(func(y_test, y_pred))


    def _add_model_report(self, model_data_per_split: list, scores: list, hyperparameters: HyperParameters, train_times: list, predict_times: list):
        """
        Makes a report for each model.

        Args:
        - save_paths (list): path to the model saved for each split.
        - scores (list): scores for each split.
        - hyperparameters (HyperParameters): the hyperparameters used for the model.
        - train_times (list): the time it took to train for each split.
        - test_time (list): the time it took the test for each split
        """
        model_report = {
            "model_data_per_split": model_data_per_split,
            "scores": {name: {"mean": np.mean(arr), "std": np.std(arr)} for name, arr in scores.items()},
            "params": hyperparameters.values,
            "fit_time": {"mean": np.mean(train_times), "std": np.std(train_times)},
            "score_time": {"mean": np.mean(predict_times), "std": np.std(predict_times)},
        }
        self.report["model_data"]["model_reports"].append(model_report)


    def cleanup_tmp(self):
        """
        Cleans up the temporary directory (preserves '.gitignore').
        """
        self.tmp_dir_path = os.path.join(self.tmp_dir_path, "..")
        for item in os.listdir(self.tmp_dir_path):
            if item == ".gitignore":
                continue

            item_path = os.path.join(self.tmp_dir_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path):
                os.remove(item_path)


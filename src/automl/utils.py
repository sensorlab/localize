import importlib
from typing import Callable, Union

import numpy as np
import pandas as pd


class utils:
    @staticmethod
    def get_class(module_name: str, class_name: str):
        """Import module and return the required class."""
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    @staticmethod
    def parse_choice_param(key, value, hp):
        """Parse choice parameters for discrete values."""
        return hp.Choice(key, value["values"])

    @staticmethod
    def parse_range_param(key, value, hp):
        """Parse range parameters for integer and float values."""
        min_val = value.get("min", 1 if isinstance(value["max"], int) else 0.0)
        max_val = value["max"]

        if isinstance(min_val, int) and isinstance(max_val, int):
            # Integer range
            return hp.Int(key, min_value=min_val, max_value=max_val, step=value.get("step", 1))
        elif isinstance(min_val, float) and isinstance(max_val, float):
            # Float range
            range_step = value.get("step", None)
            if range_step is None:
                range_step = 0.1 if value.get("sampling", "linear") == "linear" else 10.0
            return hp.Float(
                key, min_value=min_val, max_value=max_val, step=range_step, sampling=value.get("sampling", "linear")
            )
        else:
            raise ValueError(f"Unsupported range type for key: {key}\n{value}")

    @staticmethod
    def parse_class_param(value):
        """
        Parse class parameters.
        Allows for parssing a class whe no args specified, or an object to which
        the args dict is passed as kwargs.
        """
        param_module = value.get("module", "keras_tuner")
        param_class = value["class"]

        imported_class = utils.get_class(param_module, param_class)
        param_args = value.get("args", None)
        if param_args is not None:
            return imported_class(**param_args) if isinstance(param_args, dict) else imported_class()
        return imported_class

    @staticmethod
    def parse_class_list_param(value):
        """Parse list of class parameters."""
        parsed_list = []
        for arg in value:
            parsed_list.append(utils.parse_class_param(arg))
        return parsed_list

    @staticmethod
    def to_numpy(data: pd.DataFrame | np.ndarray):
        """
        Converts to numpy array.
        """
        if isinstance(data, pd.DataFrame):
            return data.to_numpy()
        return data

    @staticmethod
    def parse_args(args, hp):
        """Parse hyperparameter arguments."""

        parsed_args = {}

        for key, value in args.items():
            if isinstance(value, dict):
                if "values" in value:
                    parsed_args[key] = utils.parse_choice_param(key, value, hp)
                elif "max" in value:
                    parsed_args[key] = utils.parse_range_param(key, value, hp)
                elif "class" in value:
                    parsed_args[key] = utils.parse_class_param(value)
                else:
                    raise ValueError(f"Unsupported hyperparameter configuration for key: {key}")
            elif isinstance(value, list) and len(value) != 0 and isinstance(value[0], dict) and "class" in value[0]:
                parsed_args[key] = utils.parse_class_list_param(value)
            elif type(value) in ["int", "str", "bool"]:
                parsed_args[key] = hp.Fixed(key, value)
            else:
                parsed_args[key] = value

        return parsed_args

    @staticmethod
    def contains_instance(instance_list, instance_type):
        """Checks the list if an instance of a specified type is present."""
        return any([isinstance(instance, instance_type) for instance in instance_list])

    @staticmethod
    def select_columns(
        columns: list[str], func: Callable[[str, str], bool], search_str: Union[list[str], str]
    ) -> list[str]:
        """Select columns specified by the function, and modify the function to operate on lists."""
        if isinstance(search_str, list):

            def wrapped_func(col, vals):
                return any(func(col, val) for val in vals)

            func = wrapped_func

        return [col for col in columns if func(col, search_str)]

from typing import Callable, Union

import numpy as np
import sklearn


def mean_euclidean_distance_error(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1)).mean()


custom_metrics = {"mean_euclidean_distance_error": mean_euclidean_distance_error}

# Abbreviation to full-name mappings
metrics_mappings = {
    "mede": "mean_euclidean_distance_error",
    "rmse": "root_mean_squared_error",
    "mae": "median_absolute_error",
    "r_squared": "r2_score",
    "mape": "mean_absolute_percentage_error",
}


class MetricsHandler:
    def __init__(self, metrics: Union[dict[str, bool], list[str]]):
        self.metrics_names = metrics
        self.metrics = {name: self.get_metric(name) for name in metrics}

    def get_metric(self, metric_name: str) -> Callable:
        metric_full_name = metrics_mappings.get(metric_name.lower(), metric_name)

        if metric_full_name in sklearn.metrics.__all__:
            return getattr(sklearn.metrics, metric_full_name)

        if metric_full_name in custom_metrics:
            return custom_metrics.get(metric_full_name)

        raise ValueError(f"No metric with the name '{metric_full_name}' exists.")

    def to_GridsearchCV(self):
        if isinstance(self.metrics_names, list):
            self.metrics_names = {name: False for name in self.metrics_names}

        scorers = {}
        for name, metric_func in self.metrics.items():
            scorers[name] = sklearn.metrics.make_scorer(metric_func, greater_is_better=self.metrics_names[name])

        return scorers

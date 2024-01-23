import json
from pathlib import Path
from typing import Literal, Iterable

import click
import numpy as np
from sklearn import metrics

from src import load_data, safe_indexing
from sklearn.base import BaseEstimator

from collections import defaultdict


def merge_dicts(items: Iterable[dict[str, float]]) -> dict[str, list[float]]:
    result = defaultdict(lambda: [])
    for item in items:
        for k, v in item.items():
            result[k].append(v)

    return result


def summarize_scores(metrics: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    result = {}
    for k, v in metrics.items():
        result[k] = {
            "mean": np.mean(v),
            "std": np.std(v),
            "median": np.median(v),
            "iqr": np.percentile(v, 75) - np.percentile(v, 25),
            "min": np.min(v),
            "max": np.max(v),
        }

    return result

    #         "mean": 0.95,
    #         "std_dev": 0.02,
    #         "median": 0.95,
    #         "iqr": 0.03,
    #         "best": 0.98,
    #         "worst": 0.90


# @click.command()
# @click.option(
#     "--model",
#     "data",
#     type=click.Path(exists=True, dir_okay=False, path_type=Path),
#     required=True,
#     help='Path to the trained model',
# )
# @click.option(
#     "--data",
#     "data",
#     type=click.Path(exists=True, dir_okay=False, path_type=Path),
#     required=True,
#     help='Path to test data',
# )
# @click.option(
#     "--indices",
#     "indices",
#     type=click.Path(exists=True, dir_okay=False, path_type=Path),
#     required=True,
#     help='Path to indices for dataset',
# )
# @click.option(
#     "--task",
#     "task",
#     type=click.Choice(['regression'], case_sensitive=False),
#     required=True,
#     help='Select type of evaluation',
# )
# @click.option(
#     "--metrics-output",
#     "metrics_output",
#     type=click.Path(dir_okay=False, path_type=Path),
#     required=True,
#     help='Path to save metrics',
# )
# def task(model: Path, data: Path, indices: Path, task: str, metrics_output: Path):
#     model = load_data(model)
#     data = load_data(data)
#     indices = load_data(indices)

#     # Ensure that output folder exists
#     metrics_output.parent.mkdir(parents=True, exist_ok=True)

#     if task == "regression":


# VALID_TASKS = ("regression", "classification")


@click.command()
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--data",
    "data_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--split-indices",
    "split_indices_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output-summary",
    "--summary",
    "output_summary_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=True,
)
@click.option(
    "--task",
    "task",
    type=click.Choice(["regression", "classification"], case_sensitive=False),
    default="regression",
    help="Specify the model type: regression or classification.",
)
def cli(
    model_path: Path,
    data_path: Path,
    split_indices_path: Path,
    output_summary_path: Path,
    task: Literal["classification", "regression"],
):
    # Load the final model
    model: BaseEstimator = load_data(model_path)

    indices = load_data(split_indices_path)

    X, y = load_data(data_path)

    scores = []

    for _, test_idx in indices:
        # X_train = safe_indexing(X, train_idx)
        # y_train = safe_indexing(y, train_idx)

        X_test = safe_indexing(X, test_idx)
        y_test = safe_indexing(y, test_idx)

        if task == "regression":
            y_pred = model.predict(X_test)
            score = {
                "r_squared": metrics.r2_score(y_test, y_pred),
                # TODO: add adjusted_r2, but requires number of samples and independet variables.
                "mae": metrics.mean_absolute_error(y_test, y_pred),
                "rmse": metrics.mean_squared_error(y_test, y_pred, squared=False),
                # L2 == Euclidean distance
                "euclidean": np.sqrt(np.sum((y_test - y_pred) ** 2, axis=1)).mean(),
            }

            scores.append(score)

        else:
            raise NotImplementedError

    # TODO: Overall analysis ... then include all metrics into details or something.
    # TODO: Rename to summary

    # Merge dictionaries
    scores = merge_dicts(scores)
    summary = summarize_scores(scores)

    # {
    #     "accuracy": {
    #         "mean": 0.95,
    #         "std_dev": 0.02,
    #         "median": 0.95,
    #         "iqr": 0.03,
    #         "best": 0.98,
    #         "worst": 0.90
    #     },
    #     "precision": {
    #         "mean": 0.94,
    #         "std_dev": 0.03,
    #         // similarly for median, iqr, best, worst
    #     },
    #     // Repeat for other metrics
    # }

    # output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary_path, mode="w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)


if __name__ == "__main__":
    cli()

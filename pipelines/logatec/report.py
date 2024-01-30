import click
from pathlib import Path
from src import load_data, save_params
from typing import Any
import numpy as np
from sklearn import metrics


@click.command()
@click.option("--input", "input_path", type=click.Path(path_type=Path))
@click.option("--output", "--output-report", "output_path", type=click.Path(path_type=Path))
def cli(
    input_path: Path,
    output_path: Path,
):
    report: dict[str, Any] = load_data(input_path)

    # TODO: Povprečen MSE
    y_true = np.concatenate(report["predictions"]["y_true"], axis=0)
    y_pred = np.concatenate(report["predictions"]["y_pred"], axis=0)

    # Calculate error distance

    # rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))

    # d = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))

    report = {
        "rmse": metrics.mean_squared_error(y_true, y_pred, squared=False),
        "euclidean": np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1)).mean(),
        "r_squared": metrics.r2_score(y_true, y_pred),
        "mae": metrics.mean_absolute_error(y_true, y_pred),
        # "rmse": {
        #     "mean": np.mean(rmse),
        #     "std": np.std(rmse),
        #     "median": np.median(rmse),
        #     "iqr": np.percentile(rmse, 75) - np.percentile(rmse, 25),
        #     "min": np.min(rmse),
        #     "max": np.max(rmse),
        # },
        # "euclidean": {
        #     "mean": np.mean(d),
        #     "std": np.std(d),
        #     "median": np.median(d),
        #     "iqr": np.percentile(d, 75) - np.percentile(d, 25),
        #     "min": np.min(d),
        #     "max": np.max(d),
        # },
        # "r_squared": {
        # }
    }

    # euclidean = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=-1)).mean()

    # mse = metrics.mean_squared_error(y_true, y_pred)

    save_params(report, output_path)

    # print(len(reports))

    # y_true = np.concatenate([r["predictions"]["y_true"] for r in reports], axis=0)
    # y_pred = np.concatenate([r["predictions"]["y_pred"] for r in reports], axis=0)

    # print(y_true.shape)

    # train_time = np.asarray([r["model_metadata"]["training_time"] for r in reports])

    # print(metrics.mean_squared_error(y_true, y_pred), np.mean(train_time))


if __name__ == "__main__":
    cli()

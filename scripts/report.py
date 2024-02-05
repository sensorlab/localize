import click
from pathlib import Path
from src import save_params
from typing import Any
import numpy as np
from sklearn import metrics
from src.metrics import mean_percentage_error
import joblib


@click.command()
@click.argument("input_reports", nargs=-1, type=click.Path(path_type=Path))
@click.option("--output", "--output-report", "output_path", type=click.Path(path_type=Path))
def cli(
    input_reports: list[Path],
    output_path: Path,
):
    output: dict[str, dict[str, dict[str, Any]]] = {}

    for report in input_reports:
        report = joblib.load(report, mmap_mode=None)

        algorithm = report["model_metadata"]["algorithm"]
        split_type = report["split_metadata"]["split_type"]

        y_true = np.concatenate(report["predictions"]["y_true"], axis=0)
        y_pred = np.concatenate(report["predictions"]["y_pred"], axis=0)

        # Calculate loss metrics
        losses = {
            "rmse": metrics.root_mean_squared_error(y_true, y_pred),
            "euclidean": np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1)).mean(),
            "r_squared": metrics.r2_score(y_true, y_pred),
            "mae": metrics.mean_absolute_error(y_true, y_pred),
            "mpe": mean_percentage_error(y_true, y_pred),
        }

        output[algorithm] = output.get(algorithm, {})

        output[algorithm][split_type] = losses

    if output_path:
        save_params(output, output_path)


if __name__ == "__main__":
    cli()

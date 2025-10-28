from pathlib import Path

import click
import joblib
import numpy as np
from sklearn import model_selection

from src import PredefinedSplit, load_params


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the raw dataset.",
)
@click.option(
    "--output-indices",
    "--indices",
    "output_indices_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=True,
)
@click.option(
    "--split",
    type=click.Choice(["KFold", "Random", "LongEdge", "ShortEdge", "Cutout"], case_sensitive=True),
    default="KFold",
    required=True,
    show_default=True,
    # help="Method to harmonize values",
)
def cli(input_path: Path, output_indices_path: Path, split: str):
    # Load parameters
    params = load_params("./params.yaml")

    split_params = params["split"]
    random_state = split_params["seed"]

    # Load dataset
    _, targets = joblib.load(input_path)
    groups = None

    # TODO: ...

    match split:
        case "Random":
            cv = model_selection.ShuffleSplit(
                n_splits=split_params["n_splits"],
                test_size=split_params["test_size"],
                random_state=random_state,
            )

        case "kfold" | "KFold":
            cv = model_selection.KFold(
                n_splits=split_params["n_splits"],
                shuffle=True,
                random_state=random_state,
            )

        case "ShortEdge":
            x, y = targets[..., 0], targets[..., 1]
            threshold = -1.0 * x + 2.0
            train_mask = y > threshold
            train_idx = np.argwhere(train_mask).flatten()
            test_idx = np.argwhere(~train_mask).flatten()
            cv = PredefinedSplit(indices_pairs=((train_idx, test_idx),))

        case "LongEdge":
            x, y = targets[..., 0], targets[..., 1]
            threshold = 0.9 * x + 4.1
            train_mask = y < threshold
            train_idx = np.argwhere(train_mask).flatten()
            test_idx = np.argwhere(~train_mask).flatten()
            cv = PredefinedSplit(indices_pairs=((train_idx, test_idx),))

        case "Cutout":
            x, y = targets[..., 0], targets[..., 1]
            radius = 0.5
            train_mask = np.sqrt((y - 4) ** 2 + (x - 0.8) ** 2) > radius
            train_idx = np.argwhere(train_mask).flatten()
            test_idx = np.argwhere(~train_mask).flatten()
            cv = PredefinedSplit(indices_pairs=((train_idx, test_idx),))

        case _:
            raise NotImplementedError('Split type "{split}" not implemented')

    indices = []

    for train_indices, test_indices in cv.split(targets, groups=groups):
        indices.append((train_indices, test_indices))

    indices_list = {
        "indices": tuple(indices),
        "metadata": {"split_type": split},
    }

    joblib.dump(indices_list, output_indices_path)


if __name__ == "__main__":
    cli()

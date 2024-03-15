from pathlib import Path

import click
import joblib
import numpy as np
from sklearn import model_selection
from src import load_params


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
    type=click.Choice(["Random"], case_sensitive=True),
    default="Random",
    required=True,
    show_default=True,
    # help="Method to harmonize values",
)
def cli(input_path: Path, output_indices_path: Path, split: str):
    # Load parameters
    params = load_params("./params.yaml")

    random_state = params["seed"]
    split_params = params["split"]

    # Load dataset
    _, targets = joblib.load(input_path)
    _, groups = np.unique(targets, axis=0, return_inverse=True)

    assert len(targets) == len(groups), (len(targets), len(groups))

    match split:
        case "Random":
            cv = model_selection.GroupShuffleSplit(
                n_splits=split_params["n_splits"],
                test_size=split_params["test_size"],
                random_state=random_state,
            )

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

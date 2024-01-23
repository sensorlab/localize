from pathlib import Path
from typing import Literal

import click
import joblib
from sklearn import model_selection
from src import load_data, load_params


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
    type=click.Choice(["random", "kfold"], case_sensitive=False),
    default="kfold",
    required=True,
    show_default=True,
    # help="Method to harmonize values",
)
@click.option(
    "--params",
    "params_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the YAML with parameters",
)
def cli(input_path: Path, output_indices_path: Path, params_path: Path, split: Literal["random", "kfold"]):
    # Load parameters
    params = load_params(params_path)

    random_state = params["seed"]
    sparams = params["split"]

    # Load dataset
    features, targets = load_data(input_path)

    if split == "random":
        cv = model_selection.ShuffleSplit(
            n_splits=1,
            test_size=sparams["test_size"],
            random_state=random_state,
        )

    elif split == "kfold":
        cv = model_selection.KFold(
            n_splits=sparams["n_splits"],
            shuffle=True,
            random_state=random_state,
        )

    else:
        raise NotImplementedError('Split type "{split}" not implemented')

    subsets = []

    # for idx, (train_idx, test_idx) in enumerate(cv.split(features, targets)):
    #    subsets.append((train_idx, test_idx))

    subsets = list((train_idx, test_idx) for train_idx, test_idx in cv.split(features, targets))

    joblib.dump(subsets, output_indices_path)


if __name__ == "__main__":
    cli()

from pathlib import Path
from typing import Literal, Union

import click
import joblib
import numpy as np
import pandas as pd
from sklearn import model_selection
from src import load_data, load_params


class LeaveOneLabelCombinationOut(model_selection.BaseCrossValidator):
    def __init__(self, targets: Union[np.array, pd.DataFrame], random_state=None):
        # Create a unique identifier for each combination of labels
        if hasattr(targets, "iloc"):
            targets: np.array = targets.to_numpy()

        self.uniques = np.unique(targets, axis=0)
        self.random_state = random_state

    def get_n_splits(self, X, y=None, groups=None):
        return len(self.uniques)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""

        rng = np.random.default_rng(self.random_state)

        # TODO: I don't know how to do it on pandas dataframe.
        if hasattr(y, "iloc"):
            y: np.array = y.to_numpy()

        for target in self.uniques:
            mask = (y == target).all(axis=-1)
            train_idx, test_idx = np.where(~mask)[0], np.where(mask)[0]

            rng.shuffle(train_idx)
            # rng.shuffle(test_idx)

            yield train_idx, test_idx


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
    type=click.Choice(["random", "kfold", "exclude"], case_sensitive=False),
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
    split_params = params["split"]

    # Load dataset
    features, targets = load_data(input_path)

    match split:
        case "random" | "Random":
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

        case "exclude" | "OneLabelOut":
            cv = LeaveOneLabelCombinationOut(
                targets,
                random_state=random_state,
            )

        case _:
            raise NotImplementedError('Split type "{split}" not implemented')

    subsets = []

    # for idx, (train_idx, test_idx) in enumerate(cv.split(features, targets)):
    #    subsets.append((train_idx, test_idx))

    subsets = list((train_idx, test_idx) for train_idx, test_idx in cv.split(features, targets))

    joblib.dump(subsets, output_indices_path)


if __name__ == "__main__":
    cli()

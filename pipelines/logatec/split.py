from pathlib import Path

import click
import joblib
import numpy as np
from sklearn import model_selection
from src import load_data, load_params


class CombineCVs(model_selection.BaseCrossValidator):
    """TODO: Not tested."""

    def __init__(self, cvs: list[model_selection.BaseCrossValidator]) -> None:
        self.cvs = cvs

    def split(self, X, y=None, groups=None):
        X, y, groups = model_selection._split.indexable(X, y, groups)
        for cv in self.cvs:
            for train_index, test_index in cv.split(X, y, groups):
                yield train_index, test_index


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
    type=click.Choice(["KFold", "Random", "LeaveOneGroupOut"], case_sensitive=False),
    default="kfold",
    required=True,
    show_default=True,
    # help="Method to harmonize values",
)
# @click.option(
#     "--params",
#     "params_path",
#     type=click.Path(exists=True, dir_okay=False, path_type=Path),
#     help="Path to the YAML with parameters",
# )
def cli(input_path: Path, output_indices_path: Path, split: str):
    # Load parameters
    params = load_params("./params.yaml")

    random_state = params["seed"]
    split_params = params["split"]

    # Load dataset
    features, targets = load_data(input_path)
    groups = None

    match split:
        case "Random":
            cv = model_selection.ShuffleSplit(
                n_splits=split_params["n_splits"],
                test_size=split_params["test_size"],
                random_state=random_state,
            )

        case "KFold":
            cv = model_selection.KFold(
                n_splits=split_params["n_splits"],
                shuffle=True,
                random_state=random_state,
            )

        case "LeaveOneGroupOut":
            _targets = targets.to_numpy() if hasattr(targets, "iloc") else targets
            groups = np.unique(_targets, axis=0, return_inverse=True)[1]
            cv = model_selection.LeaveOneGroupOut()

        case _:
            raise NotImplementedError(f'Split type "{split}" not implemented')

    subsets = []

    # for idx, (train_idx, test_idx) in enumerate(cv.split(features, targets)):
    #    subsets.append((train_idx, test_idx))

    subsets = list((train_idx, test_idx) for train_idx, test_idx in cv.split(features, targets, groups))

    joblib.dump(subsets, output_indices_path)


if __name__ == "__main__":
    cli()

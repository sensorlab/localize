from pathlib import Path

import click
import joblib
import numpy as np
from sklearn import model_selection

import os
import sys
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    print(project_root)
    sys.path.insert(0, project_root)

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

    # subsets = []

    # for idx, (train_idx, test_idx) in enumerate(cv.split(features, targets)):
    #    subsets.append((train_idx, test_idx))

    # indices_list = []

    # for idx, (train_indices, test_indices) in enumerate(cv.split(features, targets, groups)):
    #     indices_list.append({
    #         "indices": (train_indices, test_indices),
    #         "metadata": {"split_type": split, "fold_idx": idx}
    #     })

    # indices_list = (
    #    (train_indices, test_indices, {"split_type": split, "split_idx": idx})
    #    for idx, (train_indices, test_indices) in enumerate(cv.split(features, targets, groups))
    # )

    indices = []
    # metadata = []

    for train_indices, test_indices in cv.split(features, targets, groups):
        indices.append((train_indices, test_indices))
        # metadata.append({"split_type": split, "split_idx": idx})

    indices_list = {
        "indices": tuple(indices),
        "metadata": {"split_type": split},
    }

    # subsets = list((train_idx, test_idx) for train_idx, test_idx in cv.split(features, targets, groups))

    joblib.dump(indices_list, output_indices_path, compress=9)


if __name__ == "__main__":
    cli()

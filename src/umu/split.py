from pathlib import Path

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
    type=click.Choice(["KFold", "Random", "LeaveOneGroupOut"], case_sensitive=False),
    default="kfold",
    required=True,
    show_default=True,
)
def cli(input_path: Path, output_indices_path: Path, split: str):
    # Load parameters
    params = load_params("./params.yaml")

    split_params = params["split"]
    random_state = split_params["seed"]

    # Load dataset
    features, targets = load_data(input_path)
    groups = None

    match split:
        case "KFold":
            cv = model_selection.KFold(
                n_splits=split_params["n_splits"],
                shuffle=True,
                random_state=random_state,
            )

        case _:
            raise NotImplementedError(f'Split type "{split}" not implemented')

    indices = []

    for train_indices, test_indices in cv.split(features, targets, groups):
        indices.append((train_indices, test_indices))

    indices_list = {
        "indices": tuple(indices),
        "metadata": {"split_type": split},
    }

    joblib.dump(indices_list, output_indices_path)


if __name__ == "__main__":
    cli()

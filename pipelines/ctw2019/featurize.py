from pathlib import Path

import click
import joblib

import numpy as np


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=True,
)
# @click.option(
#     "--adapt",
#     "--adapt-for",
#     "adaptation",
#     type=click.Choice(["concat"], case_sensitive=False),
#     help="Adapt dataset for certain framework",
# )
def cli(input_path: Path, output_path: Path):
    data: dict[str, np.ndarray] = joblib.load(input_path)
    h, snr, pos = data["h"], data["snr"], data["pos"]

    # Fix #1: We'll focus only on X and Y coordinates. Delete Z axis.
    targets = np.delete(pos, -1, axis=-1)
    assert targets.shape == (pos.shape[0], 2)

    # Fix #2: Swap axes from BHWC --> BCHW
    h = np.transpose(h, (0, 3, 1, 2))

    # TODO: Do some additional feature engineering.

    features = {"h": h, "snr": snr}

    # Save newly transformed dataset
    joblib.dump((features, targets), output_path)


if __name__ == "__main__":
    cli()

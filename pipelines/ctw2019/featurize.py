from pathlib import Path

import click
import joblib
from src import load_data

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
#     type=click.Choice(["sklearn", "pytorch"], case_sensitive=False),
#     help="Adapt dataset for certain framework",
# )
def cli(input_path: Path, output_path: Path):
    data: dict[str, np.ndarray] = load_data(input_path)
    h, snr, pos = data["h"], data["snr"], data["pos"]

    # Fix #1: We'll focus only on X and Y coordinates. Delete Z axis.
    targets = np.delete(pos, -1, axis=-1)
    assert targets.shape == (pos.shape[0], 2)

    # if adaptation.lower() == "sklearn":
    #     h_flat = h.reshape(h.shape[0], -1)
    #     features = np.concatenate((h_flat, snr), axis=1)
    #     assert features.shape == (h.shape[0], 16 * 924 * 2 + 16)

    # elif adaptation.lower() == "pytorch":
    #     features = {"h": h, "snr": snr}

    # else:
    #     raise NotImplementedError('Data adaptation for "{adaptation}" not implemented!')

    features = {"h": h, "snr": snr}

    # TODO: Do some additional feature engineering.

    # Save newly transformed dataset
    joblib.dump((features, targets), output_path)


if __name__ == "__main__":
    cli()

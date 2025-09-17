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
def cli(input_path: Path, output_path: Path):
    features, targets = joblib.load(input_path)
    h, snr = features["h"], features["snr"]

    # Modifications to dataset for machine learning
    h_flat = h.reshape(h.shape[0], -1)
    features = np.concatenate((h_flat, snr), axis=1, dtype=np.float32)
    assert features.shape == (h.shape[0], 56 * (924 * 2 + 1))

    # Save newly transformed dataset
    joblib.dump((features, targets), output_path)


if __name__ == "__main__":
    cli()

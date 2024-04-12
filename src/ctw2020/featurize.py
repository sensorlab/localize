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
    data: dict[str, np.ndarray] = joblib.load(input_path)
    H_Re, H_Im, SNR, Pos = data["h_re"], data["h_im"], data["snr"], data["pos"]

    h_re = []
    h_im = []
    snr = []
    pos = []

    n_samples = SNR.shape[-1]

    for r, i, s, p in zip(H_Re, H_Im, SNR, Pos, strict=True):
        for idx in range(n_samples):
            h_re.append(r[..., idx])
            h_im.append(i[..., idx])
            snr.append(s[..., idx])
            pos.append(p)

    h_re = np.stack(h_re, axis=0)
    h_im = np.stack(h_im, axis=0)
    snr = np.stack(snr, axis=0)
    pos = np.stack(pos, axis=0)

    h = np.stack([h_re, h_im], axis=-1)

    print(f"{h.shape=} {snr.shape=} {pos.shape=}")

    # Fix #1: We'll focus only on X and Y coordinates. Delete Z axis.
    targets = np.delete(pos, -1, axis=-1)
    assert targets.shape == (pos.shape[0], 2), pos.shape

    assert h.shape[1:] == (56, 924, 2)

    # TODO: Do some additional feature engineering.

    print(f"{h.shape=} {snr.shape=} {pos.shape=}")

    features = {"h": h, "snr": snr}

    # Save newly transformed dataset
    joblib.dump((features, targets), output_path)


if __name__ == "__main__":
    cli()

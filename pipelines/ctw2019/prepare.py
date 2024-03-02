from pathlib import Path

import click
import h5py
import joblib
import numpy as np

ANTENNA_OFFSET = np.asarray([3.5, -3.15, 1.8], dtype=np.float32)


@click.command()
@click.option(
    "--data-h",
    "--input-h",
    "data_h_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--data-snr",
    "--input-snr",
    "data_snr_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--data-position",
    "--data-pos",
    "--input-position",
    "data_pos_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output",
    "dataset_output_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=True,
    help="Path to save processed dataset.",
)
# @click.option(
#     "--task",
#     type=click.Choice(["classification", "regression"], case_sensitive=False),
#     default="regression",
#     show_default=True,
#     help="What is the target value",
# )
def cli(
    data_h_path: Path,
    data_snr_path: Path,
    data_pos_path: Path,
    dataset_output_path: Path,
    # task: Literal["classification", "regression"],
):
    with h5py.File(data_h_path, mode="r", swmr=True) as fp:
        h = fp["h_Estimated"][:].astype(np.float32).T

    with h5py.File(data_snr_path, mode="r", swmr=True) as fp:
        snr = fp["SNR_Est"][:].astype(np.float32).T

    with h5py.File(data_pos_path, mode="r", swmr=True) as fp:
        pos = fp["r_Position"][:].astype(np.float32).T

    # Fix #1: Correction of position data. Antenna will now be in the center.
    pos = pos - ANTENNA_OFFSET

    # Fix #2: remove measurements from time when device was placed on (or taken off) table
    idx = np.argwhere(pos[:, 2] < -0.515).flatten()
    h = h[idx]
    snr = snr[idx]
    pos = pos[idx]

    # Fix #3: Correct the order of FFT components. In Data: (1 to 511, -512 to 0)
    h = np.fft.fftshift(h, axes=2)
    assert h.shape[1:] == (16, 924, 2)
    assert h.dtype == np.float32

    assert snr.shape[1:] == (16,)
    assert snr.dtype == np.float32

    assert pos.shape[1:] == (3,)
    assert pos.dtype == np.float32

    dataset = {"h": h, "snr": snr, "pos": pos}

    joblib.dump(dataset, dataset_output_path)


if __name__ == "__main__":
    cli()

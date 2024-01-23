from pathlib import Path
from typing import Iterable, Literal

import click
import h5py
import joblib
import numpy as np

ANTENNA_OFFSET = np.asarray([3.5, -3.15, 1.8])


@click.command()
@click.argument("input_datasets", nargs=3, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dataset", nargs=1, type=click.Path(dir_okay=False, writable=True, path_type=Path))
@click.option(
    "--task",
    type=click.Choice(["classification", "regression"], case_sensitive=False),
    default="regression",
    show_default=True,
    help="What is the target value",
)
def cli(input_datasets: Iterable[Path], output_dataset: Path, task: Literal["classification", "regression"]):
    assert len(input_datasets) == 3, input_datasets

    for path in input_datasets:
        with h5py.File(path, mode="r", swmr=True) as f:
            if "SNR_Est" in f.keys():
                snr = f["SNR_Est"][:].astype(np.float32).T
                continue

            if "h_Estimated" in f.keys():
                h = f["h_Estimated"][:].astype(np.float32).T
                continue

            if "r_Position" in f.keys():
                r = f["r_Position"][:].astype(np.float32).T
                continue

    # Fix #1: Correct the order of FFT components. In Data: (1 to 511, -512 to 0)
    h = np.fft.fftshift(h, axes=2)
    assert h.shape[1:] == (16, 924, 2)
    assert h.dtype == np.float32

    # Fix #2: Correction of position data. Antenna will now be in the center.
    r = r - ANTENNA_OFFSET

    match task:
        case "regression":
            pass

        case "regression":
            raise NotImplementedError

        case _:
            raise NotImplementedError

    dataset = {"h": h, "snr": snr, "r": r}

    joblib.dump(dataset, output_dataset, compress=9)


if __name__ == "__main__":
    cli()

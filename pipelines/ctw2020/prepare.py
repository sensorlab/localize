from pathlib import Path

import click
import h5py
import joblib
import numpy as np


@click.command()
@click.argument("filenames", nargs=-1, type=click.Path(dir_okay=False, readable=True, path_type=Path))
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=True,
    help="Path to save processed dataset.",
)
def cli(filenames: Path, output_path: Path):
    filenames = sorted(filenames)
    print(f"{filenames=}")

    # Let's merge dataset files
    H_re = []
    H_im = []
    snr = []
    pos = []

    for path in filenames:
        assert path.exists()
        with h5py.File(path, mode="r", swmr=True) as fp:
            H_re.append(fp["H_Re"][:].astype(np.float32))
            H_im.append(fp["H_Im"][:].astype(np.float32))
            snr.append(fp["SNR"][:].astype(np.float32))
            pos.append(fp["Pos"][:].astype(np.float32))

    H_re = np.concatenate(H_re, axis=0)
    H_im = np.concatenate(H_im, axis=0)
    snr = np.concatenate(snr, axis=0)
    pos = np.concatenate(pos, axis=0)

    dataset = {"h_re": H_re, "h_im": H_im, "snr": snr, "pos": pos}

    joblib.dump(dataset, output_path)


if __name__ == "__main__":
    cli()

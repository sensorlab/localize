from pathlib import Path

import click
import joblib
from src import load_data


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
    data = load_data(input_path)
    h, _, pos = data["h"], data["snr"], data["pos"]

    # TODO: Do some feature engineering. Currently is just a passthrough.

    joblib.dump((h, pos), output_path)


if __name__ == "__main__":
    cli()

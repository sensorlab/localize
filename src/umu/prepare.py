from pathlib import Path

import click
import joblib
import pandas as pd


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the raw dataset.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=True,
    help="Path to save processed dataset.",
)
@click.option(
    "--method",
    type=click.Choice(["average", "resample"], case_sensitive=False),
    default="average",
    show_default=True,
    help="Method to harmonize values",
)
def cli(input_path: Path, output_path: Path, method: str):
    df = pd.read_excel(input_path)
    df = df[
        [
            "Column7",
            "Column8",
            "Column14",
            "Column15",
            "Column42",
            "Column43",
            "Column45",
            "Column46",
            "Column47",
            "Column48",
            "Column87",
            "Column88",
            "Column78",
            "Column79",
        ]
    ]
    df.columns = df.iloc[0]
    df = df[1:]

    # convert all columns to numeric
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = pd.to_numeric(df[column], errors="coerce")

    print(df.head())
    df = df.dropna()  # drom the ~2 rows with NaN

    joblib.dump(df, output_path, compress=9)


if __name__ == "__main__":
    cli()

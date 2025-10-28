from pathlib import Path

import click
import joblib


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
# @click.option(
#     "--params",
#     "params_path",
#     type=click.Path(exists=True, dir_okay=False, path_type=Path),
#     required=True,
# )
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=True,
)
@click.option(
    "--task",
    type=click.Choice(["regression"], case_sensitive=False),
    default="regression",
    show_default=True,
    help="What is the target value",
)
def cli(input_path: Path, output_path: Path, task: str):
    df = joblib.load(input_path)

    match task:
        case "regression":
            # Convert discrete values to meters
            df.pos_x = (df.pos_x - 1) * 1.2  # meters
            df.pos_y = (df.pos_y - 1) * 1.2  # meters

            df = df.rename(columns={"pos_x": "target_x", "pos_y": "target_y"})

        case _:
            raise NotImplementedError

    # Find target column(s)
    targets = ["target_x", "target_y"]

    # X are features, y are target(s)
    X, y = df.drop(targets, axis=1), df[targets]

    joblib.dump((X, y), output_path, compress=9)


if __name__ == "__main__":
    cli()

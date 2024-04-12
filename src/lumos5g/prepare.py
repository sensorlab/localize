from pathlib import Path

import click
import joblib
import pandas as pd


def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


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
# @click.option(
#     "--method",
#     type=click.Choice(["average"], case_sensitive=False),
#     default="average",
#     show_default=True,
#     help="Method to harmonize values",
# )
@click.option(
    "--task",
    type=click.Choice(["classification", "regression"], case_sensitive=False),
    default="regression",
    show_default=True,
    help="What is the target value",
)
def cli(input_path: Path, output_path: Path, task: str):
    df = pd.read_csv(input_path)

    joblib.dump(df, output_path)

    # match task:
    #     case "regression":
    #         # Convert discrete values to meters
    #         df.pos_x = (df.pos_x - 1) * 1.2  # meters
    #         df.pos_y = (df.pos_y - 1) * 1.2  # meters

    #     case _:
    #         raise NotImplementedError

    # match method:
    #     # Average samples in one second. If there are none, the value is NaN.
    #     case "average":
    #         data = []

    #         # Average the sample value within a second.
    #         for (x, y, node, ts), subset in df.groupby(by=["pos_x", "pos_y", "node", "timestamp"]):
    #             avg_value = subset.value.sum(min_count=1) / subset.value.count()
    #             item = dict(pos_x=x, pos_y=y, node=node, timestamp=ts, value=avg_value)
    #             data.append(item)

    #         df = pd.DataFrame(data)
    #         df = df.pivot(index=["timestamp", "pos_x", "pos_y"], columns=["node"], values=["value"])
    #         df = df.reset_index(drop=False)

    #         # After pivot, column names become tuples. Fix that.
    #         df.columns = ["".join(map(str, col)).strip().replace("value", "node") for col in df.columns.values]

    #         df = df.rename(columns={"pos_x": "target_x", "pos_y": "target_y"})

    #         # Fill the NaN values with some extremely low RSS value
    #         df = df.fillna(-180)

    #         # TODO: Should this be part of prepare-feature stage?
    #         # Remove datetime column
    #         df = df.drop(columns=["timestamp"])

    #     # TODO: Unroll (calc probability of droped packed and then sample from them)

    #     case _:
    #         raise NotImplementedError

    # # Find target column(s)
    # targets = [col for col in df.columns if col.startswith("target")]

    # # X are features, y are target(s)
    # X, y = df.drop(targets, axis=1), df[targets]

    # joblib.dump((X, y), output_path, compress=9)


if __name__ == "__main__":
    cli()

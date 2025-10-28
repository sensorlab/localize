import json
import re
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd


def load_raw_data(path: Path) -> pd.DataFrame:
    with open(path, mode="r") as fp:
        data = json.load(fp)

    df = []

    for position, measurements in data.items():
        digits = re.findall(r"\d+", position)
        location = tuple(int(i) for i in digits)

        # Winter dataset has measurements only in the middle (3rd) row.
        if len(location) == 1:
            location = (3, *location)

        assert len(location) == 2, f"location identifier is not length 2: {location}"

        pos_x, pos_y = location

        for device_id, samples in measurements.items():
            device_id = int(device_id)
            for sample in samples:
                timestamp, value = sample["timestamp"], sample["rss"]

                item = {"pos_x": pos_x, "pos_y": pos_y, "node": device_id, "timestamp": timestamp, "value": value}
                df.append(item)

    df = pd.DataFrame(df)
    df.timestamp = pd.to_datetime(df.timestamp, unit="s", origin="unix").astype("datetime64[s]")
    df = df.astype({"pos_x": "uint8", "pos_y": "uint8", "value": "int8", "node": "uint8"})

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
@click.option(
    "--method",
    type=click.Choice(["average", "resample"], case_sensitive=False),
    default="average",
    show_default=True,
    help="Method to harmonize values",
)
def cli(input_path: Path, output_path: Path, method: str):
    df = load_raw_data(input_path)

    if method.lower() in ("augmentation", "resample"):
        n_augmented_samples = 1_000

        data = {}

        for (x, y, node), subset in df.groupby(by=["pos_x", "pos_y", "node"]):
            # Make sure there is a position and node in the data structure
            data[(x, y)] = data.get((x, y), {})
            data[(x, y)][node] = data[(x, y)].get(node, [])

            data[(x, y)][node] = subset.value.to_list()

        # Resample datapoints
        # max_samples = 0
        for position in data:
            # Find max number of samples per position.
            # We assume that's the number of transmitted packets.
            max_samples = 0
            for node in data[position]:
                if max_samples < len(data[position][node]):
                    max_samples = len(data[position][node])

            # print(max_samples)
            for node in data[position]:
                current = len(data[position][node])
                for _ in range(max_samples - current):
                    data[position][node].append(np.nan)

                assert len(data[position][node]) == max_samples

        rng = np.random.default_rng(seed=0)
        df = []

        for position in data:
            x, y = position
            for _ in range(n_augmented_samples):
                values = {f"node_{node}": rng.choice(samples) for node, samples in data[position].items()}

                item = dict(pos_x=x, pos_y=y, **values)
                df.append(item)

        df = pd.DataFrame(df)

        # Fill the NaN values with some extremely low RSS value
        df = df.fillna(-180)

        for column in df.columns:
            if column.startswith("node_"):
                df[column] = df[column].astype(np.float32)

    # Average samples in one second. If there are none, the value is NaN.
    elif method.lower() in ("average",):
        data = []

        # Average the sample value within a second.
        for (x, y, node, ts), subset in df.groupby(by=["pos_x", "pos_y", "node", "timestamp"]):
            avg_value = subset.value.sum(min_count=1) / subset.value.count()
            item = {"pos_x": x, "pos_y": y, "node": node, "timestamp": ts, "value": avg_value}
            data.append(item)

        df = pd.DataFrame(data)
        df = df.pivot(index=["timestamp", "pos_x", "pos_y"], columns=["node"], values=["value"])
        df = df.reset_index(drop=False)

        # After pivot, column names become tuples. Fix that.
        df.columns = ["".join(map(str, col)).strip().replace("value", "node") for col in df.columns.values]

        # Fill the NaN values with some extremely low RSS value
        df = df.fillna(-180)

        # TODO: Should this be part of prepare-feature stage?
        # Remove datetime column
        df = df.drop(columns=["timestamp"])

    # TODO: Unroll (calc probability of droped packed and then sample from them)
    else:
        raise NotImplementedError

    joblib.dump(df, output_path, compress=9)


if __name__ == "__main__":
    cli()

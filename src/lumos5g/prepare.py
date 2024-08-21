from pathlib import Path

import math
import click
import joblib
import pandas as pd


def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def lat_lon_to_meters(origin_lat, origin_lon, point_lat, point_lon) -> tuple[float, float]:
    """Works "fine" for distances less than 100km"""

    # Earth's radius in meters
    R = 6_378_137

    # Convert latitude and longitude from degrees to radians
    origin_lat_rad = math.radians(origin_lat)
    # point_lat_rad = math.radians(point_lat)
    delta_lat_rad = math.radians(point_lat - origin_lat)
    delta_lon_rad = math.radians(point_lon - origin_lon)

    # Calculate distance in the latitude direction (North-South)
    delta_meters_lat = delta_lat_rad * R

    # Calculate distance in the longitude direction (East-West)
    delta_meters_lon = delta_lon_rad * R * math.cos(origin_lat_rad)

    return delta_meters_lat, delta_meters_lon


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


    # TODO: Convert to relative coordinates (instead of absolute)
    # longitude = west-east; latitude = south-north;
    # df.longitude = (df.longitude - df.longitude.min()) / (df.longitude.max() - df.longitude.min())
    # df.latitude = (df.latitude - df.latitude.min()) / (df.latitude.max() - df.latitude.min())

    origin_lat, origin_lon = df.latitude.min(), df.longitude.min()
    df[["target_x", "target_y"]] = df.apply(
        lambda row: lat_lon_to_meters(origin_lat, origin_lon, row["latitude"], row["longitude"]),
        axis=1,
        result_type="expand",
    )

    df.drop(columns=["latitude", "longitude"], inplace=True)

    # TODO: Fill missing values for (nr_ssRsrp, nr_ssRsrq, nr_ssSinr)
    # nr_ssRsrp [<-156, >-31] src: http://comtech.vsb.cz/qualmob/rsrp_5g.html
    # nr_ssRsrq [<-43, 20]    src: http://comtech.vsb.cz/qualmob/rsrq_5g.html
    # nr_ssSinr [<-23, >40]   src: http://comtech.vsb.cz/qualmob/sinr_5g.html
    # Extra info: https://webhelp.tempered.io/kb_lte_signal.html

    # The original value is either 2147483647 or NaN (detected reference/beacon signal or not)
    df["lte_rssnr"].fillna(0, inplace=True)

    # NaN values will be below lowest recognized value
    for key in ("lte_rssi", "lte_rsrp", "lte_rsrq", "nr_ssRsrp", "nr_ssRsrq", "nr_ssSinr"):
        df[key].fillna(df[key].min() - 10, inplace=True)

    # Convert string columns into "category" type
    df = df.astype({"nrStatus": "category", "mobility_mode": "category", "trajectory_direction": "category"})

    # Convert compassDirection to sin, cos
    df["compass_sin"] = df["compassDirection"].apply(lambda deg: math.sin(math.radians(deg)))
    df["compass_cos"] = df["compassDirection"].apply(lambda deg: math.cos(math.radians(deg)))
    df.drop(columns=["compassDirection"], inplace=True)

    # Drop useless columns
    df.drop(columns=["run_num", "seq_num"], inplace=True)

    # TODO: Throughput unit? Can we make it relative to specs?

    # TODO: What do I do with "trajectory_direction" and?

    # TODO: Onehot-encode for "tower_id", "trajectory_direction"

    # TODO: Should I group by run_num? What to do with seq_num? Drop?

    # TODO: Goal can be absolute localization, or link prediction (based on direction speed, etc.)

    # TODO: Lumos can potentially have multiple different targets:
    #   1. Exact position estimation (regression, same as for CTW, logatec)
    #   2. Future position estimation


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

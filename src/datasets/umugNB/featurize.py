import math
from pathlib import Path

import click
import joblib
import pandas as pd


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
)
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

    print(df.dtypes)

    gps_cols = ["gpsd_tpv_lat", "gpsd_tpv_lon"]
    for col in gps_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.')
        # Convert to numeric, forcing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Filter Invalid GPS Coordinates
    # Removes the '38018.0' outlier and any UINT_MAX errors
    initial_len = len(df)
    df = df[
        (df.gpsd_tpv_lat >= -90) & (df.gpsd_tpv_lat <= 90) &
        (df.gpsd_tpv_lon >= -180) & (df.gpsd_tpv_lon <= 180)
    ]
    
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} rows with invalid coords.")

    origin_lat, origin_lon = df.gpsd_tpv_lat.min(), df.gpsd_tpv_lon.min()
    print("ORIGIN LAT: ", origin_lat, "ORIGIN LON: ", origin_lon)

    df[["target_x", "target_y"]] = df.apply(
        lambda row: lat_lon_to_meters(origin_lat, origin_lon, row["gpsd_tpv_lat"], row["gpsd_tpv_lon"]),
        axis=1,
        result_type="expand",
    )

    df.drop(columns=["gpsd_tpv_lat", "gpsd_tpv_lon"], inplace=True)

    # Drop columns with too many missing values or low predictive value
    #columns_to_drop = ["phr", "retx_ul", "ta", "puc1"]
    #df.drop(columns=[c for c in columns_to_drop if c in df.columns], inplace=True)

    # Find target column(s)
    targets = ["target_x", "target_y"]

    # X are features, y are target(s)
    X, y = df.drop(targets, axis=1), df[targets]

    joblib.dump((X, y), output_path, compress=9)


if __name__ == "__main__":
    cli()

from pathlib import Path

import click
import joblib
import pandas as pd


def parse_bitrate(value):
    """Convert bitrate strings like '2.31k' or '1.49M' to numeric values."""
    if pd.isna(value) or value == "-" or value == "":
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    
    value = str(value).strip().replace(",", ".")
    if value.endswith("k"):
        return float(value[:-1]) * 1e3
    elif value.endswith("M"):
        return float(value[:-1]) * 1e6
    elif value.endswith("G"):
        return float(value[:-1]) * 1e9
    try:
        return float(value)
    except ValueError:
        return float("nan")


def load_csv_file(path: Path) -> pd.DataFrame:
    """Load a single CSV file, handling different decimal separators."""
    # Try reading with default settings first
    try:
        df = pd.read_csv(path)
        # Check if lat/lon columns have comma decimal separators
        if df["gpsd_tpv_lat"].dtype == object:
            # Re-read with European decimal separator
            df = pd.read_csv(path, decimal=",")
    except Exception:
        df = pd.read_csv(path, decimal=",")

    return df


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to the raw dataset folder (TCP or UDP).",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=True,
    help="Path to save processed dataset.",
)
def cli(input_path: Path, output_path: Path):
    # Find all CSV files in the input folder
    csv_files = sorted(input_path.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {input_path}")

    # Load and concatenate all CSV files
    dfs = []
    for csv_file in csv_files:
        df = load_csv_file(csv_file)
        df["source_file"] = csv_file.stem
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Select relevant columns for localization
    # Features: network metrics from gNB
    # Targets: GPS coordinates
    feature_cols = ["CL","C_ul",'mcs_ul', 'phr', 'pl', 'mcs_dl', 'txok', 'retx_dl', 'brate_ul', 'rxok', "cqi", "snr", "ri", "retx_ul", "ta"]
    
    target_cols = ["gpsd_tpv_lat", "gpsd_tpv_lon"]

    # Keep only columns that exist in the data
    available_features = [col for col in feature_cols if col in df.columns]
    available_targets = [col for col in target_cols if col in df.columns]

    df = df[available_features + available_targets]

    # Convert columns to numeric, handling special values
    for col in df.columns:
        if col in ["brate_dl", "brate_ul"]:
            df[col] = df[col].apply(parse_bitrate)
        elif df[col].dtype == object:
            # Handle "-" and other non-numeric values
            df[col] = df[col].astype(str).str.replace(",", ".").replace("-", "nan")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN in target columns
    df = df.dropna(subset=available_targets)

    # Drop rows with all NaN features
    df = df.dropna(subset=available_features, how="any")

    # Keep only columns with more than one unique value
    df = df.loc[:, df.nunique() > 1]

    print(f"Loaded {len(df)} samples from {len(csv_files)} files")
    print(f"Features: {[c for c in df.columns if c not in target_cols]}")

    joblib.dump(df, output_path, compress=9)


if __name__ == "__main__":
    cli()

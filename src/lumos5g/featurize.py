from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import OneHotEncoder



# def haversine(lon1, lat1, lon2, lat2, earth_radius=6367.):
#     """
#     Calculate the great circle distance between two points
#     on the earth (specified in decimal degrees)
#     All args must be of equal length.
#     """
#     lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
#     c = 2 * np.arcsin(np.sqrt(a))
#     km = earth_radius * c
#     meters = km * 1000.0
#     return meters


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
    df: pd.DataFrame = joblib.load(input_path)


    # Find target column(s)
    targets = [col for col in df.columns if col.startswith("target")]

    # X are features, y are target(s)
    X, y = df.drop(targets, axis=1), df[targets]

    ct = make_column_transformer(
        (OneHotEncoder(drop="if_binary", sparse_output=False), make_column_selector(dtype_include="category")),
        remainder="passthrough",
        n_jobs=-1,
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    X: pd.DataFrame = ct.fit_transform(X)

    # print(X.dtypes)
    ##print(X.isnull().values.any())

    assert not X.isna().values.any()
    assert not np.isinf(X).any().any()


    joblib.dump((X, y), output_path)


if __name__ == "__main__":
    cli()
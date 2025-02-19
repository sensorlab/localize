from json import loads
from pathlib import Path

import click
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ydata_profiling import ProfileReport

from src import load_data


class ExploratoryDataAnalysis:
    def __init__(self, df: pd, output_path: Path):
        self.df = df
        self.output_path = output_path

    def column_summary(self):
        summary_data = []

        with click.progressbar(self.df.columns, label="Generating column summary:   ") as columns:
            for col_name in columns:
                col_dtype = self.df[col_name].dtypes
                num_of_nulls = self.df[col_name].isnull().sum()
                num_of_non_nulls = self.df[col_name].notnull().sum()
                num_of_distinct_values = self.df[col_name].nunique()

                if num_of_distinct_values <= 10:
                    distinct_values_counts = self.df[col_name].value_counts().to_dict()
                else:
                    top_10_values_counts = self.df[col_name].value_counts().head(10).to_dict()
                    distinct_values_counts = {
                        k: v for k, v in sorted(top_10_values_counts.items(), key=lambda item: item[1], reverse=True)
                    }

                summary_data.append(
                    {
                        "Name": col_name.replace("_", r"\_"),
                        "DataType": col_dtype,
                        "NumNulls": num_of_nulls,
                        "NumNonNulls": num_of_non_nulls,
                        "NumDistinctVals": num_of_distinct_values,
                        "DistinctValCnts": distinct_values_counts,
                    }
                )

        summary_df = pd.DataFrame(summary_data)
        return summary_df

    def column_stats(self):
        summary_data = []
        with click.progressbar(self.df.columns, label="Generating column statistics:") as columns:
            for col_name in columns:
                stats = self.df[col_name].describe()
                stats_obj = loads(stats.to_json().replace("%", " perc"))
                col_obj = {"Name": col_name.replace("_", r"\_")}
                summary_data.append({**col_obj, **stats_obj})

            summary_df = pd.DataFrame(summary_data)
        return summary_df

    def column_histogram(self):
        # make sure the figures path exists
        figures_path = str(self.output_path).replace(self.output_path.name, "figures/")
        Path(figures_path).mkdir(parents=True, exist_ok=True)
        figures = []

        # Identify numerical columns
        numerical_columns = self.df.select_dtypes(include=[np.number]).columns

        with click.progressbar(numerical_columns, label="Generating column histograms:") as columns:
            # Perform univariate analysis on numerical columns
            for column in columns:
                # For continuous variables
                num_unique = len(self.df[column].unique())
                if num_unique > 10:  # Assuming if unique values > 10, consider it continuous
                    plt.figure(figsize=(8, 6))
                    ax = sns.histplot(
                        self.df[column], kde=True
                    )  # without bins = min(num_unique, 20) it get's stuck if extreme outliers are present, 20 is just an arbitrary value
                    plt.title(f"Histogram of {column}")
                    plt.xlabel(column)
                    plt.ylabel("Frequency")
                else:  # For discrete or ordinal variables
                    plt.figure(figsize=(8, 6))
                    ax = sns.countplot(x=column, data=self.df)
                    plt.title(f"Count of {column}")
                    plt.xlabel(column)
                    plt.ylabel("Count")

                # Annotate each bar with its count
                for p in ax.patches:
                    ax.annotate(
                        format(p.get_height(), ".0f"),
                        (p.get_x() + p.get_width() / 2.0, p.get_height()),
                        ha="center",
                        va="center",
                        xytext=(0, 5),
                        textcoords="offset points",
                    )

                figpath = Path(f"{figures_path}{self.output_path.stem}-{column}.png")
                plt.savefig(figpath)
                plt.close()
                figures.append({"ColName": column, "Path": Path(f"./figures/{self.output_path.stem}-{column}.png")})
        df_figures = pd.DataFrame(figures)
        return df_figures

    def heatmap(self):
        # provide the path to save to
        figure_path = Path(
            str(self.output_path).replace(self.output_path.name, f"figures/{self.output_path.stem}-heatmap.png")
        )

        # Create a correlation matrix
        corr_matrix = self.df.select_dtypes(include="number").corr()

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")

        # Save the heatmap to the specified output path
        plt.savefig(figure_path)
        plt.close()

    def generate_latex_document(self, dataframes, figures, output_path):
        latex_content = r"""
        \documentclass{article}
        \usepackage{graphicx}
        \usepackage{booktabs}
        \usepackage{float}
        \title{EDA}
        \author{Generated by Python}
        \date{\today}
        \begin{document}
        \maketitle
        """.replace("        ", "")

        # Add tables
        latex_content += r"\section*{Tables}" + "\n"
        for df, label, caption in dataframes:
            latex_content += r"\begin{table}[H]\centering" + "\n"
            latex_content += rf"\caption{{{escape_value(caption)}}}" + "\n"
            latex_content += rf"\label{{{escape_value(label)}}}" + "\n"
            latex_content += r"\resizebox{\textwidth}{!}{" + "\n"
            latex_content += df.to_latex(index=False, escape=True)
            latex_content += r"}" + "\n"
            latex_content += r"\end{table}" + "\n"

        # Add figures
        latex_content += r"\section*{Figures}" + "\n"
        for i, row in figures.iterrows():
            latex_content += r"\begin{figure}[H]\centering" + "\n"
            latex_content += rf"\includegraphics[width=0.8\textwidth]{{{row['Path']}}}" + "\n"
            latex_content += rf"\caption{{{row['ColName'].replace('_', ' ')}}}" + "\n"
            latex_content += rf"\label{{fig:{i+1}}}" + "\n"
            latex_content += r"\end{figure}" + "\n"

        # End LaTeX document
        latex_content += r"\end{document}"

        # Write LaTeX to file
        with open(output_path, "w") as f:
            f.write(latex_content)

        print(f"LaTeX file generated at: {output_path}")

    def gen_tex_report(self):
        # generate the various sections of an EDA
        cols_summary = self.column_summary()

        cols_stats = self.column_stats()

        hist_figs = self.column_histogram()

        self.heatmap()

        # prepare the dataframe with heatnao details to add to thee histogram df
        mapdf = pd.DataFrame([{"ColName": "heatmap", "Path": Path(f"./figures/{self.output_path.stem}-heatmap.png")}])

        # generate the .tex report
        self.generate_latex_document(
            [
                [escape_df(cols_summary), "Col summary", "Col summary"],
                [escape_df(cols_stats), "Col stats", "Col stats"],
            ],
            pd.concat([hist_figs, mapdf], ignore_index=True),
            self.output_path,
        )


def escape_df(df):
    # Escape all values in the DataFrame
    escaped_df = df.map(escape_value)

    # Escape column names
    escaped_df.columns = [escape_value(col) for col in df.columns]

    return escaped_df


def escape_value(value):
    """
    Escapes special LaTeX characters in a value and converts complex objects into LaTeX-safe strings.
    """
    if isinstance(value, str):
        # Escape LaTeX special characters
        return (
            value.replace("_", r"\_")
            .replace("%", r"\%")
            .replace("&", r"\&")
            .replace("#", r"\#")
            .replace("$", r"\$")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("^", r"\^{}")
            .replace("~", r"\textasciitilde{}")
            .replace("\\", r"\textbackslash{}")
        )
    elif isinstance(value, (dict, list)):
        # Convert complex objects to LaTeX-safe strings
        return str(value).replace("{", r"\{").replace("}", r"\}").replace("_", r"\_")
    else:
        # Convert other types to string
        return str(value)


def formatData(data: dict | np.ndarray | pd.DataFrame) -> pd.DataFrame:
    if type(data) == pd.DataFrame:
        return data

    if type(data) == np.ndarray:
        return pd.DataFrame(data.reshape(data.shape[0], -1))

    features = pd.DataFrame()

    for key, value in data.items():
        value = value.reshape(value.shape[0], -1)
        columns = [f"{key}-{_}" for _ in range(value.shape[1])]

        new_df = pd.DataFrame(value, columns=columns)
        features = pd.concat([features, new_df], axis=1)

    return features


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
    if str(output_path).find("interim") >= 0:
        df = formatData(joblib.load(input_path))

    elif str(output_path).find("prepared") >= 0:
        features, targets = map(formatData, load_data(input_path))

        df = pd.concat([features, targets], axis=1)
    else:
        print("!! eda not available for this input data")

    # # Interpolate extreme outliers (in case it hasn't been done in prepare)
    # with click.progressbar(df.select_dtypes(include='number').columns, label="Replacing extreme outliers") as dat:
    #     for i,col in enumerate(dat):
    #         df[col] = replace_extreme_outliers(df[col])

    # generate .tex eda report
    edai = ExploratoryDataAnalysis(df, output_path)
    edai.gen_tex_report()

    # generate html eda report
    profile = ProfileReport(df, title="Profiling Report")
    htmlpath = str(output_path).replace(".tex", ".html")
    profile.to_file(htmlpath)


if __name__ == "__main__":
    cli()

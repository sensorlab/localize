import os
import subprocess
from pathlib import Path
from typing import List, Optional, Union

import click
import joblib
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt


# Utility functions
def adjust_brightness(color, factor: float):
    """Adjust the brightness of an RGB color."""
    return tuple(min(max(c * factor, 0), 1) for c in color)


def filter_outliers(data: List[float]) -> List[float]:
    """Filter out outliers using the IQR method."""
    if len(data) < 3:  # Avoid filtering when insufficient data
        return data
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [x for x in data if lower <= x <= upper]


def truncate_to_significant_figures(num: float, significant_figures=3):
    """
    Truncates num to a specified number of significant figures.
    """
    if num == 0:
        return "0"
    num = float(num)
    if abs(num) < 10**-(significant_figures) or abs(num) >= 10**significant_figures:
        return f"{num:.{significant_figures-1}e}"
    return f"{num:.{significant_figures}g}"


def load_yaml_config(path: Union[str, Path]) -> dict:
    """Load a YAML configuration file."""
    with open(path, mode="r") as file:
        return yaml.safe_load(file)


def compile_tex_to_pdf(tex_path: Path, timeout: int = 5):
    """
    Compiles a LaTeX `.tex` file into a `.pdf` file, timing out after the given duration.
    """
    directory = os.path.dirname(tex_path)
    tex_file = os.path.basename(tex_path)
    try:
        subprocess.run(
            ["pdflatex", tex_file],
            cwd=directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"Compilation timed out after {timeout} seconds. Killing process.")
    except subprocess.CalledProcessError as e:
        print("Error during LaTeX compilation:", e)


class ReportsManager:
    """
    Manages the generation of reports, visualizations, and LaTeX documents based on model evaluation data.

    Usage:
    ```python
    reports = ReportsManager(report_paths, evaluation, save_dir)
    reports.for_optimizer("gridsearch").make_leaderboard()
    ```
    """

    def __init__(self, report_paths: list, evaluation: dict, save_dir: Path):
        """
        Initializes the ReportsManager with paths to reports, evaluation metrics, and the save directory.

        Args:
        - report_paths (list of Path): Paths to the `.pkl` report files.
        - evaluation (dict): Evaluation configuration from `params.yaml`.
        - save_dir (Path): Directory to save generated figures and reports.
        """
        self.score_with = evaluation["score_with"]
        self.metrics = evaluation["metrics"]
        self.save_figures_as = "png"
        self.figure_paths = []
        self.save_dir = save_dir
        self.reports = self._load_reports(report_paths)
        self._selected_optimizer = None
        self._selected_models = None

    def _load_reports(self, report_paths: List[Path]) -> dict:
        """
        Loads and organizes reports from the provided paths.

        Args:
        - report_paths (list of Path): List of paths to the `.pkl` report files.

        Returns:
        - dict: Organized dictionary of reports grouped by optimizer and model.
        """
        reports = {}
        for report_path in report_paths:
            report = joblib.load(report_path, mmap_mode=None)
            optimizer = report["optimizer_data"]["metadata"]["algorithm"]
            model = report["model_data"]["metadata"]["algorithm"]
            reports.setdefault(optimizer, {})[model] = report
        return reports

    def _get_save_path(self, path, *paths) -> Path:
        """
        Constructs a path within the `save_dir` and ensures the directories exist.

        Args:
        - path (str): The base path.
        - *paths (str): Additional subdirectories or filename.

        Returns:
        - Path: The complete path where the file should be saved.
        """
        complete_path = os.path.join(self.save_dir, path, *paths)
        Path(os.path.dirname(complete_path)).mkdir(parents=True, exist_ok=True)
        return Path(complete_path)

    def _save_figure(self, *args, close_plt: bool = True) -> Path:
        """
        Saves a plot to a file and tracks its path for inclusion in the report.

        Args:
        - *args: Path components for the figure's save location.
        - close_plt (bool): Whether to close the plot after saving.

        Returns:
        - Path: The path to the saved figure.
        """
        path = self._get_save_path("figures", *args)
        plt.savefig(path, format=self.save_figures_as)
        self.figure_paths.append(path)
        if close_plt:
            plt.close()
        return Path(path)

    def _get_selected_reports(self) -> dict:
        """
        Retrieves reports based on the selected optimizer and models.

        Returns:
        - dict: Dictionary of selected reports grouped by model name.
        """
        if self._selected_optimizer == "all":
            return {model: report for opt_reports in self.reports.values() for model, report in opt_reports.items()}
        reports = self.reports.get(self._selected_optimizer, {})
        if self._selected_models:
            return {model: reports[model] for model in self._selected_models if model in reports}
        return reports

    def for_optimizer(self, optimizer_name: str):
        """
        Filters reports by optimizer name or selects all reports.

        Args:
        - optimizer_name (str): Name of the optimizer to filter by, or "all" to select all reports.

        Returns:
        - ReportsManager: The current instance for chaining.
        """
        if optimizer_name.lower() == "all":
            self._selected_optimizer = "all"
        elif optimizer_name in self.reports:
            self._selected_optimizer = optimizer_name
        else:
            raise ValueError(f"Optimizer '{optimizer_name}' not found. Available: {self.reports.keys()}")
        return self

    def for_models(self, models: Optional[List[str]] = None):
        """
        Filters reports by specific models.

        Args:
        - models (list of str | None): List of model names to filter by, or None to select all models.

        Returns:
        - ReportsManager: The current instance for chaining.
        """
        self._selected_models = models
        return self

    def generate_latex(self, leaderboard_df: pd.DataFrame, file_name: str, pdf_title: str = "Performance Report"):
        """
        Generates a LaTeX report containing the leaderboard and figures.

        Args:
        - leaderboard_df (pd.DataFrame): The leaderboard DataFrame.
        - file_name (str): The filename for the `.tex` file.
        - pdf_title (str): The title of the LaTeX document.

        Returns:
        - Path: The path to the generated `.tex` file.
        """
        tex_path = self._get_save_path(file_name)
        with open(tex_path, "w") as tex_file:
            tex_file.write(r"\documentclass{article}" + "\n")
            tex_file.write(r"\usepackage{graphicx}" + "\n")
            tex_file.write(r"\usepackage{booktabs}" + "\n")
            tex_file.write(r"\usepackage{float}" + "\n")
            tex_file.write(r"\begin{document}" + "\n")
            tex_file.write(f"\\title{{{pdf_title}}}\n")
            tex_file.write("\\author{Generated by Python}\n")
            tex_file.write(r"\maketitle" + "\n")

            # Add leaderboard as a table
            tex_file.write(r"\section*{Leaderboard}" + "\n")
            tex_file.write(r"\resizebox{\textwidth}{!}{" + "\n")  # Resize table to fit
            tex_file.write(leaderboard_df.to_latex(index=False, escape=True))
            tex_file.write(r"}" + "\n")

            # Add each figure
            tex_file.write(r"\section*{Figures}" + "\n")
            for path in self.figure_paths:
                rel_path = os.path.relpath(path, start=os.path.dirname(tex_path))
                tex_file.write(r"\begin{figure}[H]" + "\n")
                tex_file.write(r"\centering" + "\n")
                tex_file.write(rf"\includegraphics[width=\textwidth]{{{rel_path}}}" + "\n")
                tex_file.write(r"\end{figure}" + "\n")

            tex_file.write(r"\end{document}" + "\n")

        return tex_path

    def make_predictions_scatter_plot_2d(self):
        """
        Generates a 2D scatter plot of true values vs. predicted values.

        Combines data across all splits and overlays scatter points for true values and predicted values.
        """
        selected_reports = self._get_selected_reports()

        for model_name, reports in selected_reports.items():
            model_report = reports["model_data"]["reports"][0]

            combined_y_pred = np.hstack([_["y_pred"] for _ in model_report["model_data_per_split"]])
            combined_y_true = np.hstack([_["y_true"] for _ in model_report["model_data_per_split"]])

            plt.figure(figsize=(8, 6))
            plt.scatter(combined_y_true[0], combined_y_true[1], color="blue", alpha=0.5, label="y_true", s=20)
            plt.scatter(combined_y_pred[0], combined_y_pred[1], color="orange", alpha=0.5, label="y_pred", s=10)
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"{model_name}: True vs Predicted")
            plt.legend()
            plt.grid(True)
            self._save_figure(model_name, f"predictions-scatter.{self.save_figures_as}")

    def make_comparison_plot(self):
        """
        Generates a bar plot comparing model performance on the specified score (`self.score_with`).

        The plot includes the mean score and standard deviation for each model, sorted by performance.
        """
        selected_reports = self._get_selected_reports()
        data = []
        for model_name, report in selected_reports.items():
            scores = report["model_data"]["reports"][0]["scores"]
            if self.score_with in scores:
                data.append((model_name, scores[self.score_with]["mean"], scores[self.score_with]["std"]))

        data.sort(key=lambda x: x[1], reverse=not self.metrics[self.score_with])
        model_names, means, stds = zip(*data, strict=True)

        plt.figure(figsize=(12, 6))
        plt.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7, color="skyblue")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Model")
        plt.ylabel(f"{self.score_with.upper()} Score")
        plt.title(f"Model Comparison {self.score_with.upper()}")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        self._save_figure(f"{self.score_with}-comparison.{self.save_figures_as}")

    def make_hyperparameter_performance_plot(self):
        """
        Generates plots of model performance while varying a single hyperparameter.

        This method is compatible only with the `gridsearch` optimizer, as it relies on the exploration of the full hyperparameter search space.

        """
        assert (
            self._selected_optimizer == "gridsearch"
        ), "'make_hyperparameter_performance_plot' is only compatible with the optimizer 'gridsearch'"

        selected_reports = self._get_selected_reports()
        for model_name, report in selected_reports.items():
            results = report["optimizer_data"]["additional_data"]["results"]

            param_columns = [_ for _ in results.columns if _.startswith("param_")]
            for param in param_columns:
                other_columns = [_ for _ in param_columns if _ != param]

                if other_columns:
                    best_model_values = results.iloc[0][other_columns]

                    # Choses which rows have save values for other params
                    # Handle cases where 'None' or NaN values are present
                    conditions = [
                        results[col].isna() if pd.isna(val) else results[col] == val
                        for col, val in zip(other_columns, best_model_values, strict=True)
                    ]
                    combined_condition = pd.concat(conditions, axis=1).all(axis=1)

                    rows_with_same_value = results[combined_condition]
                    param_values = rows_with_same_value[param]
                    score_mean = rows_with_same_value[f"mean_test_{self.score_with}"]
                    score_std = rows_with_same_value[f"std_test_{self.score_with}"]

                    # Separate None values (there should max be 1)
                    none_mask = param_values.isna()
                    param_values_none = param_values[none_mask]
                    score_mean_none = score_mean[none_mask]
                    score_std_none = score_std[none_mask]

                    param_values = param_values[~none_mask].to_numpy()
                    score_mean = score_mean[~none_mask].to_numpy()
                    score_std = score_std[~none_mask].to_numpy()

                    # Sorting based on param_values
                    sorted_indices = np.argsort(param_values)
                    param_values = param_values[sorted_indices]
                    score_mean = score_mean[sorted_indices]
                    score_std = score_std[sorted_indices]

                    # Plotting
                    param_name_printable = param.replace("param_", "", 1).replace("_", " ").capitalize()

                    if param_values_none.empty:
                        plt.figure(figsize=(10, 6))
                        plt.errorbar(
                            param_values,
                            score_mean,
                            yerr=score_std,
                            fmt="--o",
                            capsize=5,
                            capthick=2,
                            label="Mean ± Std",
                        )
                        plt.xlabel(param_name_printable)
                        plt.ylabel(f"{self.score_with.upper()} Score")
                        plt.title(f"{model_name}: {self.score_with.upper()} score with varying {param_name_printable}")
                        plt.tight_layout()

                    else:
                        fig, (ax1, ax2) = plt.subplots(
                            1, 2, sharey=True, gridspec_kw={"width_ratios": [9, 1]}, figsize=(10, 6)
                        )

                        # Plot numerical values on the first subplot
                        ax1.errorbar(
                            param_values,
                            score_mean,
                            yerr=score_std,
                            fmt="--o",
                            capsize=5,
                            capthick=2,
                            label="Mean ± Std",
                        )
                        ax1.set_xlabel(param_name_printable)
                        ax1.set_ylabel(f"{self.score_with.upper()} Score")
                        ax1.set_title(
                            f"{model_name}: {self.score_with.upper()} score with varying {param_name_printable}"
                        )
                        ax1.spines.right.set_visible(False)
                        ax1.yaxis.tick_left()

                        # Plot None value on the second subplot
                        if not param_values_none.empty:
                            ax2.errorbar(
                                [0],
                                score_mean_none,
                                yerr=score_std_none,
                                fmt="o",
                                capsize=5,
                                capthick=2,
                                label="Mean ± Std",
                            )
                            ax2.set_xticks([0])
                            ax2.set_xticklabels(["None"])
                            ax2.spines.left.set_visible(False)
                            ax2.yaxis.tick_right()

                        # Add broken axis markers
                        d = 0.015  # Marker size
                        kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
                        ax1.plot((1 - d / 9, 1 + d), (-d, +d), **kwargs)  # Bottom-right diagonal
                        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Top-right diagonal

                        kwargs.update(transform=ax2.transAxes)  # Switch to second subplot
                        ax2.plot((-d, +d), (-d, +d), **kwargs)  # Bottom-left diagonal
                        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Top-left diagonal

                        plt.tight_layout()

                    # Save figure
                    self._save_figure(
                        model_name, f"{param.replace('param_', '', 1)}-vs-{self.score_with}.{self.save_figures_as}"
                    )

    def make_performance_history_plot(self):
        """
        Generates a plot showing the evolution of performance metrics (e.g., loss and validation loss) across epochs.

        This method combines and filters data from multiple runs and splits, computing an average value per epoch.
        The plot visualizes how the loss and validation loss evolve over training epochs.

        Compatible only with the `automl` optimizer.

        Behavior:
        - Filters outliers for each epoch using the IQR method.
        - Computes the average loss and validation loss across splits.
        - Optionally saves a logarithmic-scaled version of the plot if the scale difference is large.
        """
        assert (
            self._selected_optimizer == "automl"
        ), "'make_performance_history_plot' is only compatible with the optimizer 'automl'"

        base_color = (0.2, 0.4, 0.8)
        darker_color = adjust_brightness(base_color, 0.8)
        brighter_color = adjust_brightness(base_color, 1.2)

        selected_reports = self._get_selected_reports()
        for model_name, report in selected_reports.items():
            histories = report["optimizer_data"]["additional_data"]["history"]

            for run_idx, split_histories in enumerate(histories):
                plt.figure(figsize=(10, 6))

                loss_by_epoch = []
                val_loss_by_epoch = []

                for history in split_histories:
                    loss = history.history.get("loss", [])
                    val_loss = history.history.get("val_loss", [])

                    if not loss or not val_loss:
                        continue

                    while len(loss_by_epoch) < len(loss):
                        loss_by_epoch.append([])
                        val_loss_by_epoch.append([])

                    # Collect the loss and val_loss for each epoch
                    for epoch_idx, (ls, vls) in enumerate(zip(loss, val_loss, strict=True)):
                        loss_by_epoch[epoch_idx].append(ls)
                        val_loss_by_epoch[epoch_idx].append(vls)

                # Compute filtered averages
                avg_loss = [
                    np.mean(filter_outliers(epoch_losses)) if epoch_losses else None for epoch_losses in loss_by_epoch
                ]
                avg_val_loss = [
                    np.mean(filter_outliers(epoch_losses)) if epoch_losses else None
                    for epoch_losses in val_loss_by_epoch
                ]

                # Determine if a logarithmic scale is needed
                all_values = [v for v in avg_loss + avg_val_loss if v is not None]
                value_range = max(all_values) / min(all_values) if all_values else 1
                use_log_scale = value_range > 1000

                plt.plot(
                    range(len(avg_loss)),
                    avg_loss,
                    label=f"Run {run_idx + 1} Avg Loss",
                    color=darker_color,
                    linestyle="--",
                )
                plt.plot(
                    range(len(avg_val_loss)),
                    avg_val_loss,
                    label=f"Run {run_idx + 1} Avg Val Loss",
                    color=brighter_color,
                    linestyle="-",
                )

                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.title(f"{model_name}: Avg Loss vs Val Loss for Run {run_idx + 1}")
                plt.legend()
                plt.tight_layout()
                self._save_figure(model_name, f"loss-vs-val_loss-{run_idx}.{self.save_figures_as}", close_plt=False)

                if use_log_scale:  # Save the log plot if needed
                    plt.yscale("log")
                    plt.ylabel("Loss (Log Scale)")
                    self._save_figure(
                        model_name, f"log-loss-vs-val_loss-{run_idx}.{self.save_figures_as}", close_plt=False
                    )
                plt.close()

    def make_leaderboard(self):
        """
        Creates a leaderboard of models based on their performance on the selected score (`self.score_with`).

        Sorts the models by the mean value of the selected score, using `self.metrics` to determine if higher is better.

        Returns:
        - pd.DataFrame: The leaderboard as a pandas DataFrame, including rank, model name, and score statistics.
        """
        selected_reports = self._get_selected_reports()
        leaderboard = []
        for model_name, report in selected_reports.items():
            for model_report in report["model_data"]["reports"]:
                scores = model_report["scores"]
                fit_time = model_report["fit_time"]
                score_time = model_report["score_time"]

                # Prepare leaderboard entry
                leaderboard_entry = {"model_name": model_name}

                # Add performance metrics (mean ± std)
                for score_name, score_values in scores.items():
                    leaderboard_entry[f"{score_name} (± std)"] = (
                        f"{score_values['mean']:.4f} (± {score_values['std']:.4f})"
                    )

                    # Add the numeric mean value as a column for sorting
                    if score_name == self.score_with:
                        leaderboard_entry["score_with"] = score_values["mean"]

                # Add fit and score time (mean ± std)
                leaderboard_entry["fit_time (± std)"] = f"{fit_time['mean']:.2f} (± {fit_time['std']:.2f})"
                leaderboard_entry["score_time (± std)"] = f"{score_time['mean']:.2f} (± {score_time['std']:.2f})"

                leaderboard.append(leaderboard_entry)

        leaderboard_df = pd.DataFrame(leaderboard)
        if self.score_with in self.metrics:
            ascending = not self.metrics[self.score_with]
            leaderboard_df = leaderboard_df.sort_values(by="score_with", ascending=ascending)
        leaderboard_df = leaderboard_df.drop(columns=["score_with"]).reset_index(drop=True)

        # Add rank column
        leaderboard_df["rank"] = leaderboard_df.index + 1
        cols = ["rank"] + [col for col in leaderboard_df.columns if col != "rank"]
        leaderboard_df = leaderboard_df[cols]

        return leaderboard_df

    def make_shortened_leaderboard(self, max_scores_per_model: int = 3):
        """
         Creates a shortened leaderboard by limiting the number of entries per model.

        Args:
        - max_scores_per_model (int): Maximum number of entries allowed per model in the leaderboard.

        Returns:
        - pd.DataFrame: The shortened leaderboard DataFrame.
        """
        leaderboard_df = self.make_leaderboard()
        shortened_df = leaderboard_df.groupby("model_name").head(max_scores_per_model)
        return shortened_df


@click.command()
@click.argument("input_reports", nargs=-1, type=click.Path(path_type=Path))
@click.option("--output", "--output-report", "output_path", type=click.Path(path_type=Path))
def cli(input_reports: list[Path], output_path: Path):
    # Path to your YAML configuration file
    config_path = "./params.yaml"

    # Load the configuration
    config = load_yaml_config(config_path)

    print("Starting report generation.")

    # Initialize the ReportsManager
    reports = ReportsManager(
        report_paths=input_reports, evaluation=config["evaluation"], save_dir=os.path.dirname(output_path)
    )

    # Generate all plots and leaderboard
    print("Making predicitons scatter plot.")
    reports.for_optimizer("All").make_predictions_scatter_plot_2d()
    print("Making performance comparison plot.")
    reports.for_optimizer("All").make_comparison_plot()
    print("Making hyperparameter performance plot.")
    reports.for_optimizer("gridsearch").make_hyperparameter_performance_plot()
    print("Making performance history plot.")
    reports.for_optimizer("automl").make_performance_history_plot()
    print("Making model performance leaderboard.")
    leaderboard_df = reports.for_optimizer("All").make_shortened_leaderboard(max_scores_per_model=10)
    # leaderboard_df = reports.for_optimizer("All").make_leaderboard()

    # Generate LaTeX report and compile it into a PDF
    print("Generating LaTex report.")
    tex_report_path = reports.generate_latex(leaderboard_df, file_name=os.path.basename(output_path))
    print("Compliing to pdf.")
    compile_tex_to_pdf(tex_report_path)


if __name__ == "__main__":
    cli()


"""
# Structure of a report:

{
    "model_data": {
        "reports": [
            {
                "model_data_per_split": [
                    {
                        "model_path": "<path_to_model_file>",
                        "y_true": "<true_values_for_output>",
                        "y_pred": "<predicted_values_for_output>",
                        "model_size": "<size_of_model_in_bytes>"
                    },
                    // one for each split
                ],
                "scores": {
                    "rmse": {"mean": "<mean_rmse_value>", "std": "<std_dev_rmse>"},
                    "r_squared": {"mean": "<mean_r_squared_value>", "std": "<std_dev_r_squared>"},
                    "mae": {"mean": "<mean_mae_value>", "std": "<std_dev_mae>"},
                    "mede": {"mean": "<mean_median_euclidean_value>", "std": "<std_dev_median_euclidean>"},
                    "mape": {"mean": "<mean_mape_value>", "std": "<std_dev_mape>"}
                },
                "params": {/* key value param pairs */},
                "fit_time": {"mean": "<mean_fit_time_seconds>", "std": "<std_dev_fit_time>"},
                "score_time": {"mean": "<mean_score_time_seconds>", "std": "<std_dev_score_time>"}
            }
        ],
        "metadata": {
            "algorithm": "<name_of_algorithm_used>"
        }
    },
    "split_data": {
        "splits": [
            "<index_array_for_split_1>",
            // one for each split
        ],
        "metadata": {
            "split_type": "<name_of_split_method_used>"
        }
    },
    "optimizer_data": {
        "metadata": {
            "algorithm": "<name_of_optimization_algorithm_used>"
        },
        "additional_data": {
            //gridsearch
            "results": // pd.Dataframe of the results

            //automl
            "history" : "<list_of_history_callbacks>"
        }
    }
}

"""

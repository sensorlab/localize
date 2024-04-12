# NANCY SaaS Localization: Self-Evolving Model Repository

This project automates the training of machine learning models using Data Version Control (DVC), focusing on a streamlined and efficient pipeline management system. It is designed to work with four distinct datasets: CTW2019, CTW2020, Log-a-Tec, and Lumos5G, each with its dedicated pipeline to cater to the specific requirements and structure of the dataset.

## Features

- **Automated Training Pipelines**: Four specialized pipelines for CTW2019, CTW2020, Log-a-Tec, and Lumos5G datasets.
- **Easy Setup**: Minimal setup required with Conda dependencies.
- **DVC Integration**: Leveraging DVC for efficient data and model versioning, ensuring reproducibility and traceability.

## Project Layout

- `artifacts/<dataset>/data/{raw,interim,splits,prepared}` contains dataset at different stages of data preparation pipeline.
- `configs/<dataset>/dvc.yaml` contains pipeline instructions for DVC tool
- `configs/<dataset>/params.yaml` contains ML model configurations

## Prerequisites

Before you begin, ensure you have the following installed:
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for managing dependencies.

## Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/sensorlab/nancy-saas-localization
```

2. Navigate to the cloned directory and install the required Conda dependencies:

```bash
conda env create -f environment.yml
conda activate nancy
```

## Usage

To build all models for all datasets run the `./run_pipelines.sh` script. Grab a :coffee: as it takes some time to build models from scratch.

> [!WARNING]
> In the current set up, we don't have artifact cache available. It will take some time to build all models from scratch.

If you wish to work on one particular dataset, follow these steps:

1. Activate conda environment using `conda activate nancy` command.
2. Enter the subfolder `configs/<dataset>` with configurations for a dataset. Replace `<dataset>` with either of `ctw2019`, `ctw2020`, `logatec`, or `lumos5g`.
3. (optional) tune/change ML model parameters in `params.yaml` file.
4. Run the following command to start the model training process:

```bash
# On first run also pull dataset dependencies with `--pull`.
# If complains something related to cache, add `--force` flag
dvc repro --pull

# On any subsequent run, it should be enough to run
dvc repro
```

## Contributions

When submitting pull request, we suggest to run `pre-commit` checks. If you don't `pre-commit` installed yet, do the following steps:

1. Run `pip install pre-commit` to install *pre-commit-hooks* tool
2. Run `pre-commit install`, and it will make the tool part of `git commit` step.

Now run `pre-commit run --all-files` to see if your changes comply with code rules.

## License

This project is licensed under the [BSD-3 Clause License](LICENSE) - see the LICENSE file for details.

## Acknowledgments

This project has received funding from the European Union's Horizon Europe Framework Programme under grant agreement No. 101096456 ([NANCY](https://nancy-project.eu/)).
The project is supported by the Smart Networks and Services Joint Undertaking and its members.

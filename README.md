# LOCALIZE

This project automates the training of wireless localization machine learning models using a low-code, configuration-first framework for radio localization in which 1) experiments are declared in human-readable configuration, 2) a workflow orchestrator runs standardized pipelines from data preparation to reporting, and 3) all artifacts, such as datasets, models, metrics, and reports are versioned. The pre-configured, versioned datasets reduce initial setup and boilerplate speeding up model development and evaluation. The design, with clear extension points, let experts add components without reworking the infrastructure.

## Features

- **A low-code, configuration-first** framework that bridges ease of use with scientific rigor by making reproducibility its default operating mode, integrating version control, execution isolation, and transparent artifact tracking
- **Automated Training Pipelines**: Five specialized pipelines for CTW2019, CTW2020, Log-a-Tec, Lumos5G and UMU datasets.
- **Easy Setup**: Minimal setup required with Conda dependencies.
- **DVC Integration**: Leveraging DVC for efficient data and model versioning, ensuring reproducibility and traceability.
- **automated report generation** consistent, comparable evaluation by applying a standardized set of metrics and reporting procedures across all methods and datasets, eliminating glue code and improving the credibility of result.
- 

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
conda env create -f environment.yaml
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

## Citation
If you use this tool please cite our paper: 
```
@article{strnad2025configuration,
  title={A Configuration-First Framework for Reproducible, Low-Code Localization},
  author={Strnad, Tim and Bertalani{\v{c}}, Bla{\v{z}} and Fortuna, Carolina},
  journal={arXiv preprint arXiv:2510.25692},
  year={2025}
}
```

## Acknowledgments

This project has received funding from the European Union's Horizon Europe Framework Programme under grant agreement No. 101096456 ([NANCY](https://nancy-project.eu/)).
The project is supported by the Smart Networks and Services Joint Undertaking and its members.

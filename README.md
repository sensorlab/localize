# NANCY SaaS Localization: Self-Evolving Model Repository

This project automates the training of machine learning models using Data Version Control (DVC), focusing on a streamlined and efficient pipeline management system. It is designed to work with four distinct datasets: CTW2019, CTW2020, Log-a-Tec, and Lumos5G, each with its dedicated pipeline to cater to the specific requirements and structure of the dataset.

## Features

- **Automated Training Pipelines**: Four specialized pipelines for CTW2019, CTW2020, Log-a-Tec, and Lumos5G datasets.
- **Easy Setup**: Minimal setup required with Conda dependencies.
- **DVC Integration**: Leveraging DVC for efficient data and model versioning, ensuring reproducibility and traceability.

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

To use this project for training models with your datasets, follow these steps:

1. **Select a Pipeline**: Navigate to the `pipelines` directory and choose the pipeline corresponding to your dataset:

```bash
cd pipelines/<dataset-name>
```

Replace `<dataset-name>` with one of the following: `ctw2019`, `ctw2020`, `logatec`, or `lumos5g`.

2. **Run DVC Reproduction**: Execute the following command to start the model training process:

```bash
# Dowload external dependencies (e.g., datasets)
dvc pull

# Run all stages for all models
dvc repro
```

This command will automate the training process based on the predefined steps in the selected pipeline.

## License

This project is licensed under the [BSD-3 Clause License](LICENSE) - see the LICENSE file for details.

## Acknowledgments

This project has received funding from the European Union's Horizon Europe Framework Programme under grant agreement No. 101096456 ([NANCY](https://nancy-project.eu/)).
The project is supported by the Smart Networks and Services Joint Undertaking and its members.

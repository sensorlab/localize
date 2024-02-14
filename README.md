# Localization-as-a-Service for NANCY project


## ML Pipelines

Project is designed in a way that allows user to define ML pipeline using through YAML configuration.

In most cases, localization is a multioutput problem, where we estimate x, y, and (sometimes) z coordiantes.

- List of scikit-learn algorithms that support multiouput [[list](https://scikit-learn.org/stable/modules/multiclass.html)]. Algorithms that do not support multioutput need to be adapted, through `MultiOutputRegressor` to train regressor for reach target column.

### Log-a-Tec BLE dataset(s)

### LUMOS 5G dataset

### CTW 2019 dataset

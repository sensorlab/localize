import autokeras as ak
import joblib
from tensorflow import keras
import yaml
from pathlib import Path
import click
from src import PredefinedSplit, safe_indexing
import numpy as np
import time

# keras.mixed_precision.set_global_policy("mixed_bfloat16")

MAX_EPOCHS = 100
MAX_TRIALS = 100


def reset_weights(model):
    for layer in model.layers:
        if isinstance(layer, keras.Model):  # if you're using a model as a layer
            reset_weights(layer)  # apply function recursively
            continue

        # where are the initializers?
        if hasattr(layer, "cell"):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            if "initializer" not in key:  # is this item an initializer?
                continue  # if no, skip it

            # find the corresponding variable, like the kernel or the bias
            if key == "recurrent_initializer":  # special case check
                var = getattr(init_container, "recurrent_kernel")  # noqa: B009
            else:
                var = getattr(init_container, key.replace("_initializer", ""))  # noqa: B009

            var.assign(initializer(var.shape, var.dtype))
            # use the initializer


def load_yaml_config(path: str | Path) -> dict:
    with open(path, mode="r") as file:
        config = yaml.safe_load(file)
    return config


@click.command()
@click.option(
    "--input",
    "--data",
    "input_dataset_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--split-indices",
    "--indices",
    "split_indices_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
)
@click.option(
    "--out-model",
    "--output-model",
    "--output",
    "--model-output",
    "model_output_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    # required=True,
    # help="Input data, parquet",
)
@click.option(
    # "--out-best-params",
    "--output-best-params",
    # "--best-params-output",
    "best_params_output_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    # required=True,
    # help="Input data, parquet",
)
@click.option(
    "--output-report",
    "output_report_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def cli(
    input_dataset_path: Path,
    split_indices_path: Path,
    model_output_path: Path,
    best_params_output_path: Path,
    output_report_path: Path,
):
    # load dataset
    features, targets = joblib.load(input_dataset_path, mmap_mode=None)
    H, snr = features["h"], features["snr"]
    print(H.dtype, snr.dtype)

    if H.shape[1:] == (2, 16, 924):
        print("Change shape to channel-last")
        H = np.transpose(H, (0, 2, 3, 1))  # (None, 2, 16, 924) --> (None, 16, 924, 2)

    # load split indices
    splits = joblib.load(split_indices_path, mmap_mode=None)

    # load indices into custom CrossValidator
    cv_indices: list[tuple[np.ndarray, np.ndarray]] = splits["indices"]
    cv = PredefinedSplit(cv_indices)

    # DEFINE autokeras model
    # Initialize the multi with multiple inputs and outputs.
    snr_input = ak.StructuredDataInput()
    snr_output = ak.DenseBlock()(snr_input)
    snr_output = ak.Flatten()(snr_output)

    h_input = ak.ImageInput()
    h_cnn_output = ak.ConvBlock(kernel_size=3, separable=False, dropout=0)(h_input)
    h_cnn_output = ak.Flatten()(h_cnn_output)

    # h_resnet_output = ak.ResNetBlock(version="v2", pretrained=False)(h_input)
    # h_resnet_output = ak.Flatten()(h_resnet_output)

    output = ak.Merge(merge_type="concatenate")([h_cnn_output, snr_output])
    output = ak.RegressionHead(metrics=["mse"])(output)

    regressor = ak.AutoModel(
        inputs=[h_input, snr_input],
        outputs=output,
        overwrite=True,
        max_trials=MAX_TRIALS,
        tuner="hyperband",
        seed=42,
    )

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=6)

    # Fit the model with prepared data.
    regressor.fit(
        x=[H, snr],
        y=targets,
        batch_size=64,
        validation_split=0.20,
        epochs=MAX_EPOCHS,
        callbacks=[early_stop],
    )

    model_or_pipeline = regressor.export_model()

    print(model_or_pipeline.summary())

    # Prepare template for the final report
    reports = {
        "predictions": {
            "y_true": [],
            "y_pred": [],
        },
        "model_metadata": {
            "algorithm": "AutoML",
            "hyperparameters": {},
            "train_time": [],
            "predict_time": [],
        },
        "split_metadata": splits["metadata"],
    }

    for train_indices, test_indices in cv.split(features, targets):
        H_train = safe_indexing(H, train_indices)
        snr_train = safe_indexing(snr, train_indices)
        y_train = safe_indexing(targets, train_indices)

        H_test = safe_indexing(H, test_indices)
        snr_test = safe_indexing(snr, test_indices)
        y_test = safe_indexing(targets, test_indices)

        print("Resetting weights ...")
        reset_weights(model_or_pipeline)

        time_start = time.perf_counter()
        model_or_pipeline.fit(
            x=[H_train, snr_train],
            y=y_train,
            validation_data=([H_test, snr_test], y_test),
            batch_size=64,
            epochs=MAX_EPOCHS,
            callbacks=[early_stop],
        )
        time_end = time.perf_counter()
        reports["model_metadata"]["train_time"].append(time_end - time_start)

        time_start = time.perf_counter()
        y_pred = model_or_pipeline.predict([H_test, snr_test])
        time_end = time.perf_counter()
        reports["model_metadata"]["predict_time"].append(time_end - time_start)

        # the output is bfloat16
        y_pred = y_pred.astype(np.float32)

        reports["predictions"]["y_true"].append(y_test)
        reports["predictions"]["y_pred"].append(y_pred)

    if output_report_path:
        joblib.dump(reports, output_report_path)

    # if best_params_output_path:
    #    save_params(best_hparams, best_params_output_path)


if __name__ == "__main__":
    cli()

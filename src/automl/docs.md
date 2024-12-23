# AutoML (AutoKeras)
The `automl` configuration section defines models on which to perform NAS using autokeras. AutoKeras allows the user to be completely hands-off by designing the architecture by itself while also allowing the user to specify the high level architecture.

## Model Definitions
The user can define custom models by specifying inputs, blocks, and outputs. Inputs and outputs are required, while blocks are optional.

### 1. Inputs
A list of inputs.
- **name** _(str)_:  specifies the name of the input, used for linking to blocks. (`"input_node{index}-{class}"`)
- **class** _(str)_: name of the class of the input. (`Input`)
- **module** _(str)_: name of the module from which to import the class. (`autokeras`)
- **input** _(list[str], dict)_: specify what part of the dataset the input recieves. When it's a list selects those column from the DataFrame.
    - **starts_with** _(str, list[str])_: selects all columns that start with the string (when a list selects columns if they fit any of the strings).
    - **ends_with** _(str, list[str])_: selects all columns that end with the string (when a list selects columns if they fit any of the strings).
    - **contains** _(str, list[str])_: selects all columns that contain the string (when a list selects columns if they fit any of the strings).
    - **key**: _(str)_ _required_\*: if the input data is a dictionary a key will be required. If the value in the dictionary at that key is a DataFrame, you can still specify other filters.
- **args** _(list)_: all values are passed as kwargs.

### 2. Blocks
A list of blocks, that define the high level architecture of the model. Not required. If not specified autokeras automatically searches for the best architecture.

For predefined blocks:
- **name** _(str)_:  specifies the name of the block, used for linking to blocks. (`"block{index}-{class}"`)
- **class** _(str)_: name of the class of the block. (`GeneralBlock`)
- **module** _(str)_: name of the module from which to import the class. (`autokeras`)
- **input** _(str)_: name of a block or input that gets linked to this block. That block has to be already defined.
- **args** _(list)_: all values are passed as kwargs.

For custom blocks:
- **name** _(str)_:  specifies the name of the block, used for linking to blocks. (`"block{index}-Custom"`)
- **layers** _(list)_: a list of `keras.layers` that get combined into a block.
    - **class** _(str)_: name of the class of the layer. (`Dense`)
    - **module** _(str)_: name of the module from which to import the class. (`keras.layers`)
    - **args** _(list)_: all values are passed as kwargs.
- **input** _(str)_: name of a block or input that gets linked to this block. That block has to be already defined.

### 3. Outputs
A list of outputs.
- ***name** _(str)_:  specifies the name of the output, used for linking to blocks. (`"output_node{index}-{class}"`)
- **class** _(str)_: name of the class of the output. (`RegressionHead`)
- **module** _(str)_: name of the module from which to import the class. (`autokeras`)
- **input** _(str)_: name of a block or input that gets linked to this block. That block has to be already defined.
- **args** _(list)_: all values are passed as kwargs.

### 4. Settings
Settings passed as kwargs to `autokeras.AutoModel()`.
- **project_name** _(str)_: The name of the AutoModel, the project will be saved to `tmp/{project_name}`
- **max_trials** _(int)_: The maximum number of different Keras Models to try. The search may finish before reaching max_trials. (`100`)
- **objective** _(str)_: name of model metric to minimize or maximize (`"val_loss"`)
- **tuner** _(str, dict)_: The tuner to use. If string, it should be one of 'greedy', 'bayesian', 'hyperband' or 'random'.
    - **class**  _(str)_: name of the subclass of AutoTuner.
    - **module** _(str)_: name of the module from which to import the class.
- **overwrite** _(bol)_: Boolean. If `False`, reloads an existing project of the same name if one is found. Otherwise, overwrites the project. (`False`)
- **seed** _(int)_: Random seed. (`42`)
- **max_model_size** _(int)_: Maximum number of scalars in the parameters of a model. Models larger than this are rejected.
- **\*\*kwargs**: Any arguments supported by `keras_tuner.Tuner`.

### 5. Fit settings
Settings passed as kwargs to `.fit()`.
- **batch_size** _(int)_: Number of samples per gradient update. (`32`)
- **epochs** _(int)_: The number of epochs to train each model during the search. Training stops if the validation loss stops improving for 10 epochs (unless you specified an EarlyStopping callback as part of the callbacks argument, in which case the EarlyStopping callback you specified will determine early stopping). (`1000`)
- **callbacks** _(list)_: List of Keras callbacks to apply during training and validation.
- **validation_split** _(float)_: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch.  The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling. This argument is not supported when `x` is a dataset. The best model found would be fit on the entire dataset including the validation data. (`0.2`)
- **verbose** _(int)_: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Note that the progress bar is not particularly useful when logged to a file, so verbose=2 is recommended when not running interactively (eg. in a production environment). Controls the verbosity of both KerasTuner search and [keras.Model.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit). (`1`)
- **\*\*kwargs**: Any arguments supported by [keras.Model.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

## Args procesing
All args are processed in the same way. Args can be passed:
- with direct assgiment: they will be passed as `hp.Fixed`
```YAML
args:
  directly_assigned_arg: 1
```
- as a dict with key values: will be passed as `hp.Choice`
```YAML
  arg_using_choice:
    values: [1, 2, 3, 4]
```
- as a range:
    - for `hp.Int` range you can pass min (`1`), max (_required_), and step (`1`).
    ```YAML
      arg_using_int_range:
        min: -4
        max: 4
        step: 2
    ```
    - for `hp.Float` range you can pass min (`0.0`), max (_required_), sampling (`linear`), step (`0.1` for `linear` otherwise `10`).
        - sampling can be `linear`, `log`, `reverse_log`.
    ```YAML
      arg_using_float_range:
        min: 0.0001
        max: 0.1
        sampling: "log"
        step: 10
    ```
- you can also specify class (_required_) and module (`keras_tuner`), or a list of class and module pairs.
    - if you don't specify `args:` the `module.class` is passed on
    - if `args:` is a dict `module.class(**args)` will be passed on
    - if `args:` is anything else `module.class()` will be passed on
```YAML
  arg_using_class:
    class: "TensorBoard"
    module: "keras.callbacks"
    args:
      log_dir: "../tmp/"

```


## Examples
```YAML
automl:
  minimal_example:
    inputs:
      - name: "data_input"

    outputs:
      - name: "regression_output"
```
This is the minimal configuration required for automl, autokeras will automatically search for the best architecture, but that might be compute and time intensive. By default it only performs 100 trials, which will mostly likely mean that the architecture won't be well optimized.

```YAML
  example_with_custom_architecture:
    inputs:
      - name: "tabular_data_input"
        input:
          key: "dataframe"

      - name: "image_data_input"
        class: "ImageInput"
        input:
          key: "image"

    blocks:
      - name: "dense_block"
        class: "DenseBlock"
        input: "tabular_data_input"

      - name: "image_block"
        class: "ImageBlock"
        input: "image_data_input"
        args:
          block_type: "vanilla"

      - name: "merge_block"
        class: "Merge"
        input: ["dense_block", "image_block"]

      - name: "final_block"
        class: "DenseBlock"
        input: "merge_block"
        args:
          num_layers:
            max: 5
          num_units:
            values: [32, 64, 128, 256, 512]
          use_bn: False

    outputs:
      - name: "first_out"
        input: "final_block"

      - name: "second_out"
        input: "final_block"

    fit_settings:
      epochs: 500

    settings:
      max_trials: 100
      overwrite: True
      seed: 42
      tuner: "bayesian"
      learning_rate:
        min: 0.000001
        max: 0.1
        sampling: "log"

```
This is an example of a model with a custom architecture. In this example autokeras will optimize all the hyperparameters that haven't been specified, while also optimizing the given values for each of the specified hyperparameters (passed as args). Autokeras will not change the values that are set to fixed values eg. `use_bn: False`, `block_type: vanilla`.
```YAML
      - name: "custom_block"
        input: "input_data"
        layers:
          - class: "Dense"
            args:
              units:
                values: [32, 64, 128, 256, 512]
          - class: "BatchNormalization"
          - class: "ReLU"
```
This is an example of a custom block that can be defined using [keras.layers](https://keras.io/api/layers/). For example this is a reimplementation of the `DenseBlock` with `use_bn: True` and `num_layers: 1`.  Note that in this example only the number of units will be auto optimized, no other hyperparameters will be changed. You cannot specify class and layers for the same block (each layer must have a class, but the block itself can't)

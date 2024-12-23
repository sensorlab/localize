import autokeras as ak
import keras
from keras_tuner import HyperParameters

from .custom_blocks import GenericCustomBlock
from .utils import utils


class AutoKerasConfigParser:
    """
    A configuration parser to create an AutoKeras model based on a user-defined
    YAML configuration file. This class handles inputs, blocks, and outputs.
    """

    def __init__(self, config: dict):
        """
        Initializes the parser with the provided configuration. It also
        sets up hyperparameters and dependencies used throughout the model.

        Args:
        - config (dict): Dictionary containing input, block, and output configurations.
        """
        # Hyperparameters container
        self.hp = HyperParameters()
        # Store dependencies between layers
        self.deps = {}
        # Store input configurations for input data configuration
        self.input_configs = []

        # Process inputs, blocks (optional), and outputs from the configuration
        self.inputs = self.process_inputs(config["inputs"])
        self.blocks = self.process_blocks(config.get("blocks", []))
        self.outputs = self.process_outputs(config["outputs"])

        # Additional args are just passed to the HyperParameters object
        utils.parse_args(config.get("additional_hyperparameters", {}), self.hp)

        # Store remaining settings for model building
        self.build_config = config["settings"]

    def process_inputs(self, input_config: list) -> dict:
        """
        Processes the input layer configurations.

        Args:
        - input_config (list): List of input configuration dictionaries.

        Returns:
        - dict: A dictionary mapping input names to inputs (connecting nodes).
        """
        inputs = {}

        for idx, cnf in enumerate(input_config):
            input_module = cnf.get("module", "autokeras")
            input_class = cnf.get("class", "Input")
            input_name = cnf.get("name", f"input_node{idx}-{input_class}")
            input_args = utils.parse_args(cnf.get("args", {}), self.hp)

            self.input_configs.append(cnf.get("input", None))
            self.deps[input_name] = inputs[input_name] = utils.get_class(input_module, input_class)(
                **input_args, name=input_name
            )

        return inputs

    def process_blocks(self, block_config: list) -> dict:
        """
        Processes the block configurations, allowing the creation of custom blocks.

        Args:
        - block_config (list): List of block configuration dictionaries.

        Returns:
        - dict: A dictionary mapping block names to blocks (connecting nodes).
        """
        blocks = {}

        for idx, cnf in enumerate(block_config):
            input_name = cnf["input"]
            if isinstance(input_name, list):
                block_input = [self.deps[name] for name in input_name]
            else:
                block_input = self.deps[input_name]

            # If block class is defined, this is a premade block
            if "class" in cnf:
                block_module = cnf.get("module", "autokeras")
                block_class = cnf.get("class", "GeneralBlock")
                block_args = utils.parse_args(cnf.get("args", {}), self.hp)
                block_name = cnf.get("name", f"block{idx}-{block_class}")
                self.deps[block_name] = blocks[block_name] = utils.get_class(block_module, block_class)(
                    **block_args, name=block_name
                )(block_input)

            # If layers are defined, creates a custom block
            elif "layers" in cnf:
                block_name = cnf.get("name", f"block{idx}-Custom")
                self.deps[block_name] = blocks[block_name] = GenericCustomBlock(cnf["layers"], name=block_name)(
                    block_input
                )

            else:
                raise NotImplementedError("Invalid block definition")

        return blocks

    def process_outputs(self, outputs_config: list) -> dict:
        """
        Processes the output layer configurations.

        Args:
        - outputs_config (list): List of output configuration dictionaries.

        Returns:
        - dict: A dictionary mapping output names to outputs (connecting nodes).
        """
        outputs = {}

        self.output_names = []

        for idx, cnf in enumerate(outputs_config):
            output_module = cnf.get("module", "autokeras")
            output_class = cnf.get("class", "RegressionHead")
            self.output_names.append(
                output_name := cnf.get("name", f"output_node{idx}-{output_class}")
            )  # saved to be used when evaling models
            output_args = utils.parse_args(cnf.get("args", {}), self.hp)

            # Check if output has an input, if yes, link them
            output_input = self.deps.get(cnf.get("input"))
            if output_input:
                outputs[output_name] = utils.get_class(output_module, output_class)(**output_args, name=output_name)(
                    output_input
                )
            else:
                outputs[output_name] = utils.get_class(output_module, output_class)(**output_args, name=output_name)

        return outputs

    def build_model(self) -> ak.AutoModel:
        """
        Builds an AutoModel using the inputs, blocks, and outputs defined in the configuration.

        Returns:
        - ak.AutoModel: The AutoKeras model.
        """

        # Set random seeds for reproducibility
        SEED = self.build_config.get("seed", 42)
        keras.utils.set_random_seed(SEED)

        return ak.AutoModel(
            inputs=list(self.inputs.values()),
            outputs=list(self.outputs.values()),
            hyperparameters=self.hp,
            **self.build_config,
        )

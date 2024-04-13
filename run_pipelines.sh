#!/bin/bash

# TODO: Check if conda environment exists

# Run a command with conda environment without entering.
conda="conda run -n nancy --no-capture-output"

# The final DVC command
dvc="${conda} dvc repro --pull"


# Quirk #1: DVC cannot retrieve Lumos5G dataset, because they have expired HTTPS certificate. Let's do it manually.
lumos_path="./artifacts/lumos5g/data/raw/Lumos5G-v1.0.zip"
lumos_url="https://lumos5g.umn.edu/downloads/Lumos5G-v1.0.zip"
if [ ! -e "${lumos_path}" ]; then
    echo "$lumos_path does not exist. Downloading..."
    eval "${conda} curl --insecure -o ${lumos_path} ${lumos_url}"
fi


for folder in ./configs/*/; do
    if [[ "$folder" != "./configs/common/" ]]; then
        # Change directory to the subfolder
        cd "$folder" || exit

        # Execute the command
        eval "$dvc"

        # Return to the parent directory
        cd - > /dev/null

    fi
done

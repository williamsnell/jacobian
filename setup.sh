#!/bin/bash
echo uv --version
if [ "$?" -ne "0" ]
then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Add UV to the path
source $HOME/.local/bin/env
# Run the script
uv run main.py

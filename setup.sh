#!/bin/bash
# Try add UV to the path, and then run it
source $HOME/.local/bin/env

# Also write this to the ~/.bashrc
echo source $HOME/.local/bin/env >> ~/.bashrc

uv --version
# Install uv if it isn't already
if [ "$?" -ne "0" ]
then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Run the script
uv run main.py

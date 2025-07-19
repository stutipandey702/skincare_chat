#!/bin/bash

# Exit if any command fails
set -e

echo "Creating virtual environment..."
python3 -m venv skin_env

echo "Virtual environment created."

# Activate the virtual environment
source skin_env/bin/activate

echo "Installing Python packages..."
python3 -m pip install --upgrade pip
python3 -m pip install torch transformers datasets peft accelerate flask

echo "Saving dependencies to requirements.txt..."
python3 -m pip freeze > requirements.txt


# echo "Setup complete. To activate the environment later, run:"
# echo "source skin_env/bin/activate"


# to make it executable later - use chmod +x setup.sh
# then run as ./setup.sh
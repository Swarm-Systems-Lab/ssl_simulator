#!/bin/bash

# ---------
# File: test_dep.sh
#
# Description:
#     This script automates the process of setting up a clean Python virtual environment 
#     for the project, installing all necessary dependencies, and verifying that the 
#     environment is correctly configured.
#
#     This script is helpful for quickly setting up the project in an isolated environment 
#     to ensure that everything works properly without needing to install packages globally.
# ---------

SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Create virtual environment
python3 -m venv $SCRIPT_DIR/.venv

# Activate virtual environment
source $SCRIPT_DIR/.venv/bin/activate

if [ $? -ne 0 ]; then
    echo "Failed to activate the virtual environment. Exiting..."
    exit 1
fi

# Install dependencies
pip install $SCRIPT_DIR/../.

# Run the project to test
pip install pytest flake8
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
pytest .

# Deactivate virtual environment when done
deactivate

# Remove the virtual environment
rm -r $SCRIPT_DIR/.venv
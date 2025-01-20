#!/bin/bash

# This script automates the process of updating and publishing the zklora package to PyPI

# Clean up previous build artifacts
rm -rf dist build zklora.egg-info

# Set up virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Update pip to the latest version
pip install --upgrade pip

# Install build and upload tools
pip install build twine

# Build the package
python -m build

# Prepare for PyPI upload
echo "Enter your PyPI API token:"
read -s PYPI_API_TOKEN

# Set up authentication for PyPI
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="$PYPI_API_TOKEN"

# Upload to PyPI
twine upload dist/*

# Deactivate the virtual environment
deactivate

echo "Update process completed. Virtual environment has been deactivated." 

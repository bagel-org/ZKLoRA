#!/bin/bash

# Exit on error
set -e

# Build Rust extension first
echo "Building Rust extension..."
cd src/zklora/libs/zklora_halo2
cargo test --all-features
cd -

# Install test dependencies if needed
pip install -e ".[test]"

# Run Python tests with coverage
echo "Running Python tests with coverage..."
pytest --cov=zklora --cov-report=html --cov-report=term-missing

# Print coverage report location
echo "Coverage report generated in htmlcov/index.html" 
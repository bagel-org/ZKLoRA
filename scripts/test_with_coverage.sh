#!/bin/bash

# Exit on error and print commands
set -ex

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Install test dependencies if not already installed
if ! python -c "import pytest_cov" 2>/dev/null; then
    pip install -e ".[test]"
fi

# Build Rust library first
echo "Building Rust library..."
cd src/zklora/libs/zklora_halo2
maturin develop --release
cd -

# Run Python tests with coverage
echo "Running Python tests with coverage..."
python -m pytest tests/ -v --cov=src/zklora --cov-report=xml --cov-report=term-missing

# Run Rust tests
echo "Running Rust tests..."
cd src/zklora/libs/zklora_halo2
cargo test --release -- --nocapture
cd -

# Print coverage report location
echo "Coverage reports:"
echo "- XML: coverage.xml"
echo "- HTML: htmlcov/index.html (if generated)" 
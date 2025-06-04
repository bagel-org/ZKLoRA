from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
import subprocess
import sys

class BuildPyCommand(build_py):
    def run(self):
        # Build Rust library
        subprocess.check_call([
            "maturin", "build",
            "--release",
            "--bindings", "pyo3",
            "--manifest-path", "src/zklora/libs/zklora_halo2/Cargo.toml",
            "--strip"
        ])
        build_py.run(self)

setup(
    name="zklora",
    version="0.1.0",
    packages=find_packages(where="src", include=["zklora", "zklora.*"]),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "onnx>=1.12.0",
        "onnxruntime>=1.12.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=3.0.0"
        ]
    },
    python_requires=">=3.8",
    cmdclass={
        'build_py': BuildPyCommand,
    },
) 
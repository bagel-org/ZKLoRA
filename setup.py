from setuptools import setup, find_packages

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() 
        for line in fh 
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="zklora",
    version="0.1.2",
    author="Bagel Team",
    author_email="team@bagel.org",
    description="Zero-knowledge proofs for LoRA model inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bagel-org/zkLoRA",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "isort>=5.11.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.20.0",
            "pytest-mock>=3.10.0",
            "pytest-timeout>=2.1.0",
            "coverage>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "zklora-server=src.scripts.optimized_lora_example:main",
        ],
    },
) 
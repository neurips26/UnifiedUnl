"""
setup.py — Install the benchmark as a Python package.
Allows: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="multimodal-unlearning-eval",
    version="1.0.0",
    description=(
        "Metric Unreliability in Multimodal Machine Unlearning: "
        "A Systematic Analysis and Principled Unified Score"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests*", "outputs*", "checkpoints*", "data_cache*"]),
    python_requires=">=3.9",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "run-benchmark=benchmark.run_benchmark:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

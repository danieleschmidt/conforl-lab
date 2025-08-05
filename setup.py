#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="conforl",
    version="0.1.0",
    author="Daniel Schmidt",
    author_email="daniel@terragonlabs.ai",
    description="Adaptive Conformal Risk Control for Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/conforl-lab",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "isort>=5.0",
            "mypy>=1.0",
            "pre-commit>=2.20",
        ],
        "vis": [
            "matplotlib>=3.5",
            "seaborn>=0.11",
            "plotly>=5.0",
            "streamlit>=1.20",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
            "notebook>=6.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "conforl=conforl.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
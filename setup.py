"""
BabyMamba-HAR - Ultra-Lightweight HAR with State Space Models.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    longDescription = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="babyMambaHar",
    version="1.0.0",
    author="Mridankan Mandal",
    description="BabyMamba-HAR: Ultra-Lightweight State Space Models for Human Activity Recognition",
    long_description=longDescription,
    long_description_content_type="text/markdown",
    url="https://github.com/WhiteMetagross/BabyMambaHAR",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
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
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
        ],
    },
)

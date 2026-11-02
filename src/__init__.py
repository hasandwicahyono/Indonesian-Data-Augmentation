"""
Indonesian Data Augmentation (IDA) Python Package.
A comprehensive framework for generative text augmentation, adaptive imbalance calibration,
and deep evaluation across low-resource Indonesian NLP benchmarks.
"""

from src import utils
from src import data
from src import augmentation
from src import modeling
from src import benchmarks

__version__ = "2.0.0"

__all__ = [
    "utils",
    "data",
    "augmentation",
    "modeling",
    "benchmarks",
]


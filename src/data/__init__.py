"""
Data loading and preprocessing package for multilingual and Indonesian NLP datasets.
"""

from src.data.dataset_manager import CustomDataset, IndonesianDatasetsManager, MultilingualDatasetsManager
from src.data.preprocessor import (
    BasePreprocessor,
    IndonesianPreprocessor,
    ArabicPreprocessor,
    PassThroughPreprocessor,
    get_preprocessor,
)
from src.data.classification import ClassificationDataset

__all__ = [
    "CustomDataset",
    "IndonesianDatasetsManager",
    "MultilingualDatasetsManager",
    "BasePreprocessor",
    "IndonesianPreprocessor",
    "ArabicPreprocessor",
    "PassThroughPreprocessor",
    "get_preprocessor",
    "ClassificationDataset",
]


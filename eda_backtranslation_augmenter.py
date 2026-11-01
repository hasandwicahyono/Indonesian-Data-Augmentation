"""
eda_backtranslation_augmenter: backward-compatible entry point delegating to src.augmentation.baselines.
"""

from src.augmentation.baselines import (
    IndonesianEDA,
    IndonesianBackTranslator,
    INDONESIAN_SYNONYMS,
)

__all__ = [
    "IndonesianEDA",
    "IndonesianBackTranslator",
    "INDONESIAN_SYNONYMS",
]

if __name__ == "__main__":
    eda = IndonesianEDA()
    text = "makanan ini sangat lezat dan enak sekali"
    augmented = eda.augment(text, n=2)
    print(f"Original: {text}")
    print(f"EDA Augmented: {augmented}")

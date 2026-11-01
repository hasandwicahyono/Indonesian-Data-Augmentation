"""
IndonesianPreprocessor: backward-compatible entry point delegating to src.data.preprocessor.
"""

from src.data.preprocessor import (
    IndonesianPreprocessor,
    INDONESIAN_SLANG_DICT,
    EMOJI_SENTIMENT_DICT,
    EMOTICON_DICT,
)

__all__ = [
    "IndonesianPreprocessor",
    "INDONESIAN_SLANG_DICT",
    "EMOJI_SENTIMENT_DICT",
    "EMOTICON_DICT",
]

if __name__ == "__main__":
    prep = IndonesianPreprocessor()
    test_text = "Pelayanan di hotel ini bener2 jelek bgt!! :( https://hotel.com @admin #kecewa"
    print(f"Original: {test_text}")
    print(f"Preprocessed: {prep.preprocess(test_text)}")

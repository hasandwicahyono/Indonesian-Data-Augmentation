"""
Indonesian Text Preprocessor for Data Augmentation and Classification.
Adapted for Bahasa Indonesia to replace ArabertPreprocessor from the original research.
Handles social media text, reviews, casing, elongation, URLs, mentions, emojis, and slang normalization.
Loads default vocabulary and mapping dictionaries dynamically from configs/preprocessor.yaml.
"""

import re
import string
from typing import Optional, Dict

from src.utils.config import load_preprocessor_config


def _get_default_dicts():
    try:
        cfg = load_preprocessor_config()
        slang = cfg.get("slang_dict", {})
        emoji = cfg.get("emoji_sentiment_dict", {})
        emoticon = cfg.get("emoticon_dict", {})
        return slang, emoji, emoticon
    except Exception:
        return {}, {}, {}


_DEFAULT_SLANG, _DEFAULT_EMOJI, _DEFAULT_EMOTICON = _get_default_dicts()

INDONESIAN_SLANG_DICT: Dict[str, str] = _DEFAULT_SLANG
EMOJI_SENTIMENT_DICT: Dict[str, str] = _DEFAULT_EMOJI
EMOTICON_DICT: Dict[str, str] = _DEFAULT_EMOTICON


class IndonesianPreprocessor:
    def __init__(
        self,
        lower: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        normalize_elongation: bool = True,
        normalize_slang: bool = True,
        remove_extra_whitespace: bool = True,
        map_emojis: bool = True,
        map_emoticons: bool = True,
        slang_dict: Optional[Dict[str, str]] = None,
        emoji_dict: Optional[Dict[str, str]] = None,
        emoticon_dict: Optional[Dict[str, str]] = None,
    ):
        self.lower = lower
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.normalize_elongation = normalize_elongation
        self.normalize_slang = normalize_slang
        self.remove_extra_whitespace = remove_extra_whitespace
        self.map_emojis = map_emojis
        self.map_emoticons = map_emoticons

        # Initialize dictionaries from config if not explicitly overridden
        if slang_dict is None or emoji_dict is None or emoticon_dict is None:
            cfg_slang, cfg_emoji, cfg_emoticon = _get_default_dicts()
        else:
            cfg_slang, cfg_emoji, cfg_emoticon = {}, {}, {}

        self.slang_dict = slang_dict if slang_dict is not None else cfg_slang
        self.emoji_dict = emoji_dict if emoji_dict is not None else cfg_emoji
        self.emoticon_dict = emoticon_dict if emoticon_dict is not None else cfg_emoticon

    def preprocess(self, text: str) -> str:
        """Applies comprehensive Indonesian preprocessing to the input string."""
        if not isinstance(text, str):
            text = str(text)

        # 1. Map emojis to sentiment/emotion tokens
        if self.map_emojis and self.emoji_dict:
            for emoji, token in self.emoji_dict.items():
                if emoji in text:
                    text = text.replace(emoji, token)

        # 2. Map ASCII emoticons
        if self.map_emoticons and self.emoticon_dict:
            for emo, token in self.emoticon_dict.items():
                if emo in text:
                    text = text.replace(emo, token)

        # 3. URL removal
        if self.remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # 4. User mentions removal (@username)
        if self.remove_mentions:
            text = re.sub(r"@\w+", "", text)

        # 5. Hashtag normalization
        if self.remove_hashtags:
            text = re.sub(r"#\w+", "", text)
        else:
            text = re.sub(r"#(\w+)", r"\1", text)

        # 6. Lowercasing
        if self.lower:
            text = text.lower()

        # 7. Normalize character elongation (e.g., "bagusssss" -> "bagus")
        if self.normalize_elongation:
            text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        # 8. Normalize slang
        if self.normalize_slang and self.slang_dict:
            tokens = text.split()
            tokens = [self.slang_dict.get(t, t) for t in tokens]
            text = " ".join(tokens)

        # 9. Clean up extra whitespaces
        if self.remove_extra_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        return text

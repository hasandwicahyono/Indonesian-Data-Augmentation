"""
Indonesian Text Preprocessor for Data Augmentation and Classification
Adapted for Bahasa Indonesia to replace ArabertPreprocessor from the original research.
Handles social media text, reviews, casing, elongation, URLs, mentions, emojis, and slang normalization.
"""

import re
import string
from typing import Optional, Dict


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
        slang_dict: Optional[Dict[str, str]] = None,
    ):
        self.lower = lower
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.normalize_elongation = normalize_elongation
        self.normalize_slang = normalize_slang
        self.remove_extra_whitespace = remove_extra_whitespace

        # Default common Indonesian slang dictionary for social media text
        self.slang_dict = slang_dict or {
            "yg": "yang",
            "dgn": "dengan",
            "tp": "tapi",
            "bgt": "banget",
            "gk": "tidak",
            "gak": "tidak",
            "ga": "tidak",
            "ngga": "tidak",
            "nggak": "tidak",
            "tdk": "tidak",
            "sdh": "sudah",
            "udh": "sudah",
            "udah": "sudah",
            "blm": "belum",
            "krn": "karena",
            "bisaa": "bisa",
            "jd": "jadi",
            "jgn": "jangan",
            "bkn": "bukan",
            "klo": "kalau",
            "kalo": "kalau",
            "kl": "kalau",
            "sm": "sama",
            "org": "orang",
            "sy": "saya",
            "gw": "saya",
            "gue": "saya",
            "lu": "kamu",
            "lo": "kamu",
            "km": "kamu",
            "bgtu": "begitu",
            "trs": "terus",
            "dr": "dari",
            "utk": "untuk",
            "aja": "saja",
            "bener": "benar",
            "beneran": "benaran",
            "mantap": "bagus",
            "mantul": "mantap betul",
            "bapuk": "jelek",
            "bagusss": "bagus",
            "keren": "bagus",
            "oke": "baik",
        }

        # Regex patterns
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.mention_pattern = re.compile(r"@[\w_]+")
        self.email_pattern = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
        self.hashtag_pattern = re.compile(r"#(\w+)")
        self.elongation_pattern = re.compile(r"(.)\1{2,}")  # 3+ repeated chars -> 2 chars
        self.whitespace_pattern = re.compile(r"\s+")

    def preprocess(self, text: str) -> str:
        """
        Cleans and normalizes Indonesian text.
        """
        if not isinstance(text, str):
            text = str(text)

        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub(" ", text)

        # Remove email addresses
        text = self.email_pattern.sub(" ", text)

        # Remove or normalize user mentions
        if self.remove_mentions:
            text = self.mention_pattern.sub(" ", text)

        # Handle hashtags (extract word without # symbol)
        if self.remove_hashtags:
            text = self.hashtag_pattern.sub(r"\1", text)

        # Normalize elongation (e.g., 'baguuuus' -> 'bagus')
        if self.normalize_elongation:
            text = self.elongation_pattern.sub(r"\1\1", text)

        # Lowercase
        if self.lower:
            text = text.lower()

        # Slang dictionary replacement
        if self.normalize_slang and self.slang_dict:
            words = text.split()
            words = [self.slang_dict.get(w, w) for w in words]
            text = " ".join(words)

        # Remove extra whitespaces
        if self.remove_extra_whitespace:
            text = self.whitespace_pattern.sub(" ", text).strip()

        return text

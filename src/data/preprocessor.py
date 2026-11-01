"""
Indonesian Text Preprocessor for Data Augmentation and Classification.
Adapted for Bahasa Indonesia to replace ArabertPreprocessor from the original research.
Handles social media text, reviews, casing, elongation, URLs, mentions, emojis, and slang normalization.
"""

import re
import string
from typing import Optional, Dict

# Emoji semantic mapping for Indonesian emotion and sentiment analysis
EMOJI_SENTIMENT_DICT: Dict[str, str] = {
    # Anger / Marah
    "😡": " _emosi_marah_ ", "😠": " _emosi_marah_ ", "🤬": " _emosi_marah_ ", "👿": " _emosi_marah_ ",
    "💢": " _emosi_marah_ ", "😤": " _emosi_marah_ ", "🖕": " _emosi_marah_ ", "😾": " _emosi_marah_ ",
    # Sadness / Sedih
    "😭": " _emosi_sedih_ ", "😢": " _emosi_sedih_ ", "😞": " _emosi_sedih_ ", "😔": " _emosi_sedih_ ",
    "💔": " _emosi_sedih_ ", "😿": " _emosi_sedih_ ", "🥺": " _emosi_sedih_ ", "😥": " _emosi_sedih_ ",
    "😓": " _emosi_sedih_ ", "😪": " _emosi_sedih_ ", "🥀": " _emosi_sedih_ ",
    # Fear / Takut / Cemas
    "😱": " _emosi_takut_ ", "😨": " _emosi_takut_ ", "😰": " _emosi_takut_ ", "😧": " _emosi_takut_ ",
    "🥶": " _emosi_takut_ ", "😖": " _emosi_takut_ ", "😬": " _emosi_takut_ ", "😳": " _emosi_takut_ ",
    # Love / Cinta / Sayang
    "❤️": " _emosi_cinta_ ", "💖": " _emosi_cinta_ ", "💕": " _emosi_cinta_ ", "💓": " _emosi_cinta_ ",
    "🥰": " _emosi_cinta_ ", "😍": " _emosi_cinta_ ", "😚": " _emosi_cinta_ ", "😻": " _emosi_cinta_ ",
    "😘": " _emosi_cinta_ ", "💗": " _emosi_cinta_ ", "💘": " _emosi_cinta_ ", "💙": " _emosi_cinta_ ",
    "💜": " _emosi_cinta_ ", "💛": " _emosi_cinta_ ", "💚": " _emosi_cinta_ ", "🤍": " _emosi_cinta_ ",
    # Happy / Bahagia / Senang
    "🤣": " _emosi_bahagia_ ", "😂": " _emosi_bahagia_ ", "😄": " _emosi_bahagia_ ", "😆": " _emosi_bahagia_ ",
    "😊": " _emosi_bahagia_ ", "🥳": " _emosi_bahagia_ ", "😁": " _emosi_bahagia_ ", "😹": " _emosi_bahagia_ ",
    "😃": " _emosi_bahagia_ ", "🙂": " _emosi_bahagia_ ", "😀": " _emosi_bahagia_ ", "✨": " _emosi_bahagia_ ",
    "🎉": " _emosi_bahagia_ ", "👏": " _emosi_bahagia_ ", "👍": " _emosi_bahagia_ ",
}

# ASCII emoticons mapping
EMOTICON_DICT: Dict[str, str] = {
    ":)": " _emosi_bahagia_ ", ":-)": " _emosi_bahagia_ ", ":D": " _emosi_bahagia_ ",
    ":-D": " _emosi_bahagia_ ", "xD": " _emosi_bahagia_ ", "XD": " _emosi_bahagia_ ",
    "xd": " _emosi_bahagia_ ", "^^": " _emosi_bahagia_ ", "^_^": " _emosi_bahagia_ ",
    ":(": " _emosi_sedih_ ", ":-(": " _emosi_sedih_ ", ":'(": " _emosi_sedih_ ",
    ":'-(": " _emosi_sedih_ ", "T_T": " _emosi_sedih_ ", "T.T": " _emosi_sedih_ ",
    "T-T": " _emosi_sedih_ ", ";-;": " _emosi_sedih_ ", ":<": " _emosi_sedih_ ",
    "</3": " _emosi_sedih_ ",
    ":O": " _emosi_takut_ ", ":-O": " _emosi_takut_ ", ":o": " _emosi_takut_ ",
    ":-o": " _emosi_takut_ ", ">_<": " _emosi_takut_ ",
    ">:[": " _emosi_marah_ ", ">:(": " _emosi_marah_ ", ":@": " _emosi_marah_ ",
    "<3": " _emosi_cinta_ ", ":*": " _emosi_cinta_ ",
}

# Default common Indonesian slang dictionary for social media text
INDONESIAN_SLANG_DICT: Dict[str, str] = {
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
    "skrg": "sekarang",
    "lg": "lagi",
    "dr": "dari",
    "pd": "pada",
    "utk": "untuk",
    "dlm": "dalam",
    "aja": "saja",
    "aj": "saja",
    "bener": "benar",
    "bener2": "benar-benar",
    "banget2": "sangat",
    "bgtt": "banget",
    "mantul": "mantap betul",
    "gokil": "gila keren",
    "parah": "sangat buruk",
}


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

        self.emoji_dict = emoji_dict or EMOJI_SENTIMENT_DICT
        self.emoticon_dict = emoticon_dict or EMOTICON_DICT
        self.slang_dict = slang_dict or INDONESIAN_SLANG_DICT

    def preprocess(self, text: str) -> str:
        """Applies comprehensive Indonesian preprocessing to the input string."""
        if not isinstance(text, str):
            text = str(text)

        # 1. Map emojis to sentiment/emotion tokens before lowercasing or punctuation stripping
        if self.map_emojis:
            for emoji, token in self.emoji_dict.items():
                if emoji in text:
                    text = text.replace(emoji, token)

        # 2. Map ASCII emoticons
        if self.map_emoticons:
            for emo, token in self.emoticon_dict.items():
                if emo in text:
                    text = text.replace(emo, token)

        # 3. URL removal
        if self.remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # 4. User mentions removal (@username)
        if self.remove_mentions:
            text = re.sub(r"@\w+", "", text)

        # 5. Hashtag removal or keep token
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
        if self.normalize_slang:
            tokens = text.split()
            tokens = [self.slang_dict.get(t, t) for t in tokens]
            text = " ".join(tokens)

        # 9. Clean up extra whitespaces
        if self.remove_extra_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        return text

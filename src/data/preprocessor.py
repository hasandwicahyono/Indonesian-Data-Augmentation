"""
Text Preprocessor Suite for Data Augmentation and Classification.
Provides modular, language-aware preprocessors:
- IndonesianPreprocessor: Social media, reviews, casing, elongation, URLs, mentions, emojis, and slang normalization.
- ArabicPreprocessor: Diacritics stripping, tatweel removal, alef/hamza normalization, elongation reduction, URLs, mentions.
- PassThroughPreprocessor: Language-agnostic baseline preprocessor.
- get_preprocessor: Dynamic factory matching language string to the appropriate preprocessor instance.
"""

import abc
import os
import re
import string
from typing import Optional, Dict

from src.utils.config import load_preprocessor_config
from src.utils.path_utils import resolve_path


class BasePreprocessor(abc.ABC):
    """Abstract base class for all language preprocessors."""

    @abc.abstractmethod
    def preprocess(self, text: str) -> str:
        """Applies language-specific preprocessing to the input string."""
        pass


def load_mle_emoji_lexicon(
    table1_path: Optional[str] = None,
    table2_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Constructs the Multidimensional Lexicon of Emojis (MLE) mapping
    grounded in Godard & Holtzman (2022) [Frontiers in Psychology, 13:921388].
    Reads Table 1.XLSX (emotional mean scores across 359 emojis) and
    Table 2.XLSX (variance and standard error estimates) from data/emoji/,
    categorizing each emoji into canonical Indonesian affective tokens:
    _emosi_bahagia_, _emosi_sedih_, _emosi_marah_, _emosi_takut_, _emosi_cinta_.
    """
    p1 = resolve_path(table1_path or os.path.join("data", "emoji", "Table 1.XLSX"))
    p2 = resolve_path(table2_path or os.path.join("data", "emoji", "Table 2.XLSX"))

    if not os.path.exists(p1) or not os.path.exists(p2):
        # Fallback to preprocessor configuration
        cfg = load_preprocessor_config()
        return cfg.get("emoji_sentiment_dict", {})

    try:
        import pandas as pd

        df1 = pd.read_excel(p1)
        df2 = pd.read_excel(p2)
        merged = pd.merge(df1, df2, on=["id", "name", "emoji"])

        def _map_row(row):
            name = str(row["name"]).lower()
            if re.search(r"\b(heart|hearts|kiss|kisses|kissing|love|cupid|couple|hug|hugging)\b", name):
                if "broken heart" not in name:
                    return " _emosi_cinta_ "
            if "tears of joy" in name or "rolling on the floor laughing" in name:
                return " _emosi_bahagia_ "
            if re.search(r"\b(crying|sob|sad|disappointed|frown|frowning|pensive|broken heart|wilted|droplet|sweat)\b", name):
                if "grinning face with sweat" not in name and "sweat droplets" not in name:
                    return " _emosi_sedih_ "
            if re.search(r"\b(rage|angry|pouting|swearing|vomit|vomiting|nauseat|nauseated|middle finger|devil|horns|skull|poo|unamused)\b", name):
                return " _emosi_marah_ "
            if re.search(r"\b(scream|screaming|fear|fearful|anxious|cold|anguish|anguished|grimac|grimacing|plead|pleading)\b", name):
                return " _emosi_takut_ "

            a = row["anger"]
            d = row["disgust"]
            f = row["fear"]
            s = row["sadness"]
            pos = row["positive"]
            neg = row["negative"]

            if neg > pos:
                max_neg = max(a, d, f, s)
                if max_neg in (a, d):
                    return " _emosi_marah_ "
                elif max_neg == s:
                    return " _emosi_sedih_ "
                elif max_neg == f:
                    return " _emosi_takut_ "

            return " _emosi_bahagia_ "

        mle_dict = {}
        for _, r in merged.iterrows():
            mle_dict[str(r["emoji"])] = _map_row(r)
        return mle_dict
    except Exception:
        cfg = load_preprocessor_config()
        return cfg.get("emoji_sentiment_dict", {})


def load_colloquial_lexicon(csv_path: Optional[str] = None) -> Dict[str, str]:
    """
    Loads the Colloquial Indonesian Lexicon established by Salsabila et al. (2018)
    [Proceedings of the 2018 International Conference on Asian Language Processing (IALP), pp. 226-229].
    Reads data/slangs/colloquial-indonesian-lexicon.csv, normalizing colloquial Indonesian
    slang, abbreviations, and informal word forms to standard formal Indonesian equivalents.
    """
    p = resolve_path(csv_path or os.path.join("data", "slangs", "colloquial-indonesian-lexicon.csv"))
    if not os.path.exists(p):
        cfg = load_preprocessor_config()
        return cfg.get("slang_dict", {})
    try:
        import pandas as pd
        df = pd.read_csv(p)
        clean_df = df.dropna(subset=["slang", "formal"])
        slang_map = dict(zip(
            clean_df["slang"].astype(str).str.lower().str.strip(),
            clean_df["formal"].astype(str).str.lower().str.strip()
        ))
        return slang_map
    except Exception:
        cfg = load_preprocessor_config()
        return cfg.get("slang_dict", {})


AFFECTIVE_TWITTER_SLANG: Dict[str, str] = {
    # Love / Romantic affection
    "bucin": "sangat cinta",
    "ayang": "sayang",
    "ayank": "sayang",
    "ayg": "sayang",
    "lop": "cinta",
    "luvv": "cinta",
    "luvvv": "cinta",
    "uwu": "lucu menggemaskan",
    "gemes": "gemas sayang",
    "gemess": "gemas sayang",
    "kgn": "kangen rindu",
    "rinduu": "rindu",
    # Sadness / Grief / Heartbreak
    "ambyar": "hancur sedih",
    "nyesek": "sedih menyakitkan",
    "patahhati": "patah hati sedih",
    "galau": "sedih gelisah",
    "mewek": "menangis sedih",
    "nangiss": "menangis sedih",
    "nangis": "menangis sedih",
    "sedihh": "sedih",
    # Anger / Outrage
    "bgsd": "bangsat marah",
    "bgst": "bangsat marah",
    "kesel": "kesal marah",
    "kzll": "kesal marah",
    "kzl": "kesal",
    "emosian": "emosi marah",
    "murka": "sangat marah",
    "jancuk": "jancuk marah",
    "jancok": "jancok marah",
    "kampret": "kampret marah",
    # Happy / Celebration
    "hepi": "gembira",
    "hppy": "senang gembira",
    "seneng": "senang gembira",
    "sneng": "senang",
    "girang": "gembira senang",
    "asik": "asyik senang",
    "seruu": "seru gembira",
    "hore": "hore gembira",
}


def _get_default_dicts():
    try:
        cfg = load_preprocessor_config()
        slang = cfg.get("slang_dict", {})
        if not slang:
            slang = load_colloquial_lexicon()
        # Merge affective Twitter colloquial slang to disambiguate fine-grained emotions
        slang_combined = dict(slang)
        slang_combined.update(AFFECTIVE_TWITTER_SLANG)

        emoji = cfg.get("emoji_sentiment_dict", {})
        if not emoji:
            emoji = load_mle_emoji_lexicon()
        emoticon = cfg.get("emoticon_dict", {})
        return slang_combined, emoji, emoticon
    except Exception:
        return dict(AFFECTIVE_TWITTER_SLANG), {}, {}


_DEFAULT_SLANG, _DEFAULT_EMOJI, _DEFAULT_EMOTICON = _get_default_dicts()

INDONESIAN_SLANG_DICT: Dict[str, str] = _DEFAULT_SLANG
EMOJI_SENTIMENT_DICT: Dict[str, str] = _DEFAULT_EMOJI
EMOTICON_DICT: Dict[str, str] = _DEFAULT_EMOTICON


class IndonesianPreprocessor(BasePreprocessor):
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


class ArabicPreprocessor(BasePreprocessor):
    """
    Arabic Text Preprocessor for Data Augmentation and Classification.
    Provides standard Arabic normalization:
    - Diacritics (tashkeel) removal
    - Tatweel (kashida) stripping
    - Alef / Hamza normalization
    - URL and user mention removal
    - Character elongation reduction
    - Whitespace normalization
    """

    def __init__(
        self,
        remove_diacritics: bool = True,
        remove_tatweel: bool = True,
        normalize_alef: bool = True,
        normalize_elongation: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_extra_whitespace: bool = True,
    ):
        self.remove_diacritics = remove_diacritics
        self.remove_tatweel = remove_tatweel
        self.normalize_alef = normalize_alef
        self.normalize_elongation = normalize_elongation
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_extra_whitespace = remove_extra_whitespace

    def preprocess(self, text: str) -> str:
        """Applies comprehensive Arabic preprocessing to the input string."""
        if not isinstance(text, str):
            text = str(text)

        # 1. URL removal
        if self.remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # 2. Mention removal
        if self.remove_mentions:
            text = re.sub(r"@\w+", "", text)

        # 3. Tashkeel / diacritics removal (Fathatan, Dammatan, Kasratan, Fatha, Damma, Kasra, Shadda, Sukun)
        if self.remove_diacritics:
            text = re.sub(r"[\u064B-\u0652\u0670]", "", text)

        # 4. Tatweel / kashida removal
        if self.remove_tatweel:
            text = re.sub(r"\u0640+", "", text)

        # 5. Alef and Hamza normalization
        if self.normalize_alef:
            # Normalize [إأآا] to ا
            text = re.sub(r"[إأآ]", "ا", text)
            # Normalize final ى to ي
            text = re.sub(r"ى\b", "ي", text)
            # Normalize ة to ه
            text = re.sub(r"ة\b", "ه", text)

        # 6. Character elongation reduction (e.g. جمييييل -> جميل)
        if self.normalize_elongation:
            text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        # 7. Clean up extra whitespaces
        if self.remove_extra_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        return text


class PassThroughPreprocessor(BasePreprocessor):
    """Fallback preprocessor for generic languages."""

    def __init__(
        self,
        lower: bool = False,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_extra_whitespace: bool = True,
    ):
        self.lower = lower
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_extra_whitespace = remove_extra_whitespace

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        if self.remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", "", text)
        if self.remove_mentions:
            text = re.sub(r"@\w+", "", text)
        if self.lower:
            text = text.lower()
        if self.remove_extra_whitespace:
            text = re.sub(r"\s+", " ", text).strip()
        return text


def get_preprocessor(language: str = "indonesian", **kwargs) -> BasePreprocessor:
    """Factory returning the appropriate preprocessor for the specified language."""
    lang = str(language).lower().strip()
    if lang in ["indonesian", "id", "indo"]:
        return IndonesianPreprocessor(**kwargs)
    elif lang in ["arabic", "ar"]:
        return ArabicPreprocessor(**kwargs)
    else:
        return PassThroughPreprocessor(**kwargs)

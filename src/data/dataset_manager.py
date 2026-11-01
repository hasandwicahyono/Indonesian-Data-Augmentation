"""
Dataset manager and loader for Indonesian NLP datasets.
Provides structured containers (CustomDataset) and automated loaders for the 4 Indonesian datasets:
1. SmSA (doc-sentiment-prosa)
2. EmoT (emotion-twitter)
3. HoASA (absa-airy)
4. MovieOpini (dataset_tweet_sentiment_opini_film)
"""

import os
import pandas as pd
from typing import List, Dict, Optional
from sklearn.model_selection import train_test_split

from src.utils.path_utils import resolve_path


class CustomDataset:
    """Class to hold and structure datasets."""

    def __init__(
        self,
        name: str,
        train: pd.DataFrame,
        test: pd.DataFrame,
        label_list: List[str],
        val: Optional[pd.DataFrame] = None,
    ):
        self.name = name
        self.train = train.reset_index(drop=True)
        self.test = test.reset_index(drop=True)
        self.val = val.reset_index(drop=True) if val is not None else None
        self.label_list = [str(lbl) for lbl in label_list]


class IndonesianDatasetsManager:
    """Loads and standardizes Indonesian datasets strictly from project data repository."""

    def __init__(self, data_dir: Optional[str] = None, project_root: Optional[str] = None):
        default_raw_dir = resolve_path("data/raw")

        if data_dir is not None:
            self.data_dir = os.path.abspath(data_dir)
        elif project_root is not None:
            candidates = [
                os.path.join(project_root, "data", "raw"),
                os.path.join(project_root, "Indonesian-Data-Augmentation", "data", "raw"),
                os.path.abspath(project_root),
                default_raw_dir,
            ]
            resolved = None
            for cand in candidates:
                if os.path.exists(cand) and (
                    os.path.exists(os.path.join(cand, "smsa_doc-sentiment-prosa"))
                    or os.path.exists(os.path.join(cand, "dataset_tweet_sentiment_opini_film.csv"))
                ):
                    resolved = cand
                    break
            self.data_dir = resolved if resolved is not None else default_raw_dir
        else:
            self.data_dir = default_raw_dir

    def load_smsa(self) -> CustomDataset:
        """SmSA (Prosa Document Sentiment): 3 classes (positive, negative, neutral)"""
        smsa_dir = os.path.join(self.data_dir, "smsa_doc-sentiment-prosa")
        train_df = pd.read_csv(
            os.path.join(smsa_dir, "train_preprocess.tsv"),
            sep="\t",
            header=None,
            names=["text", "label"],
        )
        val_df = pd.read_csv(
            os.path.join(smsa_dir, "valid_preprocess.tsv"),
            sep="\t",
            header=None,
            names=["text", "label"],
        )
        test_df = pd.read_csv(
            os.path.join(smsa_dir, "test_preprocess.tsv"),
            sep="\t",
            header=None,
            names=["text", "label"],
        )
        labels = sorted(list(train_df["label"].unique()))
        return CustomDataset("SmSA", train_df, test_df, labels, val=val_df)

    def load_emot(self) -> CustomDataset:
        """EmoT (Twitter Emotion Analysis): 5 classes (anger, happy, sadness, fear, love)"""
        emot_dir = os.path.join(self.data_dir, "emot_emotion-twitter")
        train_raw = pd.read_csv(os.path.join(emot_dir, "train_preprocess.csv"))
        val_raw = pd.read_csv(os.path.join(emot_dir, "valid_preprocess.csv"))
        test_raw = pd.read_csv(os.path.join(emot_dir, "test_preprocess.csv"))

        train_df = pd.DataFrame({"text": train_raw["tweet"], "label": train_raw["label"]})
        val_df = pd.DataFrame({"text": val_raw["tweet"], "label": val_raw["label"]})
        test_df = pd.DataFrame({"text": test_raw["tweet"], "label": test_raw["label"]})
        labels = sorted(list(train_df["label"].unique()))
        return CustomDataset("EmoT", train_df, test_df, labels, val=val_df)

    def load_hoasa(self, aspect_target: str = "general") -> CustomDataset:
        """HoASA (Airy Rooms Hotel Reviews): Aspect sentiment (default: 'general')"""
        hoasa_dir = os.path.join(self.data_dir, "hoasa_absa-airy")
        train_raw = pd.read_csv(os.path.join(hoasa_dir, "train_preprocess.csv"))
        val_raw = pd.read_csv(os.path.join(hoasa_dir, "valid_preprocess.csv"))
        test_raw = pd.read_csv(os.path.join(hoasa_dir, "test_preprocess.csv"))

        train_df = pd.DataFrame({"text": train_raw["review"], "label": train_raw[aspect_target]})
        val_df = pd.DataFrame({"text": val_raw["review"], "label": val_raw[aspect_target]})
        test_df = pd.DataFrame({"text": test_raw["review"], "label": test_raw[aspect_target]})
        labels = sorted(list(train_df["label"].unique()))
        return CustomDataset("HoASA", train_df, test_df, labels, val=val_df)

    def load_movie_opini(
        self, val_size: float = 0.15, test_size: float = 0.15, random_state: int = 42
    ) -> CustomDataset:
        """MovieOpini (Indonesian Movie Opinion Tweets): 200 records (positive, negative)."""
        csv_path = os.path.join(self.data_dir, "dataset_tweet_sentiment_opini_film.csv")
        raw_df = pd.read_csv(csv_path)
        clean_df = pd.DataFrame({"text": raw_df["Text Tweet"], "label": raw_df["Sentiment"]})
        train_val_df, test_df = train_test_split(
            clean_df, test_size=test_size, random_state=random_state, stratify=clean_df["label"]
        )
        rel_val_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=rel_val_size, random_state=random_state, stratify=train_val_df["label"]
        )
        labels = sorted(list(clean_df["label"].unique()))
        return CustomDataset("MovieOpini", train_df, test_df, labels, val=val_df)

    def load_all_datasets(self) -> Dict[str, CustomDataset]:
        """Loads and returns all 4 comparable Indonesian datasets."""
        return {
            "SmSA": self.load_smsa(),
            "EmoT": self.load_emot(),
            "HoASA": self.load_hoasa(),
            "MovieOpini": self.load_movie_opini(),
        }

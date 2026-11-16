"""
Dataset manager and loader for Indonesian NLP datasets.
Provides structured containers (CustomDataset) and automated loaders for the 4 Indonesian datasets:
1. SmSA (doc-sentiment-prosa)
2. HoASA (absa-airy)
3. EmoT (emotion-twitter)
4. NusaX-Indonesian (nusax-sentiment-indonesian)
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

    def load_nusax_indonesian(self, binary_only: bool = False) -> CustomDataset:
        """NusaX Indonesian Sentiment (Winata et al., 2023):
        Total 1,000 samples (500 train, 100 valid, 400 test) across positive, neutral, negative.
        If binary_only=True, filters to positive and negative (761 samples, IR ~ 1.01x),
        serving as a standardized direct equivalent to MovieOpini.
        """
        nusax_dir = os.path.join(self.data_dir, "nusax_sentiment_indonesian")
        train_raw = pd.read_csv(os.path.join(nusax_dir, "train.csv"))
        val_raw = pd.read_csv(os.path.join(nusax_dir, "valid.csv"))
        test_raw = pd.read_csv(os.path.join(nusax_dir, "test.csv"))

        train_df = pd.DataFrame({"text": train_raw["text"], "label": train_raw["label"]})
        val_df = pd.DataFrame({"text": val_raw["text"], "label": val_raw["label"]})
        test_df = pd.DataFrame({"text": test_raw["text"], "label": test_raw["label"]})

        if binary_only:
            train_df = train_df[train_df["label"].isin(["positive", "negative"])].reset_index(drop=True)
            val_df = val_df[val_df["label"].isin(["positive", "negative"])].reset_index(drop=True)
            test_df = test_df[test_df["label"].isin(["positive", "negative"])].reset_index(drop=True)

        labels = sorted(list(train_df["label"].unique()))
        name = "NusaX-Indonesian-Binary" if binary_only else "NusaX-Indonesian"
        return CustomDataset(name, train_df, test_df, labels, val=val_df)

    def load_all_datasets(self, include_movie_opini: bool = False) -> Dict[str, CustomDataset]:
        """Loads and returns all investigated Indonesian datasets according to canonical config."""
        from src.utils.config import get_investigated_datasets, get_investigated_settings

        datasets = {}
        for dname in get_investigated_datasets():
            setting = get_investigated_settings(dname)
            method_name = setting.get("loader_method", f"load_{dname.lower().replace('-', '_')}")
            if hasattr(self, method_name):
                datasets[dname] = getattr(self, method_name)()
            elif dname in ["NusaX", "NusaX-Indonesian"]:
                datasets[dname] = self.load_nusax_indonesian()
            else:
                datasets[dname] = self.load_generic_dataset(dname)

        if include_movie_opini and "MovieOpini" not in datasets:
            datasets["MovieOpini"] = self.load_movie_opini()
        return datasets

    def load_generic_dataset(
        self,
        dataset_name_or_path: str,
        text_col: Optional[str] = None,
        label_col: Optional[str] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        seed: int = 42,
    ) -> CustomDataset:
        """
        Generic plug-and-play dataset auto-loader.
        Supports:
          - A dataset name located under self.data_dir or data/raw/
          - A directory containing train/val/test splits (CSV, TSV, Parquet)
          - A single standalone dataset file, auto-creating stratified train/val/test splits
        Auto-detects text and label columns, extracts unique labels, and returns CustomDataset.
        """
        def _read_table(fpath: str) -> pd.DataFrame:
            ext = os.path.splitext(fpath)[-1].lower()
            if ext == ".tsv":
                return pd.read_csv(fpath, sep="\t")
            elif ext == ".parquet":
                return pd.read_parquet(fpath)
            else:
                return pd.read_csv(fpath)

        def _standardize_cols(df: pd.DataFrame, u_text: Optional[str], u_label: Optional[str]) -> pd.DataFrame:
            cols_lower = {str(c).lower(): c for c in df.columns}
            t_cand = [u_text] if u_text else ["text", "tweet", "sentence", "content", "sms", "review", "document", "teks"]
            l_cand = [u_label] if u_label else ["label", "sentiment", "category", "class", "target", "kategori", "label_id"]

            t_col = None
            for c in t_cand:
                if c and c.lower() in cols_lower:
                    t_col = cols_lower[c.lower()]
                    break
            if t_col is None:
                str_cols = df.select_dtypes(include=["object", "string"]).columns
                t_col = str_cols[0] if len(str_cols) > 0 else df.columns[0]

            l_col = None
            for c in l_cand:
                if c and c.lower() in cols_lower and cols_lower[c.lower()] != t_col:
                    l_col = cols_lower[c.lower()]
                    break
            if l_col is None:
                remaining = [c for c in df.columns if c != t_col]
                l_col = remaining[-1] if remaining else df.columns[-1]

            out_df = pd.DataFrame({
                "text": df[t_col].astype(str),
                "label": df[l_col].astype(str).str.strip(),
            })
            out_df = out_df.dropna(subset=["text", "label"])
            out_df = out_df[out_df["text"].str.strip() != ""].reset_index(drop=True)
            return out_df

        # Resolve path
        resolved_path = None
        candidates = [
            dataset_name_or_path,
            os.path.join(self.data_dir, dataset_name_or_path),
            os.path.join(resolve_path("data/raw"), dataset_name_or_path),
            os.path.join(resolve_path("data"), dataset_name_or_path),
        ]
        # Also check with extensions
        ext_candidates = []
        for c in candidates:
            ext_candidates.append(c)
            for ext in [".csv", ".tsv", ".parquet"]:
                ext_candidates.append(c + ext)

        for cand in ext_candidates:
            if os.path.exists(cand):
                resolved_path = os.path.abspath(cand)
                break

        if resolved_path is None:
            raise FileNotFoundError(
                f"Dataset '{dataset_name_or_path}' not found. Searched in: {candidates}"
            )

        dname = os.path.basename(os.path.normpath(resolved_path))
        for ext in [".csv", ".tsv", ".parquet"]:
            if dname.lower().endswith(ext):
                dname = dname[:-len(ext)]

        if os.path.isdir(resolved_path):
            files = os.listdir(resolved_path)
            train_f, test_f, val_f = None, None, None
            for f in sorted(files):
                f_low = f.lower()
                full_f = os.path.join(resolved_path, f)
                if not os.path.isfile(full_f):
                    continue
                if any(k in f_low for k in ["train", "latih"]):
                    train_f = full_f
                elif any(k in f_low for k in ["test", "uji"]):
                    test_f = full_f
                elif any(k in f_low for k in ["val", "valid"]):
                    val_f = full_f

            if train_f is not None and test_f is not None:
                train_df = _standardize_cols(_read_table(train_f), text_col, label_col)
                test_df = _standardize_cols(_read_table(test_f), text_col, label_col)
                val_df = _standardize_cols(_read_table(val_f), text_col, label_col) if val_f else None
            elif train_f is not None:
                # Only train file in dir -> split
                full_df = _standardize_cols(_read_table(train_f), text_col, label_col)
                min_class_cnt = full_df["label"].value_counts().min()
                strat = full_df["label"] if (full_df["label"].nunique() > 1 and min_class_cnt >= 2) else None
                train_val_df, test_df = train_test_split(full_df, test_size=test_size, random_state=seed, stratify=strat)
                if val_size > 0:
                    rel_val = val_size / (1.0 - test_size)
                    strat_tv = train_val_df["label"] if (train_val_df["label"].nunique() > 1 and train_val_df["label"].value_counts().min() >= 2) else None
                    train_df, val_df = train_test_split(train_val_df, test_size=rel_val, random_state=seed, stratify=strat_tv)
                else:
                    train_df, val_df = train_val_df, None
            else:
                # Pick first data file in directory
                first_f = [os.path.join(resolved_path, f) for f in files if f.endswith((".csv", ".tsv", ".parquet"))]
                if not first_f:
                    raise FileNotFoundError(f"No tabular data file (csv/tsv/parquet) found in {resolved_path}")
                full_df = _standardize_cols(_read_table(first_f[0]), text_col, label_col)
                min_class_cnt = full_df["label"].value_counts().min()
                strat = full_df["label"] if (full_df["label"].nunique() > 1 and min_class_cnt >= 2) else None
                train_val_df, test_df = train_test_split(full_df, test_size=test_size, random_state=seed, stratify=strat)
                if val_size > 0:
                    rel_val = val_size / (1.0 - test_size)
                    strat_tv = train_val_df["label"] if (train_val_df["label"].nunique() > 1 and train_val_df["label"].value_counts().min() >= 2) else None
                    train_df, val_df = train_test_split(train_val_df, test_size=rel_val, random_state=seed, stratify=strat_tv)
                else:
                    train_df, val_df = train_val_df, None
        else:
            # Standalone file
            full_df = _standardize_cols(_read_table(resolved_path), text_col, label_col)
            min_class_cnt = full_df["label"].value_counts().min()
            strat = full_df["label"] if (full_df["label"].nunique() > 1 and min_class_cnt >= 2) else None
            train_val_df, test_df = train_test_split(full_df, test_size=test_size, random_state=seed, stratify=strat)
            if val_size > 0:
                rel_val = val_size / (1.0 - test_size)
                strat_tv = train_val_df["label"] if (train_val_df["label"].nunique() > 1 and train_val_df["label"].value_counts().min() >= 2) else None
                train_df, val_df = train_test_split(train_val_df, test_size=rel_val, random_state=seed, stratify=strat_tv)
            else:
                train_df, val_df = train_val_df, None

        labels = sorted(list(train_df["label"].unique()))
        print(f"  [GENERIC DATASET LOADED] '{dname}': Train={len(train_df)}, Val={len(val_df) if val_df is not None else 0}, Test={len(test_df)}, Classes ({len(labels)})={labels}")
        return CustomDataset(dname, train_df, test_df, labels, val=val_df)


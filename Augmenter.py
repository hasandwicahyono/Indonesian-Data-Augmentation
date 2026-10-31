"""
Indonesian Data Augmenter Pipeline (Phase 1: Generation & Similarity Computation).
Uses an Indonesian GPT-2 model (e.g. cahya/gpt2-small-indonesian-522M) and multilingual SentenceTransformer.
Computes Euclidean, Cosine, Jaccard, and BLEU similarities for each sample.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from tqdm.auto import tqdm
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer

from IndonesianPreprocessor import IndonesianPreprocessor
from SimilarityMeasures import (
    euclidean_distance,
    cos_similarity,
    jaccard_similarity,
    calculate_bleu_scores,
    distance_to_similarity,
)


class IndonesianAugmenter:
    def __init__(
        self,
        gen_model_name: str = "cahya/gpt2-small-indonesian-522M",
        embed_model_name: str = "sentence-transformers/distiluse-base-multilingual-cased-v2",
        device: Optional[int] = None,
        max_length: int = 128,
        num_beams: int = 5,
        top_p: float = 0.95,
        repetition_penalty: float = 2.5,
        no_repeat_ngram_size: int = 3,
    ):
        """
        Args:
            gen_model_name: Hugging Face Indonesian generative model.
            embed_model_name: SentenceTransformer model for embedding representations.
            device: GPU device ID (0, 1...) or -1 for CPU.
        """
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if device is None:
            if num_gpus > 0:
                self.device = 0
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = -1
        else:
            self.device = device

        self.gen_model_name = gen_model_name
        self.embed_model_name = embed_model_name
        self.max_length = max_length
        self.num_beams = num_beams
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size

        self.preprocessor = IndonesianPreprocessor()
        self._init_models()

    def _init_models(self):
        """Initializes generator and embedder models."""
        print(f"Loading Generative Model: {self.gen_model_name} on device: {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.gen_model_name)
            pad_id = self.tokenizer.eos_token_id or 0
            self.gen_pipeline = pipeline(
                "text-generation",
                model=self.gen_model_name,
                tokenizer=self.tokenizer,
                device=self.device,
                pad_token_id=pad_id,
            )
        except Exception as e:
            print(f"Warning: Could not load {self.gen_model_name} ({e}).")
            print("Falling back to Hugging Face Indonesian GPT2 pipeline.")
            self.gen_pipeline = pipeline(
                "text-generation",
                model="flax-community/indonesian-gpt2",
                device=self.device,
            )

        print(f"Loading Sentence Embedding Model: {self.embed_model_name}")
        if isinstance(self.device, str) and (self.device.startswith("cuda") or self.device in ("mps", "cpu")):
            embed_device = self.device
        elif isinstance(self.device, int) and self.device >= 0:
            embed_device = f"cuda:{self.device}"
        else:
            embed_device = "cpu"
        self.embed_model = SentenceTransformer(self.embed_model_name, device=embed_device)

    def generate_continuation(self, text: str) -> str:
        """Generates continuation text using causal language model."""
        clean_text = self.preprocessor.preprocess(text)
        prompt = clean_text[:self.max_length]

        try:
            output = self.gen_pipeline(
                prompt,
                max_new_tokens=40,
                num_beams=self.num_beams,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                early_stopping=True,
            )
            full_gen = output[0]["generated_text"]
            return full_gen
        except Exception as e:
            print(f"Generation error: {e}")
            return prompt

    def process_dataset(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "label",
        sample_limit: Optional[int] = None,
        target_labels: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Processes a dataset: generates augmentations and computes similarity metrics.
        Returns intermediate DataFrame ready for Phase 2 threshold filtering.
        If target_labels is provided, focuses generation strictly on those target/minority classes.
        """
        records = []
        if target_labels is not None:
            norm_targets = [str(l).strip().lower() for l in target_labels]
            filtered_df = df[df[label_column].astype(str).str.strip().str.lower().isin(norm_targets)].copy()
            target_df = filtered_df if sample_limit is None else filtered_df.head(sample_limit)
            print(f"Targeting minority classes {target_labels}: {len(target_df)} seed rows selected.")
        else:
            target_df = df if sample_limit is None else df.head(sample_limit)

        print(f"Starting generation and similarity computation for {len(target_df)} rows...")
        for _, row in tqdm(target_df.iterrows(), total=len(target_df)):
            orig_text = str(row[text_column])
            label = row[label_column]

            clean_orig = self.preprocessor.preprocess(orig_text)
            full_gen = self.generate_continuation(clean_orig)

            # Split new text and all text
            if len(full_gen) > len(clean_orig):
                new_text = full_gen[len(clean_orig):].strip()
            else:
                new_text = full_gen.strip()
            all_text = full_gen.strip()

            # Encode embeddings
            emb_orig = self.embed_model.encode(clean_orig).tolist()
            emb_new = self.embed_model.encode(new_text if new_text else clean_orig).tolist()

            # Compute similarities
            e_dist = euclidean_distance(emb_orig, emb_new)
            e_sim = distance_to_similarity(e_dist)
            c_sim = cos_similarity(emb_orig, emb_new)
            j_sim = jaccard_similarity(clean_orig, new_text if new_text else all_text)
            b1, b2, b3, b4 = calculate_bleu_scores(clean_orig, new_text if new_text else all_text)

            records.append({
                "text": orig_text,
                "label": label,
                "new_text": new_text,
                "all_text": all_text,
                "original_embedding": ",".join(f"{v:.5f}" for v in emb_orig[:50]),
                "new_embedding": ",".join(f"{v:.5f}" for v in emb_new[:50]),
                "ecu_distance": e_dist,
                "ecu_similarity": e_sim,
                "cos_similarity": c_sim,
                "jacc_similarity": j_sim,
                "bleu_sim_1": b1,
                "bleu_sim_2": b2,
                "bleu_sim_3": b3,
                "bleu_sim_4": b4,
            })

        return pd.DataFrame(records)

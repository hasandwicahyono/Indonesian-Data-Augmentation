"""
Indonesian Data Augmenter Pipeline (Phase 1: Generation & Similarity Computation).
Supports both First-Generation (2020–2021) and Contemporary Foundation (2024) Generative Models.
Computes Euclidean, Cosine, Jaccard, and BLEU similarities for each sample.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from tqdm.auto import tqdm
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer

from src.data.preprocessor import IndonesianPreprocessor
from src.augmentation.similarity import (
    euclidean_distance,
    cos_similarity,
    jaccard_similarity,
    calculate_bleu_scores,
    distance_to_similarity,
)
from src.augmentation.registry import (
    GENERATIVE_MODELS_REGISTRY,
    list_candidate_models,
    display_model_comparisons,
)
from src.augmentation.generative_models import benchmark_candidate_generative_models


def _is_accelerate_available() -> bool:
    """Checks if Hugging Face accelerate library is installed for multi-GPU device_map."""
    try:
        import accelerate  # noqa: F401
        return True
    except ImportError:
        return False


class IndonesianAugmenter:
    def __init__(
        self,
        gen_model_name: str = "cahya/gpt2-small-indonesian-522M",
        embed_model_name: str = "sentence-transformers/distiluse-base-multilingual-cased-v2",
        device: Optional[Any] = None,
        max_length: int = 128,
        num_beams: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        do_sample: Optional[bool] = None,
    ):
        """
        Args:
            gen_model_name: Hugging Face model identifier or registry alias (e.g. 'gpt2-small', 'sailor-0.5b', 'qwen2.5-0.5b').
            embed_model_name: SentenceTransformer model for embedding representations.
            device: GPU device ID (0, 1...) or 'mps' or -1 / 'cpu'.
        """
        resolved_model_id = gen_model_name
        self.model_meta: Optional[Dict[str, Any]] = None

        norm_name = gen_model_name.strip().lower()
        if norm_name in GENERATIVE_MODELS_REGISTRY:
            self.model_meta = GENERATIVE_MODELS_REGISTRY[norm_name]
            resolved_model_id = self.model_meta["model_id"]
        else:
            for k, meta in GENERATIVE_MODELS_REGISTRY.items():
                if meta["model_id"].lower() == norm_name:
                    self.model_meta = meta
                    resolved_model_id = meta["model_id"]
                    break

        self.gen_model_name = resolved_model_id
        self.embed_model_name = embed_model_name
        self.max_length = max_length

        defaults = self.model_meta["default_params"] if self.model_meta is not None else {
            "num_beams": 5,
            "top_p": 0.95,
            "temperature": 1.0,
            "repetition_penalty": 2.5,
            "no_repeat_ngram_size": 3,
            "do_sample": False,
        }

        self.num_beams = num_beams if num_beams is not None else defaults["num_beams"]
        self.top_p = top_p if top_p is not None else defaults["top_p"]
        self.temperature = temperature if temperature is not None else defaults["temperature"]
        self.repetition_penalty = repetition_penalty if repetition_penalty is not None else defaults["repetition_penalty"]
        self.no_repeat_ngram_size = no_repeat_ngram_size if no_repeat_ngram_size is not None else defaults["no_repeat_ngram_size"]
        self.do_sample = do_sample if do_sample is not None else defaults["do_sample"]

        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.use_device_map = False
        accelerate_ready = _is_accelerate_available()

        if device is None:
            if self.num_gpus > 1 and accelerate_ready:
                self.device = "auto"
                self.use_device_map = True
            elif self.num_gpus > 0:
                self.device = 0
                if self.num_gpus > 1 and not accelerate_ready:
                    print(
                        f"Notice: {self.num_gpus} GPUs detected. 'accelerate' package is not installed in this environment; "
                        f"defaulting text generation to GPU 0."
                    )
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = -1
        elif isinstance(device, str) and device.lower() in ("auto", "balanced", "sequential"):
            if accelerate_ready:
                self.device = device.lower()
                self.use_device_map = True
            else:
                print(f"Warning: device='{device}' requires 'accelerate'. Defaulting text generation to GPU 0 / CPU.")
                self.device = 0 if self.num_gpus > 0 else -1
        else:
            self.device = device

        self.preprocessor = IndonesianPreprocessor()
        self._init_models()

    @classmethod
    def list_candidate_models(cls) -> List[Dict[str, Any]]:
        return list_candidate_models()

    @classmethod
    def display_model_comparisons(cls):
        display_model_comparisons()

    @classmethod
    def benchmark_generative_models(
        cls,
        models: Optional[List[str]] = None,
        prompts: Optional[List[str]] = None,
        device: Optional[str] = None,
        output_csv: str = "data/results/extended/generative_models_comparison.csv",
        output_tex: str = "paper/table_generative_models_comparison.tex",
        force: bool = False,
    ) -> pd.DataFrame:
        return benchmark_candidate_generative_models(
            models=models,
            prompts=prompts,
            device=device,
            output_csv=output_csv,
            output_tex=output_tex,
            force=force,
        )

    def _init_models(self):
        """Initializes generator and embedder models."""
        gen_desc = f"{self.model_meta['generation']} ({self.model_meta['parameters']})" if self.model_meta else "Custom"
        if self.use_device_map:
            device_desc = f"Multi-GPU (device_map='{self.device}', {self.num_gpus} GPUs detected)"
        else:
            device_desc = str(self.device)
        print(f"Loading Generative Model: {self.gen_model_name} [{gen_desc}] on device: {device_desc}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.gen_model_name, trust_remote_code=True)
        pad_id = self.tokenizer.eos_token_id or 0
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        pipeline_kwargs: Dict[str, Any] = {
            "model": self.gen_model_name,
            "tokenizer": self.tokenizer,
            "pad_token_id": pad_id,
            "trust_remote_code": True,
        }
        if self.use_device_map:
            pipeline_kwargs["device_map"] = self.device
        else:
            pipeline_kwargs["device"] = self.device

        try:
            self.gen_pipeline = pipeline("text-generation", **pipeline_kwargs)
        except Exception as e:
            if self.use_device_map:
                print(f"Warning: Loading with device_map='{self.device}' failed ({e}).")
                fallback_device = 0 if self.num_gpus > 0 else -1
                print(f"Retrying loading '{self.gen_model_name}' on single device (device={fallback_device})...")
                self.use_device_map = False
                self.device = fallback_device
                self.gen_pipeline = pipeline(
                    "text-generation",
                    model=self.gen_model_name,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    pad_token_id=pad_id,
                    trust_remote_code=True,
                )
            else:
                raise RuntimeError(f"Failed to load generative model '{self.gen_model_name}': {e}") from e

        print(f"Loading Sentence Embedding Model: {self.embed_model_name}")
        if self.use_device_map or self.device == "auto":
            embed_device = "cuda:0" if self.num_gpus > 0 else "cpu"
        elif isinstance(self.device, str) and (self.device.startswith("cuda") or self.device in ("mps", "cpu")):
            embed_device = self.device
        elif isinstance(self.device, int) and self.device >= 0:
            embed_device = f"cuda:{self.device}"
        else:
            embed_device = "cpu"
        self.embed_model = SentenceTransformer(self.embed_model_name, device=embed_device)

    def generate_continuation(self, text: str) -> str:
        """Generates continuation text using causal language model with sentence boundary truncation."""
        clean_text = self.preprocessor.preprocess(text)
        prompt = clean_text[:self.max_length]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": 40,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
        }
        if self.do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["top_p"] = self.top_p
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["num_beams"] = self.num_beams
            gen_kwargs["early_stopping"] = True

        try:
            output = self.gen_pipeline(prompt, **gen_kwargs)
            full_gen = output[0]["generated_text"]

            if len(full_gen) > len(prompt):
                suffix = full_gen[len(prompt):].strip()
                last_punct = max(suffix.rfind("."), suffix.rfind("!"), suffix.rfind("?"))
                if last_punct > 5:
                    suffix = suffix[: last_punct + 1].strip()
                full_gen = prompt + " " + suffix
            return full_gen.strip()
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
        """
        records = []
        if target_labels is not None:
            norm_targets = [str(l).strip().lower() for l in target_labels]
            filtered_df = df[df[label_column].astype(str).str.strip().str.lower().isin(norm_targets)].copy()
            target_df = filtered_df if sample_limit is None else filtered_df.head(sample_limit)
            print(f"Targeting minority classes {target_labels}: {len(target_df)} seed rows selected.")
        else:
            target_df = df if sample_limit is None else df.head(sample_limit)

        gen_model_tag = self.model_meta["generation"] if self.model_meta else self.gen_model_name
        print(f"Starting generation [{gen_model_tag}] & similarity computation for {len(target_df)} rows...")

        for _, row in tqdm(target_df.iterrows(), total=len(target_df)):
            orig_text = str(row[text_column])
            label = row[label_column]

            clean_orig = self.preprocessor.preprocess(orig_text)
            full_gen = self.generate_continuation(clean_orig)

            if len(full_gen) > len(clean_orig):
                new_text = full_gen[len(clean_orig):].strip()
            else:
                new_text = full_gen.strip()
            all_text = full_gen.strip()

            emb_orig = self.embed_model.encode(clean_orig).tolist()
            emb_new = self.embed_model.encode(new_text if new_text else clean_orig).tolist()

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
                "gen_model": self.gen_model_name,
            })

        return pd.DataFrame(records)

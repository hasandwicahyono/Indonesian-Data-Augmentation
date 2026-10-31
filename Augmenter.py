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
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from IndonesianPreprocessor import IndonesianPreprocessor
from SimilarityMeasures import (
    euclidean_distance,
    cos_similarity,
    jaccard_similarity,
    calculate_bleu_scores,
    distance_to_similarity,
)


# =====================================================================
# Generative Models Registry: First-Generation (2020–2021) vs. Contemporary (2024)
# =====================================================================
GENERATIVE_MODELS_REGISTRY: Dict[str, Dict[str, Any]] = {
    # -----------------------------------------------------------------
    # First-Generation Indonesian Autoregressive Models (2020–2021)
    # -----------------------------------------------------------------
    "gpt2-small": {
        "model_id": "cahya/gpt2-small-indonesian-522M",
        "generation": "First-Generation (2020)",
        "release_year": 2020,
        "author": "Cahya Wirawan",
        "parameters": "117M",
        "architecture": "GPT-2 Decoder-Only",
        "corpus": "Indonesian Wikipedia (522 MB)",
        "default_params": {
            "num_beams": 5,
            "top_p": 0.95,
            "repetition_penalty": 2.5,
            "no_repeat_ngram_size": 3,
            "temperature": 1.0,
            "do_sample": False,
        },
        "description": "Baseline first-generation autoregressive model matching Refai et al. (2023) AraGPT2 baseline.",
    },
    "gpt2-flax": {
        "model_id": "flax-community/indonesian-gpt2",
        "generation": "First-Generation (2021)",
        "release_year": 2021,
        "author": "Hugging Face Community (JAX/Flax Sprint)",
        "parameters": "124M",
        "architecture": "GPT-2 Decoder-Only",
        "corpus": "Indonesian Wikipedia + OSCAR corpus",
        "default_params": {
            "num_beams": 5,
            "top_p": 0.95,
            "repetition_penalty": 2.5,
            "no_repeat_ngram_size": 3,
            "temperature": 1.0,
            "do_sample": False,
        },
        "description": "Direct peer architecture trained on TPUv3-8 during the 2021 Flax Community Sprint.",
    },
    "gpt2-medium": {
        "model_id": "cahya/gpt2-medium-indonesian-522M",
        "generation": "First-Generation (2020)",
        "release_year": 2020,
        "author": "Cahya Wirawan",
        "parameters": "345M",
        "architecture": "GPT-2 Decoder-Only",
        "corpus": "Indonesian Wikipedia (522 MB)",
        "default_params": {
            "num_beams": 4,
            "top_p": 0.95,
            "repetition_penalty": 2.2,
            "no_repeat_ngram_size": 3,
            "temperature": 1.0,
            "do_sample": False,
        },
        "description": "Scaled medium first-generation model to evaluate parameter capacity impact.",
    },

    # -----------------------------------------------------------------
    # Contemporary Foundation Generation Models (2024)
    # -----------------------------------------------------------------
    "sailor-0.5b": {
        "model_id": "sail/Sailor-0.5B",
        "generation": "Contemporary Foundation (2024)",
        "release_year": 2024,
        "author": "Sea AI Lab (Singapore)",
        "parameters": "490M",
        "architecture": "Modern Decoder (SwiGLU, RoPE, GQA)",
        "corpus": "Southeast Asian web crawls + Indo Wikipedia (~200B+ SEA tokens)",
        "default_params": {
            "num_beams": 1,
            "top_p": 0.90,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 3,
            "temperature": 0.7,
            "do_sample": True,
        },
        "description": "Specialized Southeast Asian foundation model explicitly optimized for Indonesian grammar and slang.",
    },
    "qwen2.5-0.5b": {
        "model_id": "Qwen/Qwen2.5-0.5B",
        "generation": "Contemporary Foundation (2024)",
        "release_year": 2024,
        "author": "Alibaba Cloud",
        "parameters": "490M",
        "architecture": "Modern Decoder (SwiGLU, RoPE, GQA)",
        "corpus": "18 Trillion multilingual tokens with extensive Indonesian web coverage",
        "default_params": {
            "num_beams": 1,
            "top_p": 0.90,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 3,
            "temperature": 0.7,
            "do_sample": True,
        },
        "description": "State-of-the-art compact multilingual model with high syntactic coherence and diverse vocabulary.",
    },
    "smollm2-135m": {
        "model_id": "HuggingFaceTB/SmolLM2-135M",
        "generation": "Contemporary Foundation (2024)",
        "release_year": 2024,
        "author": "Hugging Face",
        "parameters": "135M",
        "architecture": "Modern Compact Decoder (RoPE, GQA)",
        "corpus": "FineWeb-Edu & Cosmopedia (2T tokens)",
        "default_params": {
            "num_beams": 1,
            "top_p": 0.90,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "temperature": 0.7,
            "do_sample": True,
        },
        "description": "Exact parameter size counterpart to GPT-2 Small (135M vs 117M), isolating 2020 vs 2024 architectural progress.",
    },
    "seallms-1.5b": {
        "model_id": "SeaLLMs/SeaLLMs-v3-1.5B",
        "generation": "Contemporary Foundation (2024)",
        "release_year": 2024,
        "author": "Alibaba DAMO Academy",
        "parameters": "1.5B",
        "architecture": "Modern Decoder (GQA, RoPE)",
        "corpus": "Dedicated Southeast Asian instruction & pretraining corpora",
        "default_params": {
            "num_beams": 1,
            "top_p": 0.90,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 3,
            "temperature": 0.7,
            "do_sample": True,
        },
        "description": "Leading Southeast Asian foundation model with high naturalness in colloquial and formal Indonesian.",
    },
}


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
        # Resolve model name and metadata from registry if alias is provided
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

        # Load default parameters from metadata if available, otherwise fallback to sensible defaults
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

        # Resolve device
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

        self.preprocessor = IndonesianPreprocessor()
        self._init_models()

    @classmethod
    def list_candidate_models(cls) -> List[Dict[str, Any]]:
        """Returns metadata of all candidate generative models in the registry."""
        return list(GENERATIVE_MODELS_REGISTRY.values())

    @classmethod
    def display_model_comparisons(cls):
        """Prints a structured comparison table of First-Gen vs. Contemporary models."""
        print("=" * 105)
        print(f"{'Alias':<14} | {'Release':<7} | {'Generation':<26} | {'Params':<7} | {'Architecture':<22} | {'HuggingFace ID'}")
        print("-" * 105)
        for alias, info in GENERATIVE_MODELS_REGISTRY.items():
            print(f"{alias:<14} | {info['release_year']:<7} | {info['generation']:<26} | {info['parameters']:<7} | {info['architecture']:<22} | {info['model_id']}")
        print("=" * 105)

    def _init_models(self):
        """Initializes generator and embedder models."""
        gen_desc = f"{self.model_meta['generation']} ({self.model_meta['parameters']})" if self.model_meta else "Custom"
        print(f"Loading Generative Model: {self.gen_model_name} [{gen_desc}] on device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.gen_model_name, trust_remote_code=True)
            pad_id = self.tokenizer.eos_token_id or 0
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.gen_pipeline = pipeline(
                "text-generation",
                model=self.gen_model_name,
                tokenizer=self.tokenizer,
                device=self.device,
                pad_token_id=pad_id,
                trust_remote_code=True,
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

        gen_model_tag = self.model_meta["generation"] if self.model_meta else self.gen_model_name
        print(f"Starting generation [{gen_model_tag}] & similarity computation for {len(target_df)} rows...")
        
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
                "gen_model": self.gen_model_name,
            })

        return pd.DataFrame(records)

"""
Generative Backbone Models Registry for Indonesian Text Augmentation.
Contains metadata, configurations, and default hyperparameters for First-Generation (2020-2021)
and Contemporary Foundation (2024) models.
"""

from typing import Dict, Any, List

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
        "model_id": "flax-community/gpt2-small-indonesian",
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
        "model_id": "cahya/gpt2-medium-indonesian",
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


def list_candidate_models() -> List[Dict[str, Any]]:
    """Returns metadata of all candidate generative models in the registry."""
    return list(GENERATIVE_MODELS_REGISTRY.values())


def display_model_comparisons() -> None:
    """Prints a structured comparison table of First-Gen vs. Contemporary models."""
    print("=" * 105)
    print(f"{'Alias':<14} | {'Release':<7} | {'Generation':<26} | {'Params':<7} | {'Architecture':<22} | {'HuggingFace ID'}")
    print("-" * 105)
    for alias, info in GENERATIVE_MODELS_REGISTRY.items():
        print(f"{alias:<14} | {info['release_year']:<7} | {info['generation']:<26} | {info['parameters']:<7} | {info['architecture']:<22} | {info['model_id']}")
    print("=" * 105)

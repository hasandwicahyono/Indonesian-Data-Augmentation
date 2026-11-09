"""
Generative Backbone Models Benchmark and Comparison Suite.
Evaluates First-Generation (GPT-2 Small, GPT-2 Flax) vs. Contemporary (SmolLM2-135M, Qwen2.5-0.5B)
generative decoders on Indonesian text augmentation.

Measures:
1. Latency (ms/sample) and Throughput (tokens/s)
2. Lexical Diversity (Distinct-1, Distinct-2, Shannon Entropy)
3. Semantic Fidelity to Seed Prompt (Cosine Similarity, Jaccard Index, BLEU-2)
4. Model Parameters and Release Timeline

Exports:
- data/results/extended/generative_models_comparison.csv
- paper/table_generative_models_comparison.tex
"""

import os
import sys
import time
import math
import collections
import warnings
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from src.utils.path_utils import resolve_path, ensure_dir
from src.utils.hardware import resolve_device
from src.utils.metrics import compute_distinct_n, compute_entropy
from src.utils.config import load_benchmark_config
from src.augmentation.similarity import (
    cos_similarity,
    jaccard_similarity,
    calculate_bleu_scores,
)
from src.augmentation.registry import GENERATIVE_MODELS_REGISTRY
from src.data.preprocessor import IndonesianPreprocessor


def _get_default_prompts() -> List[str]:
    try:
        cfg = load_benchmark_config()
        return cfg.get("generative_benchmark", {}).get("prompts", [])
    except Exception:
        return []

BENCHMARK_PROMPTS = _get_default_prompts()


def compute_distinct_ngrams(texts: List[str]) -> Tuple[float, float, float]:
    """Computes Distinct-1, Distinct-2, and Shannon Entropy across generated texts."""
    d1 = compute_distinct_n(texts, n=1)
    d2 = compute_distinct_n(texts, n=2)
    entropy = compute_entropy(texts)
    return float(round(d1, 4)), float(round(d2, 4)), float(round(entropy, 4))


def _export_latex_table(df: pd.DataFrame, output_tex: str):
    """Exports dynamic LaTeX table comparing generative models."""
    latex_rows = []
    for _, r in df.iterrows():
        name = str(r["Model Key"])
        params = str(r["Parameters"])
        era = str(r["Generation Era"]).split("(")[0].strip()
        latency = f"{r['Latency (ms)']:.1f}"
        throughput = f"{r['Throughput (tok/s)']:.1f}"
        d1 = f"{r['Distinct-1']:.3f}"
        d2 = f"{r['Distinct-2']:.3f}"
        cos = f"{r['Cosine Sim']:.3f}"
        bleu = f"{r['BLEU-2']:.3f}"

        latex_rows.append(
            f"\\textbf{{{name}}} & {params} & {era} & {latency} & {throughput} & {d1} & {d2} & {cos} & {bleu} \\\\"
        )

    tex_content = """\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{l c l c c c c c c}
\\toprule
\\textbf{Generative Model} & \\textbf{Params} & \\textbf{Architectural Era} & \\textbf{Latency (ms)} & \\textbf{Throughput (tok/s)} & \\textbf{Distinct-1} & \\textbf{Distinct-2} & \\textbf{Cosine Sim} & \\textbf{BLEU-2} \\\\
\\midrule
""" + "\n".join(latex_rows) + """
\\bottomrule
\\end{tabular}%
}"""

    ensure_dir(os.path.dirname(output_tex))
    with open(output_tex, "w") as f:
        f.write(tex_content)
    print(f"[Saved] Generative models comparison LaTeX table saved to: {output_tex}")


def benchmark_candidate_generative_models(
    models: Optional[List[str]] = None,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 32,
    device: Optional[str] = None,
    output_csv: str = "data/results/extended/generative_models_comparison.csv",
    output_tex: str = "paper/table_generative_models_comparison.tex",
    force: bool = False,
) -> pd.DataFrame:
    """
    Executes and benchmarks generative models across throughput, latency, lexical diversity,
    and semantic similarity.
    """
    csv_path = resolve_path(output_csv)
    tex_path = resolve_path(output_tex)

    if os.path.exists(csv_path) and not force:
        print(f"[Generative Benchmark] Loading existing comparison results from: {csv_path}")
        df = pd.read_csv(csv_path)
        _export_latex_table(df, tex_path)
        return df

    if models is None:
        models = [
            "gpt2-small",
            "gpt2-flax",
            "smollm2-135m",
            "sailor-0.5b",
            "qwen2.5-0.5b",
            "seallms-v3-1.5b",
        ]

    if prompts is None:
        prompts = BENCHMARK_PROMPTS

    device_str = device if device is not None else resolve_device()

    print("=" * 85)
    print("BENCHMARKING GENERATIVE BACKBONE MODELS FOR INDONESIAN DATA AUGMENTATION")
    print(f"Device: {device_str} | Candidate Models: {models} | Evaluated Prompts: {len(prompts)}")
    print("=" * 85)

    print("Loading Sentence Embedding Model for Semantic Fidelity Evaluation...")
    embed_model = SentenceTransformer(
        "sentence-transformers/distiluse-base-multilingual-cased-v2",
        device=device_str,
    )
    prompt_embeddings = embed_model.encode(prompts, convert_to_numpy=True)

    records = []
    preprocessor = IndonesianPreprocessor()

    for model_key in models:
        meta = GENERATIVE_MODELS_REGISTRY.get(model_key, None)
        model_id = meta["model_id"] if meta else model_key
        params = meta["parameters"] if meta else "N/A"
        release = meta["release_year"] if meta else "N/A"
        generation_era = meta["generation"] if meta else "Custom"
        arch = meta["architecture"] if meta else "Decoder-Only"

        print(f"\nEvaluating Model: {model_key} [{model_id}] ({params}, {generation_era})...")

        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token or tok.unk_token or ""

            torch_dtype = torch.float16 if device_str.startswith("cuda") else torch.float32
            mod = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            active_device = device_str
            try:
                mod.to(active_device)
                mod.eval()
                dummy_ids = tok.encode("Uji coba", return_tensors="pt").to(active_device)
                with torch.no_grad():
                    mod.generate(dummy_ids, max_new_tokens=4, do_sample=False)
            except Exception as dev_err:
                print(f"  [Device fallback] Error on {device_str} ({dev_err}). Falling back to CPU...")
                active_device = "cpu"
                mod.to(active_device)
                mod.eval()

            generated_texts = []
            new_tokens_count = 0
            latencies = []

            for p in prompts:
                clean_p = preprocessor.preprocess(p)
                input_ids = tok.encode(clean_p, return_tensors="pt").to(active_device)

                t0 = time.perf_counter()
                with torch.no_grad():
                    gen_ids = mod.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tok.pad_token_id,
                    )
                t1 = time.perf_counter()

                latency = (t1 - t0) * 1000.0  # ms
                latencies.append(latency)

                new_ids = gen_ids[0][input_ids.shape[1]:]
                new_tokens_count += len(new_ids)
                gen_text = tok.decode(new_ids, skip_special_tokens=True).strip()
                generated_texts.append(gen_text if gen_text else clean_p)

            # Performance Metrics
            avg_latency = float(round(np.mean(latencies), 2))
            total_time_s = sum(latencies) / 1000.0
            throughput = float(round(new_tokens_count / max(0.001, total_time_s), 2))

            # Diversity Metrics
            d1, d2, entropy = compute_distinct_ngrams(generated_texts)

            # Semantic Fidelity Metrics
            gen_embeddings = embed_model.encode(generated_texts, convert_to_numpy=True)
            cos_sims = [cos_similarity(prompt_embeddings[i], gen_embeddings[i]) for i in range(len(prompts))]
            jacc_sims = [jaccard_similarity(prompts[i], generated_texts[i]) for i in range(len(prompts))]
            bleu_scores = [calculate_bleu_scores(prompts[i], generated_texts[i])[1] for i in range(len(prompts))]

            avg_cos = float(round(np.mean(cos_sims), 4))
            avg_jacc = float(round(np.mean(jacc_sims), 4))
            avg_bleu2 = float(round(np.mean(bleu_scores), 4))

            rec = {
                "Model Key": model_key,
                "Model ID": model_id,
                "Generation Era": generation_era,
                "Release": release,
                "Parameters": params,
                "Architecture": arch,
                "Latency (ms)": avg_latency,
                "Throughput (tok/s)": throughput,
                "Distinct-1": d1,
                "Distinct-2": d2,
                "Entropy": entropy,
                "Cosine Sim": avg_cos,
                "Jaccard Sim": avg_jacc,
                "BLEU-2": avg_bleu2,
            }
            records.append(rec)
            print(f"  -> Latency: {avg_latency} ms | Throughput: {throughput} tok/s | Distinct-2: {d2} | Cosine: {avg_cos}")

        except Exception as e:
            print(f"  [Warning] Failed to benchmark {model_key}: {e}")

    df = pd.DataFrame(records)
    ensure_dir(os.path.dirname(csv_path))
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved] Generative models benchmark CSV saved to: {csv_path}")

    _export_latex_table(df, tex_path)
    return df


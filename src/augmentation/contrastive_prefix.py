"""
Contrastive Aspect Prefix Generation Module.
Steers generative language models using structured aspect-polarity prefixes
and contrastive log-likelihood ratio verification to eliminate cross-valence
semantic drift, hallucination, and aspect leakage.

Mathematical Formulation:
Given an input sample (x, y) with detected or specified aspect a in A:
1. Target Prefix Conditioning:
     P_{target}(a, y) = "[Aspek: a | Sentimen: LabelName(y)] "
     Prompt_{target}  = P_{target}(a, y) + x

2. Contrastive (Polarity-Opposite) Prefix:
     P_{contrast}(a, y) = "[Aspek: a | Sentimen: OppositeLabel(y)] "
     Prompt_{contrast}  = P_{contrast}(a, y) + x

3. Contrastive Likelihood Ratio Scoring:
   For candidate continuation c:
     S_{target}   = P_{target}   + x + " " + c
     S_{contrast} = P_{contrast} + x + " " + c
     Delta_{contrast}(c) = (1 / |c|) * [ log P_LM(c | P_{target}, x) - log P_LM(c | P_{contrast}, x) ]

   Continuations with Delta_{contrast}(c) >= tau_{contrast} (e.g. 0.0) strongly embody
   the target aspect and sentiment while rejecting cross-valence sign-flip errors.
"""

import re
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union


# =====================================================================
# Domain-Specific Aspect Dictionaries
# =====================================================================
HOTEL_ASPECT_TAXONOMY: Dict[str, List[str]] = {
    "kamar": [
        "kamar", "bed", "ranjang", "ac", "sprei", "bantal", "pintu", "jendela",
        "ruangan", "bocor", "sempit", "luas", "kasur", "selimut", "lemari"
    ],
    "kebersihan": [
        "kotor", "bersih", "bau", "jorok", "debu", "sampah", "kecoak", "wangi",
        "kamar mandi", "toilet", "wc", "seprai", "handuk", "berdebu", "kumal"
    ],
    "pelayanan": [
        "staf", "pelayanan", "resepsionis", "ramah", "jutek", "lambat", "cepat",
        "satpam", "pelayan", "check-in", "checkout", "waiter", "service", "bantuan"
    ],
    "fasilitas": [
        "wifi", "kolam", "sarapan", "makan", "parkir", "lift", "resto", "gym",
        "tv", "air", "panas", "dingin", "restoran", "makanan", "fitness", "menu"
    ],
    "lokasi": [
        "lokasi", "strategis", "dekat", "jauh", "macet", "pusat", "kota", "akses",
        "pantai", "stasiun", "bandara", "jalan", "suasana", "pemandangan"
    ],
    "harga": [
        "harga", "mahal", "murah", "worth", "biaya", "promo", "diskon", "tarif",
        "bayar", "sebanding", "kantong", "budget"
    ],
}

EMOTION_THEME_TAXONOMY: Dict[str, List[str]] = {
    "sosial_relasi": [
        "teman", "sahabat", "orang", "orang-orang", "netizen", "publik", "sosial",
        "tetangga", "rekan", "kawan"
    ],
    "pekerjaan_studi": [
        "kerja", "kantor", "kuliah", "tugas", "sekolah", "gaji", "bos", "ujian",
        "skripsi", "proyek", "karier", "dosen", "guru"
    ],
    "keluarga_cinta": [
        "keluarga", "ibu", "ayah", "pasangan", "pacar", "suami", "istri", "cinta",
        "sayang", "rindu", "kangen", "anak", "orang tua"
    ],
    "kesehatan_diri": [
        "sakit", "lelah", "capek", "sehat", "tidur", "stres", "pusing", "hati",
        "jiwa", "mental", "semangat", "istirahat"
    ],
}

SENTIMENT_ASPECT_TAXONOMY: Dict[str, List[str]] = {
    "kualitas": [
        "kualitas", "bagus", "jelek", "rusak", "awet", "cacat", "mutu", "original",
        "asli", "palsu", "kondisi", "material", "fungsi"
    ],
    "pelayanan": [
        "pelayanan", "staf", "ramah", "cs", "admin", "respon", "penjual", "seller",
        "tanggapan", "cepat", "lambat", "komplain"
    ],
    "pengiriman": [
        "kirim", "pengiriman", "kurir", "cepat", "lambat", "packing", "paket", "rapi",
        "rusak", "sampai", "ekspedisi", "aman"
    ],
    "harga_nilai": [
        "harga", "murah", "mahal", "biaya", "ongkir", "diskon", "promo", "worth",
        "ekonomis", "terjangkau"
    ],
}


def detect_aspect_from_text(
    text: str,
    domain: str = "hotel",
    custom_taxonomy: Optional[Dict[str, List[str]]] = None,
) -> str:
    """
    Detects the most salient aspect mentioned in the text using domain keyword matching.
    Falls back to 'umum' if no specific aspect keyword matches.
    """
    if custom_taxonomy:
        taxonomy = custom_taxonomy
    else:
        d_lower = domain.lower()
        if "hotel" in d_lower or "hoasa" in d_lower:
            taxonomy = HOTEL_ASPECT_TAXONOMY
        elif "emotion" in d_lower or "emot" in d_lower:
            taxonomy = EMOTION_THEME_TAXONOMY
        else:
            taxonomy = SENTIMENT_ASPECT_TAXONOMY

    clean = text.lower()
    aspect_scores: Dict[str, int] = {}

    for aspect_name, keywords in taxonomy.items():
        score = 0
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", clean):
                score += 1
            elif kw in clean:
                score += 0.5
        if score > 0:
            aspect_scores[aspect_name] = score

    if not aspect_scores:
        return "umum"

    # Return aspect with highest occurrence
    return max(aspect_scores.items(), key=lambda x: x[1])[0]


def format_contrastive_prefixes(
    aspect: str,
    label: str,
    domain: str = "hotel",
) -> Tuple[str, str]:
    """
    Builds the target-conditioned prefix and its polar-opposite contrastive prefix.

    Returns:
        (target_prefix, contrastive_prefix)
    """
    lbl = str(label).strip().lower()
    d_lower = domain.lower()

    if "hotel" in d_lower or "hoasa" in d_lower:
        if lbl in ("neg", "negative", "0"):
            target_desc = "Negatif / Keluhan"
            contrast_desc = "Positif / Pujian"
        elif lbl in ("pos", "positive", "2", "1"):
            target_desc = "Positif / Pujian"
            contrast_desc = "Negatif / Keluhan"
        else:  # neut / neutral
            target_desc = "Netral / Informatif"
            contrast_desc = "Polar / Ekstrem"
        target_prefix = f"[Aspek: {aspect} | Sentimen: {target_desc}] "
        contrast_prefix = f"[Aspek: {aspect} | Sentimen: {contrast_desc}] "

    elif "emotion" in d_lower or "emot" in d_lower:
        positive_emotions = {"happy", "love"}
        negative_emotions = {"anger", "fear", "sadness"}

        if lbl in positive_emotions:
            target_desc = f"{lbl.capitalize()} (Valensi Positif)"
            contrast_desc = "Marah / Sedih (Valensi Negatif)"
        elif lbl in negative_emotions:
            target_desc = f"{lbl.capitalize()} (Valensi Negatif)"
            contrast_desc = "Bahagia / Cinta (Valensi Positif)"
        else:
            target_desc = f"{lbl.capitalize()}"
            contrast_desc = "Netral / Kontras"
        target_prefix = f"[Tema: {aspect} | Emosi: {target_desc}] "
        contrast_prefix = f"[Tema: {aspect} | Emosi: {contrast_desc}] "

    else:  # General sentiment (SmSA, NusaX, MovieOpini)
        if lbl in ("positive", "pos", "1"):
            target_desc = "Positif / Rekomendasi"
            contrast_desc = "Negatif / Kekecewaan"
        elif lbl in ("negative", "neg", "0"):
            target_desc = "Negatif / Kekecewaan"
            contrast_desc = "Positif / Rekomendasi"
        else:
            target_desc = "Netral / Objektif"
            contrast_desc = "Subjektif / Emosional"
        target_prefix = f"[Aspek: {aspect} | Sentimen: {target_desc}] "
        contrast_prefix = f"[Aspek: {aspect} | Sentimen: {contrast_desc}] "

    return target_prefix, contrast_prefix


def compute_continuation_log_prob(
    model: Any,
    tokenizer: Any,
    prefix_text: str,
    continuation_text: str,
    device: Optional[Any] = None,
) -> float:
    """
    Computes the conditional log-probability of continuation_text given prefix_text under causal LM:
      sum_{t in continuation} log P_LM(c_t | prefix, c_{<t})
    """
    if not continuation_text.strip():
        return 0.0

    full_text = prefix_text.rstrip() + " " + continuation_text.lstrip()

    # Tokenize full sequence and prefix
    enc_full = tokenizer(full_text, return_tensors="pt")
    enc_prefix = tokenizer(prefix_text.rstrip(), return_tensors="pt")

    input_ids = enc_full["input_ids"]
    full_len = input_ids.shape[1]
    prefix_len = enc_prefix["input_ids"].shape[1]

    if full_len <= prefix_len:
        return 0.0

    if device is not None:
        input_ids = input_ids.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, full_len, vocab_size)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Per-token cross-entropy
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss_per_token = loss_fct(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1)
    ).view(1, full_len - 1)

    # Slice continuation tokens: from (prefix_len - 1) to (full_len - 1)
    cont_loss = loss_per_token[:, prefix_len - 1:].sum(dim=-1).item()
    return -float(cont_loss)


def compute_contrastive_margin(
    model: Any,
    tokenizer: Any,
    target_prefix: str,
    contrast_prefix: str,
    base_text: str,
    continuation: str,
    device: Optional[Any] = None,
) -> float:
    """
    Computes the contrastive margin score:
      Delta = (log P(c | P_{target}, base) - log P(c | P_{contrast}, base)) / |tokens(c)|
    A positive Delta indicates the continuation statistically aligns with the target aspect/polarity
    rather than the contrastive opposite polarity.
    """
    if not continuation.strip():
        return 0.0

    p_target_full = target_prefix + base_text
    p_contrast_full = contrast_prefix + base_text

    log_p_target = compute_continuation_log_prob(model, tokenizer, p_target_full, continuation, device=device)
    log_p_contrast = compute_continuation_log_prob(model, tokenizer, p_contrast_full, continuation, device=device)

    # Token length of continuation for length-normalization
    cont_tokens = tokenizer.encode(continuation.strip(), add_special_tokens=False)
    n_tokens = max(1, len(cont_tokens))

    margin = (log_p_target - log_p_contrast) / float(n_tokens)
    return float(margin)


class ContrastiveAspectPrefixGenerator:
    """
    Aspect-conditioned and polarity-steered contrastive text generator.
    Enforces semantic aspect fidelity and guards against cross-valence drift.
    """

    def __init__(
        self,
        domain: str = "hotel",
        custom_taxonomy: Optional[Dict[str, List[str]]] = None,
        margin_threshold: float = 0.0,
        strip_prefix_in_output: bool = True,
    ):
        self.domain = domain
        self.custom_taxonomy = custom_taxonomy
        self.margin_threshold = float(margin_threshold)
        self.strip_prefix_in_output = bool(strip_prefix_in_output)

    def generate(
        self,
        gen_pipeline: Any,
        text: str,
        label: str,
        max_length: int = 128,
        num_candidates: int = 3,
        gen_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generates continuation steered by target aspect prefix and verified by contrastive margin.

        Returns:
            (augmented_full_text, metadata_dict)
        """
        # 1. Detect Aspect
        aspect = detect_aspect_from_text(text, domain=self.domain, custom_taxonomy=self.custom_taxonomy)

        # 2. Build Target and Contrastive Prefixes
        target_prefix, contrast_prefix = format_contrastive_prefixes(aspect, label, domain=self.domain)

        # 3. Construct Steered Prompt
        steered_prompt = target_prefix + text[:max_length]

        kwargs = gen_kwargs.copy() if gen_kwargs else {}
        max_new_tokens = kwargs.get("max_new_tokens", 40)
        repetition_penalty = kwargs.get("repetition_penalty", 2.5)
        no_repeat_ngram_size = kwargs.get("no_repeat_ngram_size", 3)

        pipeline_kwargs = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "num_return_sequences": max(1, num_candidates),
            "do_sample": True,
            "top_p": kwargs.get("top_p", 0.92),
            "temperature": kwargs.get("temperature", 0.85),
        }

        # Model and tokenizer handles for contrastive scoring
        model = getattr(gen_pipeline, "model", None)
        tokenizer = getattr(gen_pipeline, "tokenizer", None)
        device = getattr(model, "device", None)

        try:
            outputs = gen_pipeline(steered_prompt, **pipeline_kwargs)
        except Exception as e:
            # Fallback to single generation if batch sequences fail
            pipeline_kwargs["num_return_sequences"] = 1
            outputs = gen_pipeline(steered_prompt, **pipeline_kwargs)

        best_continuation = ""
        best_margin = -float("inf")
        selected_full = ""

        candidates = []
        for out in outputs:
            full_gen = out["generated_text"]
            if len(full_gen) > len(steered_prompt):
                suffix = full_gen[len(steered_prompt):].strip()
                last_punct = max(suffix.rfind("."), suffix.rfind("!"), suffix.rfind("?"))
                if last_punct > 5:
                    suffix = suffix[:last_punct + 1].strip()
            else:
                suffix = ""

            candidates.append(suffix)

        # Contrastive Scoring across candidates
        if model is not None and tokenizer is not None and len(candidates) > 1:
            for cand in candidates:
                if not cand.strip():
                    continue
                margin = compute_contrastive_margin(
                    model=model,
                    tokenizer=tokenizer,
                    target_prefix=target_prefix,
                    contrast_prefix=contrast_prefix,
                    base_text=text[:max_length],
                    continuation=cand,
                    device=device,
                )
                if margin > best_margin:
                    best_margin = margin
                    best_continuation = cand
        elif candidates:
            best_continuation = candidates[0]
            best_margin = 0.0

        # Construct final augmented text
        if self.strip_prefix_in_output:
            if best_continuation:
                augmented_text = text.strip() + " " + best_continuation.strip()
            else:
                augmented_text = text.strip()
        else:
            if best_continuation:
                augmented_text = target_prefix + text.strip() + " " + best_continuation.strip()
            else:
                augmented_text = target_prefix + text.strip()

        meta = {
            "aspect": aspect,
            "target_prefix": target_prefix,
            "contrast_prefix": contrast_prefix,
            "contrastive_margin": best_margin,
            "is_contrastive_valid": best_margin >= self.margin_threshold,
            "num_candidates_evaluated": len(candidates),
        }

        return augmented_text.strip(), meta

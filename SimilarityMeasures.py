"""
Similarity Measures for Filtering Generated Text.
Implements the 4 metrics evaluated in Refai et al. (2023):
1. Euclidean Distance / Similarity
2. Cosine Similarity
3. Jaccard Similarity
4. BLEU-1 to BLEU-4 Scores
"""

import math
import numpy as np
from typing import List, Tuple, Union
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction


def euclidean_distance(x: List[float], y: List[float]) -> float:
    """Calculates Euclidean distance between two embedding vectors."""
    if len(x) == 0 or len(y) == 0:
        return 0.0
    return float(math.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y))))


def distance_to_similarity(distance: float) -> float:
    """Converts a distance metric into an exponential similarity score [0, 1]."""
    try:
        return float(1.0 / math.exp(distance))
    except OverflowError:
        return 0.0


def squared_sum(x: List[float]) -> float:
    """Returns the square root of the sum of squares."""
    return math.sqrt(sum(a * a for a in x))


def cos_similarity(x: List[float], y: List[float]) -> float:
    """Calculates cosine similarity between two embedding vectors."""
    if len(x) == 0 or len(y) == 0:
        return 0.0
    denom = squared_sum(x) * squared_sum(y)
    if denom == 0:
        return 0.0
    numerator = sum(a * b for a, b in zip(x, y))
    return float(round(numerator / float(denom), 4))


def jaccard_similarity(text1: Union[str, List[str]], text2: Union[str, List[str]]) -> float:
    """
    Calculates token-level Jaccard similarity between two texts or token lists.
    |Intersection(A, B)| / |Union(A, B)|
    """
    if isinstance(text1, str):
        set1 = set(text1.split())
    else:
        set1 = set(text1)

    if isinstance(text2, str):
        set2 = set(text2.split())
    else:
        set2 = set(text2)

    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(round(intersection / float(union), 4))


def calculate_bleu_scores(
    reference: Union[str, List[str]], hypothesis: Union[str, List[str]]
) -> Tuple[float, float, float, float]:
    """
    Calculates BLEU 1-4 scores based on NLTK sentence_bleu.

    Args:
        reference: Reference text (original sentence)
        hypothesis: Generated hypothesis sentence

    Returns:
        (bleu_1, bleu_2, bleu_3, bleu_4)
    """
    if isinstance(reference, str):
        ref_tokens = [reference.split()]
    elif isinstance(reference, list) and (len(reference) == 0 or isinstance(reference[0], str)):
        ref_tokens = [reference]
    else:
        ref_tokens = reference

    if isinstance(hypothesis, str):
        hyp_tokens = hypothesis.split()
    else:
        hyp_tokens = hypothesis

    if len(hyp_tokens) == 0:
        return 0.0, 0.0, 0.0, 0.0

    smoothie = SmoothingFunction().method1

    try:
        b1 = sentence_bleu(ref_tokens, hyp_tokens, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smoothie)
        b2 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.50, 0.50, 0.0, 0.0), smoothing_function=smoothie)
        b3 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.34, 0.33, 0.33, 0.0), smoothing_function=smoothie)
        b4 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        return float(round(b1, 4)), float(round(b2, 4)), float(round(b3, 4)), float(round(b4, 4))
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

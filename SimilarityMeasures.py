"""
SimilarityMeasures: backward-compatible entry point delegating to src.augmentation.similarity.
"""

from src.augmentation.similarity import (
    euclidean_distance,
    cos_similarity,
    jaccard_similarity,
    calculate_bleu_scores,
    distance_to_similarity,
)

__all__ = [
    "euclidean_distance",
    "cos_similarity",
    "jaccard_similarity",
    "calculate_bleu_scores",
    "distance_to_similarity",
]

if __name__ == "__main__":
    t1 = "Pelayanan di hotel sangat ramah dan bagus"
    t2 = "Staf hotel sangat ramah dan pelayanan menyenangkan"
    print(f"Jaccard Similarity: {jaccard_similarity(t1, t2):.4f}")
    print(f"BLEU Scores: {calculate_bleu_scores(t1, t2)}")

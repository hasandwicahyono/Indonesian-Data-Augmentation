"""
ClassificationDataset: backward-compatible entry point delegating to src.data.classification.
"""

from src.data.classification import ClassificationDataset

__all__ = ["ClassificationDataset"]

if __name__ == "__main__":
    ds = ClassificationDataset(
        text=["Pelayanan sangat baik", "Mengecewakan sekali"],
        target=["positive", "negative"],
        model_name="indobenchmark/indobert-base-p1",
        max_len=32,
        label_map={"positive": 0, "negative": 1},
    )
    print(f"ClassificationDataset initialized with {len(ds)} samples.")

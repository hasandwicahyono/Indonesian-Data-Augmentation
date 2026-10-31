"""
CLI script to run Phase 1: Text Generation & Similarity Scoring on Indonesian Datasets.
Supports: SmSA, EmoT, HoASA, MovieOpini.
"""

import os
import argparse
from DatasetsManager import IndonesianDatasetsManager
from Augmenter import IndonesianAugmenter


def main():
    parser = argparse.ArgumentParser(description="Generate augmentations and similarity scores for Indonesian datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="SmSA",
        choices=["SmSA", "EmoT", "HoASA", "MovieOpini"],
        help="Select Indonesian dataset to augment",
    )
    parser.add_argument(
        "--gen_model",
        type=str,
        default="cahya/gpt2-small-indonesian-522M",
        help="Pretrained Indonesian generative model",
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default="sentence-transformers/distiluse-base-multilingual-cased-v2",
        help="Pretrained sentence embedding model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Output directory for intermediate generated data",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Optional limit on number of samples to process (for testing/quick runs)",
    )
    parser.add_argument(
        "--target_labels",
        nargs="+",
        default=None,
        help="Optional minority classes to focus augmentation on (e.g. fear love, neg pos)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    manager = IndonesianDatasetsManager()

    print(f"Loading Indonesian dataset: {args.dataset}")
    if args.dataset == "SmSA":
        ds = manager.load_smsa()
    elif args.dataset == "EmoT":
        ds = manager.load_emot()
    elif args.dataset == "HoASA":
        ds = manager.load_hoasa()
    elif args.dataset == "MovieOpini":
        ds = manager.load_movie_opini()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Loaded {ds.name}: Train rows={len(ds.train)}, Labels={ds.label_list}")
    augmenter = IndonesianAugmenter(
        gen_model_name=args.gen_model,
        embed_model_name=args.embed_model,
    )

    out_df = augmenter.process_dataset(
        ds.train,
        text_column="text",
        label_column="label",
        sample_limit=args.sample_limit,
        target_labels=args.target_labels,
    )

    out_file = os.path.join(args.output_dir, f"{args.dataset}-Augmented-raw.csv")
    out_df.to_csv(out_file, index=False)
    print(f"Phase 1 complete! Saved generated and similarity-scored data to: {out_file}")


if __name__ == "__main__":
    main()

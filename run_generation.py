"""
CLI script to run Phase 1: Text Generation & Similarity Scoring on Indonesian Datasets.
Supports both First-Generation (2020–2021) and Contemporary Foundation (2024) models:
- First-Gen: gpt2-small, gpt2-flax, gpt2-medium
- Contemporary: sailor-0.5b, qwen2.5-0.5b, smollm2-135m, seallms-1.5b
"""

import os
import argparse
from DatasetsManager import IndonesianDatasetsManager
from Augmenter import IndonesianAugmenter, GENERATIVE_MODELS_REGISTRY


def main():
    parser = argparse.ArgumentParser(
        description="Generate augmentations and similarity scores for Indonesian datasets across generative models."
    )
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
        help="HuggingFace model ID or preset alias: gpt2-small, gpt2-flax, gpt2-medium, sailor-0.5b, qwen2.5-0.5b, smollm2-135m, seallms-1.5b",
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
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="Display comparison table of all candidate generative models and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        print("\nCandidate Generative Models for Indonesian Text Augmentation:")
        IndonesianAugmenter.display_model_comparisons()
        return

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

    # Sanitize model tag for file naming
    model_tag = args.gen_model.split("/")[-1].replace("-indonesian-522M", "").replace("-indonesian", "")
    out_file = os.path.join(args.output_dir, f"{args.dataset}-Augmented-{model_tag}-raw.csv")
    out_df.to_csv(out_file, index=False)
    print(f"\nPhase 1 complete! Saved generated and similarity-scored data to: {out_file}")


if __name__ == "__main__":
    main()

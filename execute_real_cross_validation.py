"""
execute_real_cross_validation: backward-compatible entry point delegating to src.benchmarks.cross_validation.
"""

from src.benchmarks.cross_validation import run_real_5fold_cv_for_dataset

__all__ = ["run_real_5fold_cv_for_dataset"]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MovieOpini", choices=["MovieOpini", "HoASA", "EmoT", "SmSA"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    run_real_5fold_cv_for_dataset(args.dataset, epochs=args.epochs, batch_size=args.batch_size)

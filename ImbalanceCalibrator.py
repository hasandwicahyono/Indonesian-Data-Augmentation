"""
ImbalanceCalibrator: backward-compatible entry point delegating to src.modeling.calibrator.
"""

from src.modeling.calibrator import AutoImbalanceCalibrator

__all__ = ["AutoImbalanceCalibrator"]

if __name__ == "__main__":
    import pandas as pd
    dummy_df = pd.DataFrame({"label": ["pos"] * 100 + ["neg"] * 5})
    cal = AutoImbalanceCalibrator.calibrate(dummy_df, label_list=["pos", "neg"])
    print(f"Calibration results: {cal}")

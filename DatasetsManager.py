"""
DatasetsManager: backward-compatible entry point delegating to src.data.dataset_manager.
"""

from src.data.dataset_manager import CustomDataset, IndonesianDatasetsManager

__all__ = ["CustomDataset", "IndonesianDatasetsManager"]

if __name__ == "__main__":
    manager = IndonesianDatasetsManager()
    datasets = manager.load_all_datasets()
    print("Available datasets loaded successfully:")
    for name, ds in datasets.items():
        print(f"  - {name}: Train={len(ds.train)}, Val={len(ds.val) if ds.val is not None else 0}, Test={len(ds.test)}, Classes={ds.label_list}")

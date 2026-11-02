"""
Path and Environment Resolution Utilities for Indonesian Data Augmentation.
Ensures consistent resolution of file paths across varied working directories.
"""

import os
import sys

# Compute project root (directory containing src/ or Indonesian-Data-Augmentation)
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "..", ".."))

# Ensure project root and src are on sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def get_project_root() -> str:
    """Returns the absolute path to the project root directory."""
    return PROJECT_ROOT


def resolve_path(path: str) -> str:
    """
    Resolves relative paths gracefully across execution directories (repo root, notebooks/, subdirectories).
    Checks inside PROJECT_ROOT first, normalizes leading '../' or './', then checks cwd, then absolute.
    """
    if not path:
        return PROJECT_ROOT
    if os.path.isabs(path):
        return os.path.abspath(path)

    # 1. Direct join with PROJECT_ROOT
    cand1 = os.path.abspath(os.path.join(PROJECT_ROOT, path))
    if os.path.exists(cand1):
        return cand1

    # 2. Strip leading '../' or './' if executed from a subfolder like notebooks/
    clean = path
    while clean.startswith("../") or clean.startswith("./"):
        if clean.startswith("../"):
            clean = clean[3:]
        elif clean.startswith("./"):
            clean = clean[2:]

    cand_clean = os.path.abspath(os.path.join(PROJECT_ROOT, clean))
    if os.path.exists(cand_clean):
        return cand_clean

    # 3. Check cwd
    if os.path.exists(path):
        return os.path.abspath(path)

    # 4. Target path for write operations: if points to standard project directories, anchor to PROJECT_ROOT
    first_part = clean.split(os.sep)[0]
    if first_part in {"data", "configs", "notebooks", "paper", "src", "results", "figures"}:
        return cand_clean

    return cand1


def ensure_dir(dir_path: str) -> str:
    """Ensures a directory exists, creating parent directories if needed."""
    resolved = resolve_path(dir_path)
    os.makedirs(resolved, exist_ok=True)
    return resolved


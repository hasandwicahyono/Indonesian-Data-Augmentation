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
    Resolves relative paths gracefully across execution directories.
    Checks inside PROJECT_ROOT first, then cwd, then absolute.
    """
    if os.path.isabs(path):
        return path
    p_root = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(p_root):
        return os.path.abspath(p_root)
    if os.path.exists(path):
        return os.path.abspath(path)
    return p_root


def ensure_dir(dir_path: str) -> str:
    """Ensures a directory exists, creating parent directories if needed."""
    resolved = resolve_path(dir_path)
    os.makedirs(resolved, exist_ok=True)
    return resolved


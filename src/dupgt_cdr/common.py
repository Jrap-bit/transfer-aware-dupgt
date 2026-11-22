"""File helpers. Relative paths start from this project, not the shell folder."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


# Resolve paths from the project folder.
def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


# Save JSON without leaving a half-written file.
def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


# Hash files to check data integrity.
def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def config_sha256(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


# Save compact NumPy arrays.
def save_arrays(path: Path, **arrays) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, path)


DATA_KEYS = ("dataset_release", "source_path", "target_path", "seed", "max_shared_users",
             "max_source_only_users", "max_target_only_users", "validation_fraction", "test_fraction")


def data_sha256(config: dict) -> str:
    """Training hyperparameters do not invalidate an unchanged prepared split."""
    return config_sha256({key: config[key] for key in DATA_KEYS if key in config})


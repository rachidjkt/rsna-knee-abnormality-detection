"""Mount discovery; never downloads competition data or creates network connections."""
from __future__ import annotations

import os
from pathlib import Path


def find_competition_root(input_dir: str | Path = "/kaggle/input", explicit: str | Path | None = None,
                          training: bool = True) -> Path:
    required = ["train.csv", "train_series.csv"] if training else ["test_series.csv", "sample_submission.csv"]
    if explicit:
        candidate = Path(explicit)
        if all((candidate / name).exists() for name in required):
            return candidate
        raise FileNotFoundError(f"Competition metadata missing from {candidate}")
    input_dir = Path(input_dir)
    candidates = [input_dir / "competitions/rsna-knee-abnormality-detection", input_dir / "rsna-knee-abnormality-detection"]
    candidates += sorted(input_dir.glob("*/rsna-knee-abnormality-detection"))
    for candidate in candidates:
        if all((candidate / name).exists() for name in required):
            return candidate
    raise FileNotFoundError("Attach the official competition dataset; accept Kaggle rules in your account first")


def find_image_root(data_root: str | Path, partition: str) -> Path:
    if partition not in ("train", "test"):
        raise ValueError("partition must be train or test")
    # Official images live under *_series. *_images is supported only for development fixtures.
    for directory in (Path(data_root) / f"{partition}_series", Path(data_root) / f"{partition}_images"):
        if directory.is_dir():
            return directory
    raise FileNotFoundError(f"No {partition}_series DICOM directory mounted at {data_root}")


def find_checkpoint(input_dir: str | Path, explicit: str | Path | None = None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(Path(input_dir).rglob("best.pt"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Found {len(candidates)} best.pt checkpoints; set KNEE_CHECKPOINT to the intended mounted model")
    return candidates[0]

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from knee_ai.constants import STUDY_ID, TARGETS
from knee_ai.data.dataset import StudyDataset
from knee_ai.data.metadata import load_series, load_inference_template
from knee_ai.models.multislice import MultiSliceClassifier
from knee_ai.preprocessing.dicom import PREPROCESS_VERSION
from knee_ai.training.engine import predict_loader, resolve_device
from knee_ai.utils.reproducibility import sha256_file, write_json


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[MultiSliceClassifier, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("targets") != TARGETS:
        raise ValueError("Checkpoint version or target order mismatch")
    if checkpoint.get("provenance", {}).get("preprocessing_version") != PREPROCESS_VERSION:
        raise ValueError("Checkpoint preprocessing differs from this code; use its original notebook/source or retrain")
    config = checkpoint["config"]
    # The complete encoder is in the checkpoint. No external weight file is needed at inference.
    model = MultiSliceClassifier(config["encoder"], config["dropout"], freeze_encoder=config["freeze_encoder"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def write_submission(template: pd.DataFrame, study_ids: list[str], probabilities: np.ndarray, path: str | Path) -> pd.DataFrame:
    if template.columns.tolist() != [STUDY_ID, *TARGETS] or template[STUDY_ID].duplicated().any():
        raise ValueError("Invalid submission template")
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(study_ids), 12) or len(study_ids) != len(set(study_ids)):
        raise ValueError("Duplicate study IDs or invalid prediction shape")
    if set(study_ids) != set(template[STUDY_ID]):
        raise ValueError("Predicted study IDs do not exactly match the template")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Predictions contain invalid probabilities")
    frame = pd.DataFrame(probabilities, columns=TARGETS, index=study_ids)
    result = frame.loc[template[STUDY_ID]].reset_index(names=STUDY_ID)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    reread = pd.read_csv(path, dtype={STUDY_ID: str})
    if reread.columns.tolist() != template.columns.tolist() or reread[STUDY_ID].tolist() != template[STUDY_ID].tolist():
        raise RuntimeError("Submission serialization validation failed")
    return result


def infer(checkpoint_path: str | Path, data_dir: str | Path, image_root: str | Path,
          output_path: str | Path = "submission.csv", device: str = "auto", batch_size: int = 2,
          num_workers: int = 0, encode_chunk_size: int = 64, cache_dir: str | Path | None = None,
          max_hours: float = 8.) -> dict:
    start = time.perf_counter()
    if max_hours <= 0 or batch_size < 1 or encode_chunk_size < 1:
        raise ValueError("Inference budget, batch size and encode_chunk_size must be positive")
    device = resolve_device(device)
    model, checkpoint = load_model(checkpoint_path, device)
    config = checkpoint["config"]
    if config.get("smoke_test", False):
        raise ValueError("A smoke-test checkpoint is diagnostic only; use a full training run for competition inference")
    data_dir = Path(data_dir)
    series = load_series(data_dir / "test_series.csv")
    template = load_inference_template(data_dir, series)
    dataset = StudyDataset(template[[STUDY_ID]], series, image_root, config["slices_per_series"],
                           config["image_size"], config["strategy"], config["central_fraction"], cache_dir=cache_dir,
                           error_policy=config["error_policy"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    ids, _, probabilities, _ = predict_loader(model, loader, device, config["amp"], encode_chunk_size,
                                               deadline=start + max_hours * 3600)
    write_submission(template, ids, probabilities, output_path)
    elapsed = time.perf_counter() - start
    diagnostics = {"synthetic_model": config["synthetic"], "studies": len(ids), "seconds": elapsed,
                   "seconds_per_study": elapsed / len(ids), "device": str(device),
                   "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
                   "checkpoint_sha256": sha256_file(checkpoint_path), "submission_sha256": sha256_file(output_path),
                   "max_hours": max_hours,
                   "runtime_warning": "Visible/synthetic examples do not validate the hidden-test 9-hour limit."}
    write_json(Path(output_path).with_suffix(".diagnostics.json"), diagnostics)
    return diagnostics

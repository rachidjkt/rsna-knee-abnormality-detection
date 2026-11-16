"""Small diagnostic training/reload check on the existing mounted image data."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from knee_ai.constants import STUDY_ID, TARGETS
from knee_ai.data.dataset import StudyDataset
from knee_ai.data.metadata import load_train, load_series
from knee_ai.inference.predict import load_model, write_submission
from knee_ai.training.engine import TrainConfig, train, resolve_device, predict_loader
from knee_ai.utils.reproducibility import write_json, sha256_file


def smoke_train(config: TrainConfig) -> dict:
    if not config.smoke_test or config.epochs != 1 or config.max_train_studies is None or config.max_val_studies is None:
        raise ValueError("Smoke test requires one epoch, explicit train/validation limits and smoke_test=True")
    start = time.perf_counter()
    summary = train(config)
    output = Path(config.output_dir)
    selection = json.loads((output / "run_studies.json").read_text())
    table = load_train(Path(config.data_dir) / "train.csv").set_index(STUDY_ID).loc[selection["val_ids"]].reset_index()
    series = load_series(Path(config.data_dir) / "train_series.csv")
    dataset = StudyDataset(table, series, config.image_root, config.slices_per_series, config.image_size,
                           config.strategy, config.central_fraction, cache_dir=None, error_policy=config.error_policy)
    device = resolve_device(config.device)
    model, checkpoint = load_model(output / "best.pt", device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    infer_start = time.perf_counter()
    ids, _, probabilities, _ = predict_loader(model, DataLoader(dataset, batch_size=config.batch_size), device,
                            config.amp, encode_chunk_size=32, deadline=start + config.max_hours * 3600)
    elapsed = time.perf_counter() - infer_start
    original = pd.read_csv(output / "validation_predictions.csv").set_index(STUDY_ID).loc[ids, TARGETS].to_numpy()
    np.testing.assert_allclose(probabilities, original, rtol=1e-4, atol=1e-5)
    template = pd.DataFrame({STUDY_ID: ids, **dict.fromkeys(TARGETS, .5)})
    write_submission(template, ids, probabilities, output / "smoke_predictions.csv")
    diagnostics = {"purpose": "pipeline diagnostic, not a medical-performance evaluation or submission",
                   "synthetic": config.synthetic, "smoke_test": True, "training": summary,
                   "checkpoint_reload_predictions_match": True, "inference_studies": len(ids),
                   "inference_seconds": elapsed, "inference_seconds_per_study": elapsed / len(ids),
                   "total_seconds": time.perf_counter() - start,
                   "inference_peak_gpu_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
                   "checkpoint_sha256": sha256_file(output / "best.pt"),
                   "runtime_limitations": "Subset timing at smoke resolution/slice count does not validate full hidden-test runtime."}
    write_json(output / "smoke_diagnostics.json", diagnostics)
    return diagnostics

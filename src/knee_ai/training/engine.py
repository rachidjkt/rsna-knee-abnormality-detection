from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from knee_ai.constants import STUDY_ID, TARGETS
from knee_ai.data.dataset import StudyDataset
from knee_ai.data.metadata import load_series, load_train, label_statistics
from knee_ai.data.splits import save_split, validate_split, include_partial_training
from knee_ai.evaluation.metrics import multilabel_metrics
from knee_ai.models.multislice import MultiSliceClassifier
from knee_ai.preprocessing.dicom import PREPROCESS_VERSION
from knee_ai.utils.reproducibility import package_source_sha256, seed_everything, seed_worker, sha256_file, write_json

LOG = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    data_dir: str = "data"
    image_root: str = "data/train_series"
    output_dir: str = "outputs/baseline"
    split_path: str = "experiments/study_split.json"
    encoder: str = "resnet18"
    encoder_weights: str | None = None
    freeze_encoder: bool = False
    slices_per_series: int = 8
    image_size: int = 224
    strategy: str = "uniform"
    central_fraction: float = .6
    dropout: float = .2
    seed: int = 42
    split_seed: int = 42
    val_fraction: float = .25
    epochs: int = 8
    batch_size: int = 2
    learning_rate: float = .0001
    weight_decay: float = .0001
    gradient_clip: float = 1.
    patience: int = 0
    checkpoint_selection: str = "validation_loss"
    class_weighting: bool = False
    augment: bool = False
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    cache_dir: str | None = None
    cache_limit_gb: float = 2.
    error_policy: str = "raise"
    max_hours: float = 7.
    synthetic: bool = False
    smoke_test: bool = False
    max_train_studies: int | None = None
    max_val_studies: int | None = None

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0 or self.num_workers < 0:
            raise ValueError("Invalid training configuration")
        if self.gradient_clip <= 0 or self.max_hours <= 0 or self.patience < 0:
            raise ValueError("Invalid training budget or clipping")
        if self.checkpoint_selection not in {"validation_loss", "last_epoch"}:
            raise ValueError("Unknown checkpoint selection policy")
        if self.checkpoint_selection == "last_epoch" and self.patience:
            raise ValueError("Fixed last-epoch selection requires patience=0")
        for limit in (self.max_train_studies, self.max_val_studies):
            if limit is not None and (limit < 1 or not self.smoke_test):
                raise ValueError("Study limits must be positive and explicitly marked smoke_test=True")

    @classmethod
    def from_dict(cls, value: dict) -> "TrainConfig":
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**value)


def resolve_device(requested: str) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def masked_bce(logits: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    if logits.shape != labels.shape or torch.isinf(labels).any():
        raise ValueError("Loss expects matching arrays with binary labels or NaN")
    known = torch.isfinite(labels)
    if not known.any():
        raise ValueError("Batch has no observed labels")
    safe_labels = torch.where(known, labels, torch.zeros_like(labels))
    if not ((safe_labels == 0) | (safe_labels == 1)).all():
        raise ValueError("Loss expects binary labels")
    losses = nn.functional.binary_cross_entropy_with_logits(logits, safe_labels, pos_weight=pos_weight, reduction="none")
    return losses[known].mean()


def positive_class_weights(table: pd.DataFrame) -> np.ndarray:
    positives = table[TARGETS].eq(1).sum().to_numpy(dtype=np.float32)
    negatives = table[TARGETS].eq(0).sum().to_numpy(dtype=np.float32)
    weights = np.clip(negatives / np.maximum(positives, 1), .25, 20)
    weights[(positives == 0) | (negatives == 0)] = 1
    return weights


@torch.inference_mode()
def predict_loader(model: MultiSliceClassifier, loader: DataLoader, device: torch.device,
                   amp: bool = True, encode_chunk_size: int = 64,
                   deadline: float | None = None) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    model.eval()
    ids, truths, predictions = [], [], []
    loss_sum, observed_count = 0., 0
    for batch in loader:
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Prediction budget exhausted; no incomplete submission is written")
        images, mask = batch["images"].to(device), batch["mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(images, mask, encode_chunk_size)
        known_count = int(torch.isfinite(labels).sum())
        if known_count:
            loss_sum += float(masked_bce(logits.float(), labels)) * known_count
            observed_count += known_count
        ids.extend(batch["study_id"])
        truths.append(labels.cpu().numpy())
        predictions.append(torch.sigmoid(logits.float()).cpu().numpy())
    if not predictions:
        raise ValueError("No studies to predict")
    return ids, np.concatenate(truths), np.concatenate(predictions), loss_sum / observed_count if observed_count else float("nan")


def train(config: TrainConfig) -> dict:
    start = time.perf_counter()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists() or (output / "last.pt").exists():
        raise FileExistsError(f"Existing run at {output}; choose a new output_dir to preserve it")
    table = load_train(Path(config.data_dir) / "train.csv")
    series = load_series(Path(config.data_dir) / "train_series.csv")
    split_path = Path(config.split_path)
    if not split_path.exists():
        save_split(table, split_path, config.split_seed, config.val_fraction)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split = include_partial_training(split, table)
    if "fold_index" in split and config.checkpoint_selection != "last_epoch":
        raise ValueError("Outer CV folds require checkpoint_selection='last_epoch'; choose epoch budget before evaluation")
    write_json(output / "split.json", split)
    common = dict(series=series, image_root=config.image_root, slices_per_series=config.slices_per_series,
                  image_size=config.image_size, strategy=config.strategy, central_fraction=config.central_fraction,
                  cache_dir=config.cache_dir, cache_limit_gb=config.cache_limit_gb, error_policy=config.error_policy)
    train_table = table.set_index(STUDY_ID).loc[split["train_ids"]].reset_index()
    val_table = table.set_index(STUDY_ID).loc[split["val_ids"]].reset_index()
    if config.max_train_studies is not None:
        train_table = train_table.iloc[:config.max_train_studies].copy()
    if config.max_val_studies is not None:
        val_table = val_table.iloc[:config.max_val_studies].copy()
    write_json(output / "run_studies.json", {"smoke_test": config.smoke_test,
               "train_ids": train_table[STUDY_ID].tolist(), "val_ids": val_table[STUDY_ID].tolist()})
    train_set = StudyDataset(train_table, augment=config.augment, **common)
    val_set = StudyDataset(val_table, augment=False, **common)
    generator = torch.Generator().manual_seed(config.seed)
    loaders = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                   pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loaders)
    val_loader = DataLoader(val_set, shuffle=False, **loaders)
    model = MultiSliceClassifier(config.encoder, config.dropout, config.encoder_weights, config.freeze_encoder).to(device)
    initial_hash = __import__("hashlib").sha256(b"".join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    pos_weight = None
    if config.class_weighting:
        pos_weight = torch.tensor(positive_class_weights(train_table), device=device)
    for target in TARGETS:
        if train_table[target].nunique() < 2:
            LOG.warning("Training class %s has one label value; no discriminative supervision", target)
    provenance = {"python": platform.python_version(), "torch": str(torch.__version__),
                  "torchvision": str(__import__("torchvision").__version__), "numpy": np.__version__,
                  "pandas": pd.__version__, "device": str(device),
                  "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  "train_csv_sha256": sha256_file(Path(config.data_dir) / "train.csv"),
                  "series_csv_sha256": sha256_file(Path(config.data_dir) / "train_series.csv"),
                  "split_sha256": sha256_file(output / "split.json"), "source_split_sha256": sha256_file(split_path),
                  "initial_model_sha256": initial_hash,
                  "encoder_weights_sha256": sha256_file(config.encoder_weights) if config.encoder_weights else None,
                  "preprocessing_version": PREPROCESS_VERSION, "package_source_sha256": package_source_sha256(),
                  "synthetic": config.synthetic}
    write_json(output / "config.json", asdict(config))
    write_json(output / "provenance.json", provenance)
    write_json(output / "split.json", split)
    write_json(output / "label_statistics.json", {"train": label_statistics(train_table), "val": label_statistics(val_table),
                                                "pos_weight": pos_weight.cpu().tolist() if pos_weight is not None else None})
    history, best_loss, stale, best_epoch = [], float("inf"), 0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        loss_sum, n_observed = 0., 0
        for batch in train_loader:
            if (time.perf_counter() - start) / 3600 > config.max_hours:
                raise TimeoutError("Training budget exhausted mid-epoch; completed checkpoints are preserved")
            images, mask, labels = (batch[key].to(device, non_blocking=True) for key in ("images", "mask", "labels"))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                logits = model(images, mask)
                loss = masked_bce(logits, labels, pos_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            observed = int(torch.isfinite(labels).sum())
            loss_sum += float(loss.detach()) * observed
            n_observed += observed
        ids, y, p, val_loss = predict_loader(model, val_loader, device, config.amp,
                                            deadline=start + config.max_hours * 3600)
        metrics = multilabel_metrics(y, p)
        record = {"epoch": epoch, "train_loss": loss_sum / n_observed, "val_loss": val_loss,
                  "macro_auc_all_12": metrics["macro_auc_all_12"], "macro_auc_defined": metrics["macro_auc_defined"],
                  "defined_classes": metrics["defined_classes"], "lr": optimizer.param_groups[0]["lr"],
                  "seconds": time.perf_counter() - epoch_start}
        history.append(record)
        scheduler.step()
        checkpoint = {"format_version": 1, "targets": TARGETS, "model": model.state_dict(),
                      "config": asdict(config), "epoch": epoch, "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                      "provenance": provenance}
        temporary = output / "last.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(output / "last.pt")
        if config.checkpoint_selection == "last_epoch" or val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save(checkpoint, output / "best.tmp")
            (output / "best.tmp").replace(output / "best.pt")
            write_json(output / "validation_metrics.json", metrics)
            frame = pd.DataFrame(p, columns=TARGETS)
            frame.insert(0, STUDY_ID, ids)
            frame.to_csv(output / "validation_predictions.csv", index=False)
            truth = pd.DataFrame(y, columns=TARGETS)
            truth.insert(0, STUDY_ID, ids)
            truth.to_csv(output / "validation_labels.csv", index=False)
        else:
            stale += 1
        write_json(output / "history.json", history)
        LOG.info("Epoch %d/%d train %.4f val %.4f AUC(defined %d/12) %s", epoch, config.epochs,
                 record["train_loss"], val_loss, metrics["defined_classes"], metrics["macro_auc_defined"])
        if config.patience and stale >= config.patience:
            break
    result = {"synthetic": config.synthetic, "smoke_test": config.smoke_test, "status": "completed", "best_epoch": best_epoch,
              "completed_epochs": len(history), "best_validation_loss": best_loss,
              "seconds": time.perf_counter() - start, "train_studies": len(train_set), "val_studies": len(val_set),
              "device": str(device), "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
              "metrics": json.loads((output / "validation_metrics.json").read_text()),
              "checkpoint": str(output / "best.pt")}
    write_json(output / "summary.json", result)
    return result

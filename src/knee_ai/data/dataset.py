from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from knee_ai.constants import PLANES, SERIES_ID, STUDY_ID, TARGETS
from knee_ai.preprocessing.dicom import PREPROCESS_VERSION, load_selected_series

LOG = logging.getLogger(__name__)


class StudyDataset(Dataset):
    def __init__(self, studies: pd.DataFrame, series: pd.DataFrame, image_root: str | Path,
                 slices_per_series: int = 8, image_size: int = 224, strategy: str = "uniform",
                 central_fraction: float = .6, cache_dir: str | Path | None = None,
                 cache_limit_gb: float = 2, augment: bool = False, error_policy: str = "raise"):
        self.studies = studies.reset_index(drop=True)
        self.series = {str(uid): group.copy() for uid, group in series.groupby(STUDY_ID)}
        self.root = Path(image_root)
        self.count, self.size, self.strategy = slices_per_series, image_size, strategy
        self.central_fraction, self.augment = central_fraction, augment
        self.error_policy = error_policy
        self.cache = Path(cache_dir) if cache_dir else None
        self.cache_limit = int(cache_limit_gb * 1024**3)
        if slices_per_series < 1 or image_size < 16 or error_policy not in ("raise", "skip_series"):
            raise ValueError("Invalid dataset configuration")
        if strategy not in ("uniform", "central") or not 0 < central_fraction <= 1:
            raise ValueError("Invalid sampling configuration")
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.studies)

    def _selected(self, uid: str) -> list[tuple[str, str]]:
        if uid not in self.series:
            raise ValueError(f"No series metadata for study {uid}")
        result = []
        for plane in PLANES:
            candidates = self.series[uid].loc[self.series[uid].Anatomical_Plane.eq(plane)]
            candidates = candidates.sort_values(["Fluid_Sensitive", "Fat_Suppression", SERIES_ID], ascending=[False, False, True])
            result.append((plane, str(candidates.iloc[0][SERIES_ID]) if len(candidates) else ""))
        return result

    def _cache_path(self, uid: str, selected: list[tuple[str, str]]) -> Path | None:
        if not self.cache:
            return None
        fingerprint = []
        for plane, sid in selected:
            directory = self.root / uid / sid
            if sid and directory.exists():
                fingerprint += [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                                for p in sorted(directory.iterdir()) if p.is_file()]
        payload = [PREPROCESS_VERSION, uid, selected, self.count, self.size, self.strategy,
                   self.central_fraction, self.error_policy, fingerprint]
        key = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
        return self.cache / f"{key}.pt"

    def __getitem__(self, index: int) -> dict:
        row = self.studies.iloc[index]
        uid = str(row[STUDY_ID])
        selected = self._selected(uid)
        cache_path = self._cache_path(uid, selected)
        if cache_path and cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            images, mask = cached["images"].float(), cached["mask"]
        else:
            all_images, all_masks = [], []
            for plane, sid in selected:
                if not sid:
                    images = torch.zeros(self.count, 1, self.size, self.size)
                    mask = torch.zeros(self.count, dtype=torch.bool)
                else:
                    try:
                        images, mask = load_selected_series(self.root / uid / sid, sid, plane, self.count,
                                                            self.size, self.strategy, self.central_fraction)
                    except Exception as exc:
                        if self.error_policy == "raise":
                            raise
                        LOG.warning("Skipping unreadable series %s (%s)", sid, type(exc).__name__)
                        images = torch.zeros(self.count, 1, self.size, self.size)
                        mask = torch.zeros(self.count, dtype=torch.bool)
                all_images.append(images)
                all_masks.append(mask)
            images, mask = torch.cat(all_images), torch.cat(all_masks)
            if not mask.any():
                raise RuntimeError(f"No usable MRI slices for study {uid}; cannot generate a credible prediction")
            if cache_path:
                occupied = sum(p.stat().st_size for p in self.cache.glob("*.pt"))
                if occupied + images.numel() * 4 + 4096 <= self.cache_limit:
                    temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
                    torch.save({"images": images, "mask": mask}, temporary)
                    temporary.replace(cache_path)
        if self.augment:
            # Study-consistent small intensity jitter; no clinically questionable anatomical flips.
            gain = .9 + .2 * torch.rand(1)
            images = (images * gain).clamp(0, 1)
        labels = torch.tensor([float(row.get(target, float("nan"))) for target in TARGETS], dtype=torch.float32)
        return {"images": images, "mask": mask, "labels": labels, "study_id": uid}

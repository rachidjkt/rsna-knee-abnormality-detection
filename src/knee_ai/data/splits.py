from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from knee_ai.constants import STUDY_ID, TARGETS
from knee_ai.data.metadata import labelled_studies, supervised_studies, label_statistics
from knee_ai.utils.reproducibility import write_json


def _groups(table: pd.DataFrame) -> tuple[np.ndarray, str]:
    if "PatientID" in table and table.PatientID.notna().all() and table.PatientID.astype(str).str.strip().ne("").all():
        return table.PatientID.astype(str).to_numpy(), "PatientID supplied by metadata (must verify provenance)"
    if "Report" in table:
        values = []
        for uid, report in zip(table[STUDY_ID], table.Report):
            normalized = re.sub(r"\s+", " ", str(report).strip().lower()) if pd.notna(report) else ""
            values.append(hashlib.sha256(normalized.encode()).hexdigest() if normalized else str(uid))
        return np.array(values), "normalized report hash proxy; NOT verified patient identity"
    return table[STUDY_ID].to_numpy(), "study ID; patient identity unavailable"


def make_split(table: pd.DataFrame, seed: int = 42, val_fraction: float = .25) -> dict:
    source_table = table
    table = labelled_studies(table)
    if not 0 < val_fraction < 1 or len(table) < 4:
        raise ValueError("Need at least four labelled studies and 0 < val_fraction < 1")
    groups, grouping = _groups(table)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("Need at least two distinct split groups")
    n_val = max(1, min(len(unique) - 1, round(len(unique) * val_fraction)))
    y = table[TARGETS].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    best = None
    # Label-aware holdout search is stratification, never prediction-based selection.
    for _ in range(1024):
        selected = rng.choice(unique, n_val, replace=False)
        mask = np.isin(groups, selected)
        train_y, val_y = y[~mask], y[mask]
        if not len(train_y) or not len(val_y):
            continue
        undefined = np.logical_or(val_y.sum(0) == 0, val_y.sum(0) == len(val_y)).sum()
        train_undefined = np.logical_or(train_y.sum(0) == 0, train_y.sum(0) == len(train_y)).sum()
        score = 100 * (undefined + train_undefined) + np.abs(val_y.mean(0) - y.mean(0)).sum() + abs(mask.mean() - val_fraction)
        if best is None or score < best[0]:
            best = score, mask
    mask = best[1]
    payload = {"seed": seed, "val_fraction": val_fraction, "grouping": grouping,
               "train_ids": table.loc[~mask, STUDY_ID].tolist(), "val_ids": table.loc[mask, STUDY_ID].tolist(),
               "labelled_studies": len(table),
               "label_table_sha256": hashlib.sha256(table[[STUDY_ID, *TARGETS]].sort_values(STUDY_ID).to_csv(index=False).encode()).hexdigest(),
               "counts": {part: {target: {"positive": int(table.loc[selection, target].sum()), "total": int(selection.sum())}
                                  for target in TARGETS} for part, selection in [("train", ~mask), ("val", mask)]}}
    validate_split(payload, table)
    return include_partial_training(payload, source_table)


def validate_split(split: dict, table: pd.DataFrame) -> None:
    train, val = split["train_ids"], split["val_ids"]
    if not train or not val or len(train) != len(set(train)) or len(val) != len(set(val)) or set(train) & set(val):
        raise ValueError("Split contains empty sets, duplicate IDs or overlapping studies")
    partial_policy = split.get("supervision_policy") == "complete_validation_partial_training"
    supervised = supervised_studies(table) if partial_policy else labelled_studies(table)
    excluded = split.get("excluded_partial_ids", []) if partial_policy else []
    if len(excluded) != len(set(excluded)) or set(excluded) & (set(train) | set(val)):
        raise ValueError("Excluded partial studies overlap the split")
    if set(train) | set(val) | set(excluded) != set(supervised[STUDY_ID]):
        raise ValueError("Split must cover exactly eligible supervised studies and declared exclusions")
    digest = hashlib.sha256(supervised[[STUDY_ID, *TARGETS]].sort_values(STUDY_ID).to_csv(index=False).encode()).hexdigest()
    if split.get("label_table_sha256") != digest:
        raise ValueError("Split label provenance mismatch")
    groups, _ = _groups(supervised)
    train_groups = set(groups[supervised[STUDY_ID].isin(train)])
    val_groups = set(groups[supervised[STUDY_ID].isin(val)])
    if train_groups & val_groups:
        raise ValueError("Split groups overlap")
    if partial_policy:
        complete_ids = set(labelled_studies(table)[STUDY_ID])
        if not set(val) <= complete_ids or set(excluded) & complete_ids:
            raise ValueError("Validation must be fully labelled; only partial studies may be excluded")
        group_by_id = dict(zip(supervised[STUDY_ID], groups))
        if any(group_by_id[uid] not in val_groups for uid in excluded):
            raise ValueError("Partial exclusion is justified only by a held-out group")


def save_split(table: pd.DataFrame, path: str | Path, seed: int = 42, val_fraction: float = .25) -> dict:
    split = include_partial_training(make_split(table, seed, val_fraction), table)
    write_json(path, split)
    return split


def include_partial_training(split: dict, table: pd.DataFrame) -> dict:
    """Extend a fixed reviewed holdout without changing its IDs or source file.

    Partial studies related to a validation group are excluded, never promoted
    into training. No report-derived labels or all-unknown rows are introduced.
    """
    validate_split(split, table)
    if split.get("supervision_policy") == "complete_validation_partial_training":
        return split
    supervised = supervised_studies(table)
    partial = supervised.loc[~supervised[TARGETS].notna().all(axis=1)]
    if partial.empty:
        return split  # Preserve existing official split byte-for-byte at its source.
    if "PatientID" in supervised and supervised.PatientID.notna().any():
        if supervised.PatientID.isna().any() or supervised.PatientID.astype(str).str.strip().eq("").any():
            raise ValueError("Partial expansion requires complete patient grouping when PatientID is supplied")
    groups, grouping = _groups(supervised)
    group_by_id = dict(zip(supervised[STUDY_ID], groups))
    held_out = {group_by_id[uid] for uid in split["val_ids"]}
    added = [uid for uid in partial[STUDY_ID] if group_by_id[uid] not in held_out]
    excluded = [uid for uid in partial[STUDY_ID] if group_by_id[uid] in held_out]
    extended = dict(split, supervision_policy="complete_validation_partial_training",
                    grouping=grouping, train_ids=[*split["train_ids"], *added], excluded_partial_ids=excluded,
                    partial_training_studies=len(added), excluded_partial_studies=len(excluded),
                    label_table_sha256=hashlib.sha256(supervised[[STUDY_ID, *TARGETS]].sort_values(STUDY_ID).to_csv(index=False).encode()).hexdigest())
    indexed = supervised.set_index(STUDY_ID)
    extended["counts"] = {part: label_statistics(indexed.loc[ids]) for part, ids in
                          [("train", extended["train_ids"]), ("val", extended["val_ids"])]}
    validate_split(extended, table)
    return extended


def make_grouped_folds(table: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> list[dict]:
    """Optional out-of-fold evaluation, separate from the established holdout."""
    complete = labelled_studies(table)
    groups, grouping = _groups(complete)
    unique = np.unique(groups)
    if not 2 <= n_splits <= len(unique):
        raise ValueError("Need 2 <= n_splits <= fully labelled groups")
    y = complete[TARGETS].to_numpy(dtype=float)
    rng, best = np.random.default_rng(seed), None
    for _ in range(512):
        assignment = np.array_split(rng.permutation(unique), n_splits)
        masks = [np.isin(groups, selected) for selected in assignment]
        score = 0.
        for mask in masks:
            for subset in (y[mask], y[~mask]):
                score += 100 * np.logical_or(subset.sum(0) == 0, subset.sum(0) == len(subset)).sum()
            score += np.abs(y[mask].mean(0) - y.mean(0)).sum() + abs(mask.mean() - 1 / n_splits)
        if best is None or score < best[0]:
            best = score, masks
    digest = hashlib.sha256(complete[[STUDY_ID, *TARGETS]].sort_values(STUDY_ID).to_csv(index=False).encode()).hexdigest()
    folds = []
    for index, mask in enumerate(best[1]):
        split = {"seed": seed, "fold_index": index, "n_splits": n_splits, "grouping": grouping,
                 "val_fraction": 1 / n_splits, "train_ids": complete.loc[~mask, STUDY_ID].tolist(),
                 "val_ids": complete.loc[mask, STUDY_ID].tolist(), "labelled_studies": len(complete),
                 "label_table_sha256": digest,
                 "counts": {part: label_statistics(complete.loc[selected]) for part, selected in [("train", ~mask), ("val", mask)]}}
        folds.append(include_partial_training(split, table))
    return folds

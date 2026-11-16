from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import pandas as pd

from knee_ai.constants import PLANES, SERIES_ID, STUDY_ID, TARGETS
from knee_ai.utils.reproducibility import sha256_file, write_json


def _read(path: str | Path, required: list[str]) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={STUDY_ID: str, SERIES_ID: str})
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"{Path(path).name}: missing columns {sorted(missing)}")
    for column in (STUDY_ID, SERIES_ID):
        if column in table:
            if table[column].isna().any() or table[column].str.strip().eq("").any():
                raise ValueError(f"Empty {column}")
            # IDs are used in paths: disallow traversal and absolute paths.
            if table[column].str.contains(r"[/\\:]|^\.{1,2}$", regex=True).any():
                raise ValueError(f"Unsafe identifier in {column}")
    return table


def load_train(path: str | Path) -> pd.DataFrame:
    table = _read(path, [STUDY_ID, *TARGETS])
    if table[STUDY_ID].duplicated().any():
        raise ValueError("Duplicate training study IDs")
    for target in TARGETS:
        table[target] = pd.to_numeric(table[target], errors="raise")
        valid = table[target].isna() | table[target].isin([0, 1])
        if not valid.all():
            raise ValueError(f"{target}: expected binary labels or missing values")
    return table


def load_series(path: str | Path) -> pd.DataFrame:
    table = _read(path, [STUDY_ID, SERIES_ID, "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"])
    if table[SERIES_ID].duplicated().any():
        raise ValueError("Duplicate series IDs")
    for field in ["Fluid_Sensitive", "Fat_Suppression"]:
        table[field] = pd.to_numeric(table[field], errors="raise")
        if not table[field].isin([0, 1]).all():
            raise ValueError(f"Invalid {field}")
    if not table.Anatomical_Plane.isin(PLANES).all():
        raise ValueError("Unknown anatomical plane")
    return table


def load_template(path: str | Path) -> pd.DataFrame:
    table = _read(path, [STUDY_ID, *TARGETS])
    if table.columns.tolist() != [STUDY_ID, *TARGETS]:
        raise ValueError("Submission columns/order differ from the official schema")
    if table[STUDY_ID].duplicated().any():
        raise ValueError("Duplicate submission study IDs")
    return table


def load_inference_template(data_dir: str | Path, series: pd.DataFrame) -> pd.DataFrame:
    """Keep official columns; derive row IDs from the active scoring metadata.

    Kaggle may replace test.csv/test_series.csv while leaving the small public
    sample submission unchanged. Never require those three sample rows at scoring.
    """
    data_dir = Path(data_dir)
    sample = load_template(data_dir / "sample_submission.csv")
    test_path = data_dir / "test.csv"
    if test_path.exists():
        # Read IDs only, even if unexpected target/report columns are present.
        studies = pd.read_csv(test_path, usecols=[STUDY_ID], dtype={STUDY_ID: str})
        if studies[STUDY_ID].isna().any() or studies[STUDY_ID].duplicated().any():
            raise ValueError("Missing or duplicate active test study IDs")
        if studies[STUDY_ID].str.contains(r"[/\\:]|^\.{1,2}$|^\s*$", regex=True).any():
            raise ValueError("Unsafe active test study identifier")
        if set(studies[STUDY_ID]) != set(series[STUDY_ID]):
            raise ValueError("Active test.csv and test_series.csv study coverage mismatch")
    else:
        studies = series[[STUDY_ID]].drop_duplicates().reset_index(drop=True)
    if set(sample[STUDY_ID]) == set(studies[STUDY_ID]):
        return sample
    logging.getLogger(__name__).warning("Sample submission rows differ from active test metadata; preserving schema and using active study IDs")
    result = studies.copy()
    for target in TARGETS:
        result[target] = .5
    return result[[STUDY_ID, *TARGETS]]


def labelled_studies(table: pd.DataFrame) -> pd.DataFrame:
    return table.loc[table[TARGETS].notna().all(axis=1)].copy().reset_index(drop=True)


def supervised_studies(table: pd.DataFrame) -> pd.DataFrame:
    """Any explicit binary target supplies supervision; all-unknown rows do not."""
    return table.loc[table[TARGETS].notna().any(axis=1)].copy().reset_index(drop=True)


def label_statistics(table: pd.DataFrame) -> dict:
    return {target: {"known": int(table[target].notna().sum()),
                     "positive": int(table[target].eq(1).sum()),
                     "negative": int(table[target].eq(0).sum()),
                     "missing": int(table[target].isna().sum())} for target in TARGETS}


def audit_metadata(data_dir: str | Path, output: str | Path | None = None) -> dict:
    data_dir = Path(data_dir)
    train = load_train(data_dir / "train.csv")
    series = load_series(data_dir / "train_series.csv")
    test = load_series(data_dir / "test_series.csv")
    template = load_template(data_dir / "sample_submission.csv")
    if set(series[STUDY_ID]) != set(train[STUDY_ID]):
        raise ValueError("Training study/series coverage mismatch")
    if set(test[STUDY_ID]) != set(template[STUDY_ID]):
        raise ValueError("Test series/template study coverage mismatch")
    if set(train[STUDY_ID]) & set(template[STUDY_ID]):
        raise ValueError("Train and test studies overlap")
    if set(series[SERIES_ID]) & set(test[SERIES_ID]):
        raise ValueError("Train and test series overlap")
    counts = series.groupby(STUDY_ID).size()
    supervised = labelled_studies(train)
    result = {
        "source": "official CSV metadata; no imaging inspected",
        "files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                  for p in sorted(data_dir.glob("*.csv"))},
        "train_studies": len(train), "train_series": len(series),
        "fully_labelled_studies": len(supervised),
        "partially_labelled_studies": int((train[TARGETS].notna().any(axis=1) & ~train[TARGETS].notna().all(axis=1)).sum()),
        "unlabelled_studies": int(train[TARGETS].isna().all(axis=1).sum()),
        "visible_test_studies": len(template), "visible_test_series": len(test),
        "label_statistics": {target: {"known": int(train[target].notna().sum()),
                                       "positive": int(train[target].sum()),
                                       "negative": int(train[target].eq(0).sum()),
                                       "missing": int(train[target].isna().sum()),
                                       "prevalence_among_known": float(train[target].mean()) if train[target].notna().any() else None}
                             for target in TARGETS},
        "anatomical_planes": {str(k): int(v) for k, v in series.Anatomical_Plane.value_counts().items()},
        "fluid_sensitive_series": int(series.Fluid_Sensitive.sum()),
        "fat_suppressed_series": int(series.Fat_Suppression.sum()),
        "series_per_study": {str(k): float(v) for k, v in counts.describe().items()},
        "report_present_count": int(train.Report.notna().sum()) if "Report" in train else 0,
        "patient_id_available": "PatientID" in train,
        "limitations": ["Prevalence describes the labelled subset, not all studies.",
                        "CSV files contain no pixel sizes, transfer syntaxes or verified patient identity.",
                        "Visible test examples do not estimate hidden-test runtime."]}
    if output:
        write_json(output, result)
    return result

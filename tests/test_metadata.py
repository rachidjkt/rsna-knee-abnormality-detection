import json

import numpy as np
import pandas as pd
import pytest

from knee_ai.constants import STUDY_ID, TARGETS
from knee_ai.data.metadata import audit_metadata, labelled_studies, load_series, load_train, load_inference_template
from knee_ai.data.splits import make_split, validate_split


def test_official_metadata_schema_and_sparse_labels():
    from pathlib import Path
    if not all((Path('data') / name).is_file() for name in ('train.csv', 'train_series.csv', 'test_series.csv', 'sample_submission.csv')):
        pytest.skip('Authorized competition metadata is not distributed; supply it in data/.')
    audit = audit_metadata("data")
    assert audit["train_studies"] == 4407
    assert audit["train_series"] == 24371
    assert audit["fully_labelled_studies"] == 58
    assert audit["unlabelled_studies"] == 4349
    assert all(item["known"] == 58 for item in audit["label_statistics"].values())


def test_unknown_labels_stay_unknown(fixtures):
    table = load_train(fixtures / "train.csv")
    assert len(labelled_studies(table)) == 8
    assert table[TARGETS].iloc[8:].isna().all().all()


@pytest.mark.parametrize("failure", ["invalid_label", "missing_column", "duplicate_id", "unsafe_id", "text_label"])
def test_rejects_invalid_labels(fixtures, tmp_path, failure):
    table = load_train(fixtures / "train.csv")
    if failure == "invalid_label":
        table.loc[0, TARGETS[0]] = 2
    elif failure == "missing_column":
        table = table.drop(columns=TARGETS[0])
    elif failure == "duplicate_id":
        table.loc[1, STUDY_ID] = table.loc[0, STUDY_ID]
    elif failure == "unsafe_id":
        table.loc[0, STUDY_ID] = "../escape"
    else:
        table[TARGETS[0]] = table[TARGETS[0]].astype(object)
        table.loc[0, TARGETS[0]] = "unknown"
    path = tmp_path / "bad.csv"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_train(path)


def test_series_requires_valid_plane(fixtures, tmp_path):
    table = load_series(fixtures / "train_series.csv")
    table.loc[0, "Anatomical_Plane"] = "Unknown"
    path = tmp_path / "bad_series.csv"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_series(path)


def test_split_deterministic_and_disjoint(fixtures):
    table = load_train(fixtures / "train.csv")
    split = make_split(table)
    assert split == make_split(table)
    assert not set(split["train_ids"]) & set(split["val_ids"])
    assert len(split["train_ids"]) + len(split["val_ids"]) == 8
    validate_split(split, table)
    broken = json.loads(json.dumps(split))
    broken["val_ids"].append(split["train_ids"][0])
    with pytest.raises(ValueError):
        validate_split(broken, table)


def test_grouped_patient_split(fixtures):
    table = load_train(fixtures / "train.csv")
    table["PatientID"] = [str(i // 2) for i in range(len(table))]
    split = make_split(table)
    validate_split(split, table)
    groups = table.set_index(STUDY_ID).PatientID
    assert not set(groups.loc[split["train_ids"]]) & set(groups.loc[split["val_ids"]])


def test_duplicate_reports_are_grouped(fixtures):
    table = load_train(fixtures / "train.csv")
    table.loc[0, "Report"] = "SAME  REPORT"
    table.loc[1, "Report"] = "same report"
    split = make_split(table)
    assert (table.loc[0, STUDY_ID] in split["train_ids"]) == (table.loc[1, STUDY_ID] in split["train_ids"])


def test_split_rejects_changed_labels(fixtures):
    table = load_train(fixtures / "train.csv")
    split = make_split(table)
    table.loc[0, TARGETS[0]] = 1 - table.loc[0, TARGETS[0]]
    with pytest.raises(ValueError, match="provenance"):
        validate_split(split, table)


def test_hidden_test_replacement_uses_active_ids_not_public_example(fixtures, tmp_path):
    sample = pd.read_csv(fixtures / "sample_submission.csv")
    sample.to_csv(tmp_path / "sample_submission.csv", index=False)
    series = load_series(fixtures / "test_series.csv")
    ids = series[STUDY_ID].unique().tolist()
    replacements = {uid: f"9.8.7.{i}" for i, uid in enumerate(ids)}
    series[STUDY_ID] = series[STUDY_ID].map(replacements)
    # A target-like extra column must never supply inference labels.
    active = pd.DataFrame({STUDY_ID: list(replacements.values()), "ACL": [1, 0]})
    active.to_csv(tmp_path / "test.csv", index=False)
    template = load_inference_template(tmp_path, series)
    assert template[STUDY_ID].tolist() == active[STUDY_ID].tolist()
    assert template[TARGETS].eq(.5).all().all()
    (tmp_path / "test.csv").unlink()
    assert set(load_inference_template(tmp_path, series)[STUDY_ID]) == set(series[STUDY_ID])


def test_active_test_coverage_mismatch_fails(fixtures, tmp_path):
    sample = pd.read_csv(fixtures / "sample_submission.csv")
    sample.to_csv(tmp_path / "sample_submission.csv", index=False)
    sample[[STUDY_ID]].iloc[:1].to_csv(tmp_path / "test.csv", index=False)
    with pytest.raises(ValueError, match="coverage mismatch"):
        load_inference_template(tmp_path, load_series(fixtures / "test_series.csv"))

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import torch

from knee_ai.constants import STUDY_ID, TARGETS
from knee_ai.data.metadata import load_train
from knee_ai.data.splits import make_split, include_partial_training, validate_split, make_grouped_folds
from knee_ai.evaluation.metrics import multilabel_metrics
from knee_ai.training.engine import TrainConfig, masked_bce, positive_class_weights, train


def test_cv_requires_prespecified_last_epoch(fixtures, tmp_path):
    table = load_train(fixtures / "train.csv")
    fold = make_grouped_folds(table, n_splits=2)[0]
    path = tmp_path / "fold.json"
    path.write_text(json.dumps(fold))
    args = dict(data_dir=str(fixtures), image_root=str(fixtures / "train_images"),
                split_path=str(path), encoder="tiny", image_size=32, slices_per_series=2,
                batch_size=2, epochs=2, device="cpu", synthetic=True)
    with pytest.raises(ValueError, match="Outer CV folds"):
        train(TrainConfig(output_dir=str(tmp_path / "rejected"), **args))
    result = train(TrainConfig(output_dir=str(tmp_path / "fixed"), checkpoint_selection="last_epoch", **args))
    checkpoint = torch.load(tmp_path / "fixed/best.pt", weights_only=True)
    last = torch.load(tmp_path / "fixed/last.pt", weights_only=True)
    assert result["best_epoch"] == checkpoint["epoch"] == last["epoch"] == 2
    for key in checkpoint["model"]:
        torch.testing.assert_close(checkpoint["model"][key], last["model"][key])
    with pytest.raises(ValueError, match="patience=0"):
        TrainConfig(checkpoint_selection="last_epoch", patience=2)


def test_partial_training_preserves_holdout_and_excludes_related_groups(fixtures):
    table = load_train(fixtures / "train.csv")
    original = make_split(table)
    table.loc[8, "ACL"] = 1
    table.loc[9, ["MCL", "Medial OA"]] = [0, 1]
    val_report = table.set_index(STUDY_ID).loc[original["val_ids"][0], "Report"]
    table.loc[10, "Report"] = val_report
    table.loc[10, "ACL"] = 0
    extended = include_partial_training(original, table)
    assert extended["val_ids"] == original["val_ids"]
    assert len(extended["train_ids"]) == len(original["train_ids"]) + 2
    assert extended["excluded_partial_ids"] == [table.loc[10, STUDY_ID]]
    assert table.loc[11, STUDY_ID] not in extended["train_ids"]
    validate_split(extended, table)
    changed = table.copy()
    changed.loc[8, "ACL"] = 0
    with pytest.raises(ValueError, match="provenance"):
        validate_split(extended, changed)


def test_patient_related_partial_is_never_added_to_training(fixtures):
    table = load_train(fixtures / "train.csv")
    table["PatientID"] = table[STUDY_ID]
    original = make_split(table)
    table.loc[8, "PatientID"] = original["val_ids"][0]
    table.loc[8, "ACL"] = 1
    extended = include_partial_training(original, table)
    assert extended["train_ids"] == original["train_ids"]
    assert extended["excluded_partial_ids"] == [table.loc[8, STUDY_ID]]
    table.loc[8, "PatientID"] = None
    with pytest.raises(ValueError, match="patient grouping"):
        include_partial_training(original, table)


def test_no_partial_labels_does_not_change_preserved_split(fixtures):
    table = load_train(fixtures / 'train.csv')
    original = make_split(table)
    assert include_partial_training(original, table) == original


def test_class_weighting_counts_only_observed_negatives():
    labels = pd.DataFrame(np.nan, index=range(100), columns=TARGETS)
    labels.loc[0, "ACL"] = 1
    labels.loc[[1, 2], "ACL"] = 0
    assert positive_class_weights(labels)[0] == 2
    assert np.all(positive_class_weights(labels)[1:] == 1)


def test_masked_loss_matches_observed_entries_and_zero_unknown_gradients():
    logits = torch.linspace(-5, 5, 24).reshape(2, 12).requires_grad_()
    labels = torch.full_like(logits, float("nan"))
    labels[0, [0, 2]], labels[1, [1, 7]] = torch.tensor([1., 0.]), torch.tensor([0., 1.])
    mask = torch.isfinite(labels)
    weights = torch.arange(1, 13, dtype=torch.float32)
    loss = masked_bce(logits, labels, weights)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], labels[mask],
                                                pos_weight=weights.expand_as(labels)[mask])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[~mask]) == 0
    changed = logits.detach().clone()
    changed[~mask] = 1000
    torch.testing.assert_close(masked_bce(changed, labels, weights), loss.detach())


def test_auc_confusion_and_prevalence_ignore_unknown_rows():
    y = np.tile(np.array([0, 1, np.nan, 0, 1])[:, None], (1, 12))
    p = np.tile(np.array([.1, .8, .99, .2, .9])[:, None], (1, 12))
    a = multilabel_metrics(y, p)
    p[2] = .01
    b = multilabel_metrics(y, p)
    assert a["macro_auc_all_12"] == b["macro_auc_all_12"] == 1
    for row in a["per_class"].values():
        assert row["known"] == 4 and row["positive"] == row["negative"] == 2 and row["missing"] == 1
        assert row["confusion_matrix_tn_fp_fn_tp"] == [2, 0, 0, 2]


def test_infinity_is_invalid_rather_than_an_unknown_label():
    y = np.zeros((2, 12))
    y[0, 0] = np.inf
    with pytest.raises(ValueError):
        multilabel_metrics(y, np.full_like(y, .5))
    with pytest.raises(ValueError):
        masked_bce(torch.zeros(2, 12), torch.tensor(y))


def test_grouped_folds_are_reproducible_and_cover_each_reviewed_study_once(fixtures):
    table = load_train(fixtures / "train.csv")
    table["PatientID"] = [str(i // 2) for i in range(len(table))]
    folds = make_grouped_folds(table, n_splits=3)
    assert folds == make_grouped_folds(table, n_splits=3)
    ids = []
    for fold in folds:
        validate_split(fold, table)
        ids.extend(fold["val_ids"])
    assert set(Counter(ids).values()) == {1}
    assert len(ids) == 8


def test_training_really_uses_partial_rows_and_saves_known_label_counts(fixtures, tmp_path):
    table = load_train(fixtures / "train.csv")
    original = make_split(table)
    table.loc[8, "ACL"] = 1
    table.loc[9, "MCL"] = 0
    table.to_csv(tmp_path / "train.csv", index=False)
    (tmp_path / "train_series.csv").write_bytes((fixtures / "train_series.csv").read_bytes())
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(original))
    original_bytes = split_path.read_bytes()
    result = train(TrainConfig(data_dir=str(tmp_path), image_root=str(fixtures / "train_images"),
                   output_dir=str(tmp_path / "run"), split_path=str(split_path), encoder="tiny",
                   image_size=32, slices_per_series=2, batch_size=2, epochs=1, device="cpu",
                   class_weighting=True, synthetic=True))
    assert result["train_studies"] == len(original["train_ids"]) + 2
    assert split_path.read_bytes() == original_bytes
    counts = json.loads((tmp_path / "run/label_statistics.json").read_text())
    assert counts["train"]["ACL"]["known"] == len(original["train_ids"]) + 1
    assert counts["train"]["ACL"]["missing"] == 1

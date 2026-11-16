"""Re-audit raw labels, the preserved holdout and optional grouped CV.

Exports aggregate statistics only. Does not extract labels from reports or read
pixels. Candidate fold IDs stay in the ignored experiments directory.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from knee_ai.constants import STUDY_ID, TARGETS
from knee_ai.data.metadata import load_train, labelled_studies, supervised_studies, label_statistics
from knee_ai.data.splits import _groups, include_partial_training, make_grouped_folds
from knee_ai.utils.reproducibility import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="experiments/study_split.json")
    parser.add_argument("--output", default="reports/label_availability")
    parser.add_argument("--fold-dir", default="experiments/cv_seed42")
    args = parser.parse_args()
    train_path = Path(args.data_dir) / "train.csv"
    table = load_train(train_path)
    source_split = json.loads(Path(args.split).read_text())
    split = include_partial_training(source_split, table)
    indexed = table.set_index(STUDY_ID)
    all_stats = label_statistics(table)
    train_stats, val_stats = label_statistics(indexed.loc[split["train_ids"]]), label_statistics(indexed.loc[split["val_ids"]])
    rows = [{"target": target, "known_positive": all_stats[target]["positive"],
             "known_negative": all_stats[target]["negative"], "missing": all_stats[target]["missing"],
             "available_supervised_examples": all_stats[target]["known"],
             **{f"{part}_{key}": value for part, stats in [("train", train_stats), ("val", val_stats)]
                for key, value in stats[target].items()}} for target in TARGETS]
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output.with_suffix(".csv"), index=False)
    with train_path.open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    raw_tokens = {target: dict(Counter(row[target] for row in raw)) for target in TARGETS}
    supervised = supervised_studies(table)
    groups, grouping = _groups(table)
    supervised_groups, _ = _groups(supervised)
    folds = make_grouped_folds(table, 5, 42)
    fold_dir = Path(args.fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    for index, fold in enumerate(folds):
        path = fold_dir / f"fold_{index}.json"
        if path.exists() and json.loads(path.read_text()) != fold:
            raise FileExistsError(f"Preserve existing different fold at {path}")
        if not path.exists():
            write_json(path, fold)
    result = {"date": "2026-10-09", "source": "supplied CSV labels; raw tokens independently counted; no pixels/relabeling",
              "train_csv_sha256": sha256_file(train_path), "preserved_split_sha256": sha256_file(args.split),
              "known_targets_per_study": {str(k): int(v) for k, v in table[TARGETS].notna().sum(axis=1).value_counts().sort_index().items()},
              "fully_labelled": len(labelled_studies(table)), "any_labelled": len(supervised),
              "partially_labelled": len(supervised) - len(labelled_studies(table)), "unlabelled": len(table) - len(supervised),
              "additional_supervised_studies_available": len(set(supervised[STUDY_ID]) - set(labelled_studies(table)[STUDY_ID])),
              "raw_target_tokens": raw_tokens, "grouping": grouping,
              "duplicate_report_rows_all_studies": len(table) - len(set(groups)),
              "distinct_supervised_groups": len(set(supervised_groups)), "patient_id_available": "PatientID" in table,
              "holdout": {"train_studies": len(split["train_ids"]), "val_studies": len(split["val_ids"]), "per_class": rows},
              "optional_grouped_cv": [{"fold": i, "train_studies": len(f["train_ids"]), "val_studies": len(f["val_ids"]),
                                       "defined_val_classes": sum(v["positive"] > 0 and v["negative"] > 0 for v in f["counts"]["val"].values()),
                                       "counts": f["counts"], "split_sha256": sha256_file(fold_dir / f"fold_{i}.json")}
                                      for i, f in enumerate(folds)],
              "conclusion": "No additional labels are recoverable by changing the split. CV reuses the same known cohort with disjoint held-out groups; no clinical results are available."}
    write_json(output.with_suffix(".json"), result)
    print(frame.to_string(index=False))
    print({key: result[key] for key in ["fully_labelled", "partially_labelled", "unlabelled", "additional_supervised_studies_available"]})


if __name__ == "__main__":
    main()

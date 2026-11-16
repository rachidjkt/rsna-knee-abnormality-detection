from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from knee_ai.constants import STUDY_ID, TARGETS
from knee_ai.data.metadata import load_train
from knee_ai.data.splits import save_split
from knee_ai.evaluation.metrics import paired_bootstrap
from knee_ai.training.engine import TrainConfig, train
from knee_ai.utils.reproducibility import write_json


def compare_conditions(config: TrainConfig, output_dir: str | Path, seeds: list[int], bootstrap_draws: int = 200) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Need distinct experiment seeds")
    if config.patience != 0:
        raise ValueError("Controlled comparison uses equal epochs; set patience=0")
    if not Path(config.split_path).exists():
        save_split(load_train(Path(config.data_dir) / "train.csv"), config.split_path, config.split_seed, config.val_fraction)
    write_json(output / "experiment_manifest.json", {"base_config": asdict(config), "seeds": seeds,
               "research_question": "Does central physical-location slice sampling improve over full-span uniform sampling?",
               "changed_variable": "strategy only; series and total slot count are identical",
               "synthetic": config.synthetic, "status": "running"})
    rows, paired = [], []
    for seed in seeds:
        summaries = {}
        for strategy in ("uniform", "central"):
            values = asdict(config)
            values.update(seed=seed, strategy=strategy, output_dir=str(output / f"{strategy}_seed{seed}"))
            summary = train(TrainConfig.from_dict(values))
            summaries[strategy] = summary
            metric = summary["metrics"]
            row = {"seed": seed, "condition": strategy, "synthetic": config.synthetic,
                   "macro_auc_all_12": metric["macro_auc_all_12"], "macro_auc_defined": metric["macro_auc_defined"],
                   "defined_classes": metric["defined_classes"], "seconds": summary["seconds"]}
            row.update({target: metric["per_class"][target]["roc_auc"] for target in TARGETS})
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
        a_dir, b_dir = output / f"uniform_seed{seed}", output / f"central_seed{seed}"
        a_prov = json.loads((a_dir / "provenance.json").read_text())
        b_prov = json.loads((b_dir / "provenance.json").read_text())
        for key in ("split_sha256", "initial_model_sha256", "train_csv_sha256", "encoder_weights_sha256"):
            if a_prov[key] != b_prov[key]:
                raise RuntimeError(f"Experiment control mismatch: {key}")
        a = pd.read_csv(a_dir / "validation_predictions.csv").set_index(STUDY_ID)
        b = pd.read_csv(b_dir / "validation_predictions.csv").set_index(STUDY_ID).loc[a.index]
        y = pd.read_csv(a_dir / "validation_labels.csv").set_index(STUDY_ID).loc[a.index]
        am, bm = summaries["uniform"]["metrics"]["macro_auc_all_12"], summaries["central"]["metrics"]["macro_auc_all_12"]
        paired.append({"seed": seed, "delta_macro_auc_all_12": bm - am if am is not None and bm is not None else None,
                       "bootstrap": paired_bootstrap(y[TARGETS].to_numpy(), a[TARGETS].to_numpy(), b[TARGETS].to_numpy(),
                                                    seed=seed, draws=bootstrap_draws)})
    result = {"synthetic": config.synthetic, "status": "completed", "rows": rows, "paired": paired,
              "interpretation": "Synthetic engineering comparison only; real MRI experiment not executed." if config.synthetic else
              "Preliminary labelled-subset comparison. Small holdout, study/report grouping and seed variability limit inference."}
    write_json(output / "comparison.json", result)
    return result

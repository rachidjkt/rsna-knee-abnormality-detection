# RSNA Knee Abnormality Detection

A study-level knee MRI pipeline exploring multi-slice learning, sparse supervision, and reproducible evaluation. Local source, CSV schemas, and notebook mount paths identify the competition as **RSNA Knee Abnormality Detection** (`rsna-knee-abnormality-detection`). It predicts twelve findings: ACL, MCL, medial/lateral meniscus, medial/lateral/patellofemoral osteoarthritis, effusion, synovitis, Baker's cyst, contusion, and fracture.

**Status:** a reviewable edition of the existing ResNet18 baseline, preprocessing, inference, and slice-selection experiment. Synthetic checks exercise DICOM decoding, training/reload, and output serialization; they do not establish medical performance. See [VERIFICATION.md](VERIFICATION.md) and [experiment scope](docs/EXPERIMENTS.md).

## Implemented pipeline

1. Validate study/series metadata; preserve unknown labels as NaN. Reports do not enter the image model or supply validation truth.
2. Choose one fluid-sensitive series per available sagittal/coronal/axial plane. Order by projected DICOM position, with unambiguous instance-number fallback.
3. Compare full-span uniform sampling with a central-location prior. Apply fixed reference normalization, letterboxing, and missing-slot/view masks.
4. Encode valid slices with shared ResNet18, masked mean pooling, and a twelve-output head. Real-data defaults: frozen ImageNet encoder, 24 slots, 224×224 pixels.
5. Train using AdamW, masked BCE, optional training-only class weights, AMP, gradient clipping, cosine scheduling, fixed seeds, and bounded cache/runtime settings.
6. Reload complete checkpoints without weight downloads. Validate prediction coverage, target order, finite probabilities, and serialized submission structure.

The small `tiny` encoder is used for synthetic checks; it is distinct from the ResNet18 baseline.

## Setup and first run

From the repository root on Windows with Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,dicom]"
.\.venv\Scripts\python.exe -m knee_ai.cli --help
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/smoke.py --output outputs/smoke_first --device cpu
```

Use a fresh output directory for every smoke run. This creates synthetic DICOMs, trains the tiny model for two epochs, reloads its checkpoint, and predicts three synthetic test studies. No data or weight download is needed. `requirements-tested.txt` records historical direct versions; `pyproject.toml` declares supported ranges. A fresh dependency installation was not verified during preparation. On Linux use `python3.12 -m venv .venv` and `.venv/bin/python`.

Separate synthetic training and inference:

```powershell
.\.venv\Scripts\python.exe -m knee_ai.cli synthetic --output outputs/synthetic_data
.\.venv\Scripts\python.exe -m knee_ai.cli train --config configs/synthetic.json
.\.venv\Scripts\python.exe -m knee_ai.cli infer --checkpoint outputs/synthetic_cpu/best.pt --data-dir outputs/synthetic_data --image-root outputs/synthetic_data/test_images --device cpu --output outputs/synthetic_submission.csv
```

## Authorized real-data workflow

Accept the Kaggle competition terms and obtain authorized inputs. Supply `train.csv`, `train_series.csv`, `test_series.csv`, and `sample_submission.csv` in `data/`, with pixels separately under `train_series/<study>/<series>/*.dcm` and `test_series/...`. No official data, reports, identifiers, images, checkpoints, caches, or predictions are distributed.

```powershell
.\.venv\Scripts\python.exe -m knee_ai.cli audit --data-dir data
.\.venv\Scripts\python.exe -m knee_ai.cli split --data-dir data --output outputs/study_split.json --seed 42
```

Copy `configs/baseline.json` to `configs/my_local_run.json`; set `data_dir`, `image_root`, `encoder_weights`, `output_dir`, `split_path`, `cache_dir`, and `device`. The supplied recipe uses explicit Kaggle paths. Obtain the torchvision ResNet18 ImageNet checkpoint separately and record provenance. `scripts/fetch_encoder_weights.py` is an optional deliberate download utility, never part of inference.

```powershell
.\.venv\Scripts\python.exe -m knee_ai.cli train --config configs/my_local_run.json
.\.venv\Scripts\python.exe -m knee_ai.cli infer --checkpoint outputs/my_run/best.pt --data-dir data --image-root data/test_series --device cpu --output outputs/submission.csv
```

Set `output_dir=outputs/my_run` to match this example. Do not use full training as a smoke test. [docs/KAGGLE.md](docs/KAGGLE.md) describes offline notebooks. Nothing submits automatically. `.env.example` documents optional notebook variables; local CLI configuration is JSON.

## Validation and supported evidence

The metric used here is mean ROC-AUC across twelve findings. Undefined per-target AUC is null. `macro_auc_all_12` is null unless all targets are evaluable; `macro_auc_defined` is separately labeled and cannot replace it.

Historical local metadata audits recorded 4,407 training studies: 58 fully labeled and 4,349 unlabelled. These are historical aggregate findings; metadata is not distributed. The baseline used 44 training/14 validation studies. Splits enforce study disjointness and group by complete patient IDs when available, otherwise normalized report-hash proxy. That proxy **does not prove patient independence**. Partial studies sharing a held-out group are excluded from training; missing labels never become negatives.

| Evidence | Supported interpretation |
|---|---|
| [Historical user-pasted log](evidence/historical_v1_user_pasted_log.json) | Reported real-MRI ResNet18 run; loss-selected epoch 8 macro AUC 0.6345418470 on 14 validation studies |
| Same log | Maximum observed epoch AUC 0.6362779582 at epochs 5–6; a different checkpoint-selection statistic |
| Current synthetic checks | Software decoding, splits/losses, training/reload, inference interfaces and serialization |

The scored checkpoint, exact original split, per-target predictions, and independent leaderboard receipt are unavailable here. The pasted log is user-provided historical evidence, not an independently reproduced result. No leaderboard score is presented as verified. The later user-reported 0.892 concerns a third-party ensemble investigation, not this baseline. Repeated selection on the small holdout limits generalization claims. Grouped CV requires fixed last-epoch selection to prevent checkpoint selection on outer validation folds.

## Main files

| Location | Role |
|---|---|
| `src/knee_ai/data/`, `preprocessing/` | Schemas, synthetic fixtures, splits, series/slice processing |
| `src/knee_ai/models/`, `training/` | Baseline, masked supervision, checkpoints |
| `src/knee_ai/evaluation/`, `inference/` | Metrics, matched sampler comparison, prediction export |
| `configs/` | Explicit data/model/cache paths, seeds and run recipes |
| `scripts/smoke.py`, `tests/` | Integration and failure-case checks |
| `notebooks/` | Source-embedded offline train, smoke, infer, and sampler experiment |

Regenerate notebooks with `python scripts/build_notebooks.py` after package source changes. Notebook outputs are cleared. Asset-dependent tests skip when optional official metadata, weights, or Linux codec bundles are absent.

## Limitations and next improvements

No clinical validation, independent patient-level external evaluation, calibration, hidden-test runtime guarantee, or new real-MRI training was performed. Enhanced multi-frame DICOM needs a separate adapter. Oblique grids are preserved rather than fully resampled. Synthetic scores are not biomedical evidence. Incompatible historical checkpoints require their matching original source; do not bypass preprocessing checks.

Next: recover scored-run artifacts, establish reliable patient grouping, and conduct a prespecified independent evaluation before comparing models.

## Attribution and rights

This edition packages the existing local `knee_ai` baseline and its engineering checks. PyTorch, torchvision, pydicom, NumPy, pandas, scikit-learn, ResNet18/ImageNet resources, and Kaggle data retain their respective terms. No dataset or pretrained-model authorship is claimed. Later borrowed ensembles with unresolved rights remain in the original workspace. No project-wide license grant is added while historical authorship and rights remain uncertain.

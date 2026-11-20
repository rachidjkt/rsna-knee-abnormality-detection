# RSNA knee MRI: multi-slice engineering baseline

A study-level MRI classification pipeline for twelve knee findings. The engineering problem is to turn variable-length DICOM series and sparse labels into consistent model inputs, reproducible training runs, and validated prediction files.

The implementation covers metadata validation, geometric slice ordering, multi-plane sampling, missing-view masks, masked-label training, checkpoint reload, and submission-format checks. A small synthetic pipeline exercises these interfaces locally; a separately scoped historical log records an earlier real-MRI ResNet18 run. This is an engineering baseline, not a clinically validated system.

## Architecture

```text
Study/series CSVs + DICOMs -> metadata and label audit -> study/group split
    -> series selection -> ordered/sampled/normalized slices + masks
    -> shared ResNet18 -> masked mean pooling -> twelve finding probabilities
    -> checkpoint reload -> coverage/order/probability checks -> submission CSV
```

- One fluid-sensitive series per available sagittal, coronal, and axial plane; geometric ordering with an unambiguous instance-number fallback.
- Uniform full-span and central-location sampling; fixed normalization, letterboxing, and missing-slot/view masks. Real-data defaults use 24 slots at 224 x 224 pixels and a frozen ImageNet encoder.
- AdamW, masked BCE, optional training-only class weights, AMP, clipping, cosine scheduling, fixed seeds, and bounded cache/runtime settings. Unknown labels stay NaN; reports are not model inputs or validation truth.
- Complete checkpoints support inference without downloading weights. The separate `tiny` encoder is used for synthetic checks.

Source is organized under `src/knee_ai/{data,preprocessing,models,training,evaluation,inference}`. JSON recipes live in `configs/`; source-embedded offline workflows live in `notebooks/`.

## Setup and synthetic demo

From the repository root in PowerShell with Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,dicom]"
.\.venv\Scripts\python.exe -m knee_ai.cli --help
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/smoke.py --output outputs/smoke_first --device cpu
```

Use a fresh smoke output directory. The demo creates synthetic DICOMs, trains the tiny model for two epochs, reloads its checkpoint, and predicts three synthetic test studies. No dataset or model-weight download is needed. On Linux, use `python3.12 -m venv .venv` and `.venv/bin/python`.

`pyproject.toml` declares supported dependency ranges; `requirements-tested.txt` records historical direct Windows versions. A fresh installation was not verified during portfolio preparation. Asset-dependent tests skip without optional official metadata, weights, or prepared Linux codec bundles.

## Authorized real-data workflow

Obtain competition inputs under the [Kaggle competition terms](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules). The CLI expects `train.csv`, `train_series.csv`, `test_series.csv`, and `sample_submission.csv` in `data/`, with pixels under `train_series/<study>/<series>/*.dcm` and `test_series/...`.

```powershell
.\.venv\Scripts\python.exe -m knee_ai.cli audit --data-dir data
.\.venv\Scripts\python.exe -m knee_ai.cli split --data-dir data --output outputs/study_split.json --seed 42
Copy-Item configs/baseline.json configs/my_local_run.json
```

Before training, edit that copy's `data_dir`, `image_root`, `encoder_weights`, `output_dir`, `split_path`, `cache_dir`, and `device`. Set `output_dir` to `outputs/my_run` for the example below and `split_path` to `outputs/study_split.json`. The supplied baseline recipe uses Kaggle mount paths. Obtain the torchvision ResNet18 ImageNet checkpoint separately and record its provenance; inference uses the complete trained checkpoint.

```powershell
.\.venv\Scripts\python.exe -m knee_ai.cli train --config configs/my_local_run.json
.\.venv\Scripts\python.exe -m knee_ai.cli infer --checkpoint outputs/my_run/best.pt --data-dir data --image-root data/test_series --device cpu --output outputs/submission.csv
```

See [docs/KAGGLE.md](docs/KAGGLE.md) for offline notebook and codec setup. Nothing submits automatically. Official data, reports, identifiers, pixels, weights, and predictions are not distributed here.

## Supported evidence

| Evidence | Scope and result |
| --- | --- |
| Historical software verification | 54 tests passed; 45 asset-dependent tests skipped |
| Historical synthetic integration | Two-epoch tiny-model train/reload/infer smoke and all four notebooks executed on synthetic DICOMs |
| [Historical user-pasted MRI log](evidence/historical_v1_user_pasted_log.json) | ResNet18, 44 training / 14 validation studies; loss-selected epoch 8 macro ROC-AUC 0.6345418470 |
| Same historical log | Maximum observed epoch AUC 0.6362779582 at epochs 5-6; distinct from the loss-selected checkpoint statistic |

Synthetic checks establish software behavior, not real-MRI performance. The historical MRI checkpoint, exact split IDs, and per-target predictions are unavailable here, so that run is not independently reproduced. No verified leaderboard score or medal is claimed. Other reported scores are not attached to this baseline; see [experiment scope](docs/EXPERIMENTS.md).

The historical metadata audit counted 4,407 studies, including 58 fully labeled and 4,349 unlabelled. Current split logic enforces study disjointness and uses complete patient IDs when available, otherwise a normalized report-hash proxy or study IDs. Partial studies sharing a held-out group are excluded from training. These safeguards do not establish patient independence.

The reported metric is mean ROC-AUC across twelve findings. Undefined per-target AUC is null; `macro_auc_all_12` is null unless all targets are evaluable. `macro_auc_defined` is a separate statistic. [VERIFICATION.md](VERIFICATION.md) retains exact commands, source hashes, skips, and verification limits.

## Limitations

The small historical holdout and repeated checkpoint inspection limit generalization claims. There is no clinical validation, verified patient-independent external evaluation, calibration study, or hidden-test runtime guarantee. Synthetic scores are not biomedical evidence. Enhanced multi-frame DICOM requires another adapter; oblique grids are not fully resampled. Historical checkpoints require matching source and preprocessing. A leaderboard screenshot without matching model/submission evidence cannot validate this baseline.

## Attribution and licensing

The model uses torchvision's ResNet18 and separately obtained ImageNet weights. PyTorch, pydicom, and other dependencies retain their own terms; competition inputs require separate authorization. [ATTRIBUTION.md](ATTRIBUTION.md) identifies upstream resources, the available source-history boundary, and the excluded ensemble scope. No project-wide license has been selected, and no dataset or pretrained-model authorship is claimed.

"""Generate standalone offline notebooks directly from the current package source."""
import base64
import gzip
import hashlib
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

SETUP = (ROOT / "scripts/notebook_bootstrap.py").read_text(encoding="utf-8")

TRAIN_CONFIG = '''from knee_ai.data.metadata import audit_metadata, load_train, load_series, labelled_studies
from knee_ai.data.splits import save_split
from knee_ai.data.dataset import StudyDataset
from knee_ai.training.engine import TrainConfig, train
from knee_ai.utils.kaggle import find_competition_root, find_image_root
from knee_ai.utils.reproducibility import configure_logging, write_json
configure_logging()
SMOKE_TEST = globals().get("SMOKE_TEST", False)

DATA = find_competition_root(INPUT, os.environ.get("KNEE_DATA_DIR"), training=True)
IMAGES = find_image_root(DATA, "train")
audit = audit_metadata(DATA, WORK / "metadata_audit.json")
print({key: audit[key] for key in ["train_studies", "train_series", "fully_labelled_studies", "unlabelled_studies"]})
# A complete provenance record is written for every run. No report text enters the model.
weights = os.environ.get("KNEE_ENCODER_WEIGHTS")
if not weights and not SYNTHETIC:
    candidates = sorted(INPUT.rglob("resnet18-f37072fd.pth"))
    if len(candidates) == 1:
        weights = str(candidates[0])
if not SYNTHETIC and not weights and os.environ.get("KNEE_ALLOW_RANDOM_INIT", "0") != "1":
    raise RuntimeError("Attach artifacts/pretrained as a Kaggle dataset; this 58-label baseline defaults to pretrained ResNet18")
if weights:
    from knee_ai.utils.reproducibility import sha256_file
    weights_path = Path(weights)
    expected = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
    if weights_path.name != "resnet18-f37072fd.pth":
        sidecar = weights_path.parent / "weights_manifest.json"
        if not sidecar.exists():
            raise RuntimeError("Custom encoder weights require weights_manifest.json with SHA256 provenance")
        expected = json.loads(sidecar.read_text())["sha256"]
    if sha256_file(weights_path) != expected:
        raise RuntimeError("Pretrained weight checksum mismatch")

config = TrainConfig(data_dir=str(DATA), image_root=str(IMAGES), output_dir=str(WORK / ("real_smoke" if SMOKE_TEST else "baseline")),
                     split_path=os.environ.get("KNEE_SPLIT_PATH", str(WORK / "study_split.json")),
                     checkpoint_selection=os.environ.get("KNEE_CHECKPOINT_SELECTION", "validation_loss"),
                     encoder=os.environ.get("KNEE_ENCODER", "tiny" if SYNTHETIC else "resnet18"),
                     encoder_weights=weights, freeze_encoder=bool(weights),
                     image_size=48 if SYNTHETIC else (160 if SMOKE_TEST else 224), slices_per_series=2 if SYNTHETIC or SMOKE_TEST else 8,
                     epochs=1 if SMOKE_TEST else (2 if SYNTHETIC else 8), batch_size=1 if SMOKE_TEST else 2, seed=42, split_seed=42,
                     device="cuda" if torch.cuda.is_available() else "cpu", amp=True, patience=0,
                     cache_dir=None if SMOKE_TEST else str(WORK / "cache"), cache_limit_gb=2,
                     max_hours=.25 if SMOKE_TEST else 7, synthetic=SYNTHETIC,
                     smoke_test=SMOKE_TEST, max_train_studies=4 if SMOKE_TEST else None, max_val_studies=2 if SMOKE_TEST else None)
write_json(WORK / "requested_config.json", __import__("dataclasses").asdict(config))
table, series = load_train(DATA / "train.csv"), load_series(DATA / "train_series.csv")
if not Path(config.split_path).exists():
    save_split(table, config.split_path, config.split_seed, config.val_fraction)
split = json.loads(Path(config.split_path).read_text())
sample = table.loc[table.StudyInstanceUID.isin(split["train_ids"][:1])]
preflight = StudyDataset(sample, series, IMAGES, config.slices_per_series, config.image_size)[0]
print({"preflight_shape": list(preflight["images"].shape), "usable_slices": int(preflight["mask"].sum()),
       "train_studies": len(split["train_ids"]), "validation_studies": len(split["val_ids"])})
'''

TRAIN_RUN = '''summary = train(config)
print({"completed_epochs": summary["completed_epochs"], "seconds": summary["seconds"],
       "macro_auc_all_12": summary["metrics"]["macro_auc_all_12"],
       "defined_classes": summary["metrics"]["defined_classes"], "synthetic": summary["synthetic"]})
assert (WORK / "baseline/best.pt").exists()
print("Save this notebook's outputs. Attach its output to the inference notebook.")
'''

INFER_RUN = '''from knee_ai.inference.predict import infer, load_model
from knee_ai.training.engine import resolve_device
from knee_ai.utils.kaggle import find_competition_root, find_image_root, find_checkpoint
from knee_ai.utils.reproducibility import configure_logging
configure_logging()
DATA = find_competition_root(INPUT, os.environ.get("KNEE_DATA_DIR"), training=False)
IMAGES = find_image_root(DATA, "test")
CHECKPOINT = find_checkpoint(INPUT, os.environ.get("KNEE_CHECKPOINT"))
device = "cuda" if torch.cuda.is_available() else "cpu"
inspected_model, inspected = load_model(CHECKPOINT, resolve_device("cpu"))
if inspected["config"]["synthetic"] and not SYNTHETIC:
    raise RuntimeError("Refusing a synthetic-fixture checkpoint for a real competition submission")
if inspected["config"].get("smoke_test", False):
    raise RuntimeError("Smoke checkpoints are diagnostic and cannot produce a competition submission")
del inspected_model, inspected
summary = infer(CHECKPOINT, DATA, IMAGES, WORK / "submission.csv", device=device,
                batch_size=2, num_workers=0, encode_chunk_size=32)
print(summary)
submission = pd.read_csv(WORK / "submission.csv")
print({"rows": len(submission), "columns": submission.columns.tolist(), "file": str(WORK / "submission.csv")})
'''

EXPERIMENT_RUN = '''from knee_ai.evaluation.experiment import compare_conditions
seeds = [42] if SYNTHETIC else [42, 43, 44]
comparison = compare_conditions(config, WORK / "slice_comparison", seeds=seeds, bootstrap_draws=200 if SYNTHETIC else 1000)
print(pd.read_csv(WORK / "slice_comparison/comparison.csv").to_string(index=False))
print(comparison["interpretation"])
'''


def embedded_source() -> tuple[str, str]:
    sources = {path.relative_to(ROOT / "src").as_posix(): path.read_text(encoding="utf-8")
               for path in sorted((ROOT / "src/knee_ai").rglob("*.py"))}
    raw = json.dumps(sources, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    payload = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
    cell = f'''# Generated embedded package; regenerate with scripts/build_notebooks.py after edits.
import base64, gzip, hashlib
raw = gzip.decompress(base64.b64decode({payload!r}))
assert hashlib.sha256(raw).hexdigest() == {digest!r}, "Embedded source checksum mismatch"
sources = json.loads(raw)
source_root = WORK / "knee_source"
for relative, source in sources.items():
    destination = source_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")
sys.path.insert(0, str(source_root))
print("Embedded source SHA256:", {digest!r})
'''
    return cell, digest


def main():
    embedded, digest = embedded_source()
    definitions = {
        "kaggle_train.ipynb": ("Kaggle GPU training: study-level knee MRI baseline",
            "Attach the official competition data, the Linux DICOM wheelhouse and the official ResNet18 weight dataset. "
            "Enable GPU, disable Internet. The notebook streams selected slices from train_series/ and trains only the 58 explicitly labelled studies. "
            "See docs/KAGGLE.md. Source is embedded; no code dataset is required. Real Kaggle execution has not yet been verified.",
            [TRAIN_CONFIG, TRAIN_RUN]),
        "kaggle_inference.ipynb": ("Offline Kaggle submission inference",
            "Attach the competition, DICOM wheelhouse and the output of kaggle_train. Enable GPU and disable Internet. "
            "Produces /kaggle/working/submission.csv with exactly the official study IDs and 12 probability columns. "
            "No external pretrained weights are needed; the checkpoint contains the complete encoder. Hidden-test runtime remains unmeasured.",
            [INFER_RUN]),
        "kaggle_slice_experiment.ipynb": ("Controlled slice-location experiment",
            "Same mounts as kaggle_train. Runs full-span uniform versus central physical-location sampling with equal series, slice counts, "
            "splits, initialization, optimizer and epoch budgets, using seeds 42/43/44. "
            "The central strategy is a spatial prior, not a validated anatomical locator. Results from 58 labelled studies will be preliminary.",
            [TRAIN_CONFIG, EXPERIMENT_RUN]),
        "kaggle_smoke.ipynb": ("Minimal mounted-MRI pipeline smoke test",
            "Attach the competition, ABI-matched Linux codec wheels and ResNet18 weights. Enable GPU and disable Internet. "
            "Run before full training: four training studies, two validation studies, one epoch, six 160px slices per study, "
            "batch one and a 15-minute budget checked between batches. Reloads the checkpoint and verifies diagnostic predictions. "
            "This checkpoint cannot be used for submission; its metrics do not estimate model performance.",
            ["SMOKE_TEST = True\n" + TRAIN_CONFIG,
             "from knee_ai.training.smoke import smoke_train\nresult = smoke_train(config)\nprint(result)\n"]),
    }
    destination = ROOT / "notebooks"
    destination.mkdir(exist_ok=True)
    for filename, (title, intro, extra) in definitions.items():
        notebook = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}"),
                                            nbf.v4.new_code_cell(SETUP), nbf.v4.new_code_cell(embedded),
                                            *[nbf.v4.new_code_cell(cell) for cell in extra]])
        notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12"},
                             "knee_ai_source_sha256": digest,
                             "kaggle": {"isInternetEnabled": False, "isGpuEnabled": True}}
        nbf.validate(notebook)
        nbf.write(notebook, destination / filename)
        print(destination / filename)


if __name__ == "__main__":
    main()

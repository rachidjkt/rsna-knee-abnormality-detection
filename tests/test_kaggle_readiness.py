import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from knee_ai.data.dataset import StudyDataset
from knee_ai.data.metadata import load_train, load_series
from knee_ai.inference.predict import infer
from knee_ai.models.multislice import MultiSliceClassifier
from knee_ai.training.engine import TrainConfig, predict_loader
from knee_ai.training.smoke import smoke_train

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("notebook_bootstrap", ROOT / "scripts/notebook_bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


@pytest.mark.parametrize("abi", ["cp312", "cp313"])
def test_real_wheel_metadata_dependency_preflight(abi):
    manifest = ROOT / "artifacts" / ("kaggle_wheels" if abi == "cp312" else "kaggle_wheels_cp313") / "wheel_manifest.json"
    if not manifest.exists():
        pytest.skip("Prepared wheelhouse not present")
    installed = {"numpy": "2.2.6", "typing-extensions": "4.14.0"}
    bootstrap.validate_wheelhouse(manifest, abi, installed, "Linux", "x86_64", {"python_version": abi[2:3] + "." + abi[3:]})
    with pytest.raises(RuntimeError, match="Incompatible offline dependency"):
        bootstrap.validate_wheelhouse(manifest, abi, {"numpy": "1.26.4"}, "Linux", "x86_64")
    with pytest.raises(RuntimeError, match="Linux"):
        bootstrap.validate_wheelhouse(manifest, abi, installed, "Windows", "AMD64")


def test_select_abi_with_multiple_wheelhouses(tmp_path):
    for abi in ("cp312", "cp313"):
        directory = tmp_path / abi
        directory.mkdir()
        (directory / "wheel_manifest.json").write_text(json.dumps({"python_abi": abi}))
    assert bootstrap.select_wheelhouse(tmp_path, "cp313").parent.name == "cp313"
    with pytest.raises(RuntimeError, match="found 0"):
        bootstrap.select_wheelhouse(tmp_path, "cp311")


def test_manifest_integrity_and_path_safety(tmp_path):
    path = tmp_path / "wheel_manifest.json"
    manifest = {"python_abi": "cp312", "wheels": {"../bad.whl": "bad"}}
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="Unsafe"):
        bootstrap.validate_wheelhouse(path, "cp312", {}, "Linux", "x86_64")
    manifest["wheels"] = {"bad.whl": "bad"}
    (tmp_path / "bad.whl").write_bytes(b"corrupt")
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="checksum"):
        bootstrap.validate_wheelhouse(path, "cp312", {}, "Linux", "x86_64")


def test_bounded_smoke_reloads_and_cannot_submit(fixtures, tmp_path):
    config = TrainConfig(data_dir=str(fixtures), image_root=str(fixtures / "train_images"),
        output_dir=str(tmp_path / "smoke"), split_path=str(tmp_path / "split.json"), encoder="tiny",
        device="cpu", image_size=32, slices_per_series=2, batch_size=1, epochs=1,
        synthetic=True, smoke_test=True, max_train_studies=4, max_val_studies=2)
    result = smoke_train(config)
    assert result["checkpoint_reload_predictions_match"]
    assert result["inference_studies"] == 2
    selection = json.loads((tmp_path / "smoke/run_studies.json").read_text())
    assert len(selection["train_ids"]) == 4 and not set(selection["train_ids"]) & set(selection["val_ids"])
    assert len(pd.read_csv(tmp_path / "smoke/smoke_predictions.csv")) == 2
    with pytest.raises(ValueError, match="diagnostic only"):
        infer(tmp_path / "smoke/best.pt", fixtures, fixtures / "test_images", tmp_path / "submission.csv", device="cpu")
    assert not (tmp_path / "submission.csv").exists()


def test_limits_require_explicit_smoke_flag():
    with pytest.raises(ValueError, match="smoke_test"):
        TrainConfig(max_train_studies=4)
    with pytest.raises(ValueError, match="one epoch"):
        smoke_train(TrainConfig(smoke_test=True))


def test_prediction_deadline(fixtures):
    data = StudyDataset(load_train(fixtures / "train.csv").iloc[:1], load_series(fixtures / "train_series.csv"),
                        fixtures / "train_images", slices_per_series=2, image_size=32)
    with pytest.raises(TimeoutError, match="budget exhausted"):
        predict_loader(MultiSliceClassifier("tiny"), DataLoader(data), torch.device("cpu"), deadline=time.perf_counter() - 1)

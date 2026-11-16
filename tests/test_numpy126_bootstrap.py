"""Regression coverage for the reported Kaggle NumPy-1.26 bootstrap failure."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / 'artifacts/kaggle_wheels/wheel_manifest.json').is_file(), reason='Optional local Linux wheel bundles are not distributed.')
spec = importlib.util.spec_from_file_location("numpy126_bootstrap", ROOT / "scripts/notebook_bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec); spec.loader.exec_module(bootstrap)


def test_reported_pyjpegls_failure_reproduced():
    manifest = ROOT / "artifacts/kaggle_wheels/wheel_manifest.json"
    with pytest.raises(RuntimeError, match=r"pyjpegls: requires numpy >=2.0, installed 1.26.4"):
        bootstrap.validate_wheelhouse(manifest, "cp312", {"numpy": "1.26.4"}, "Linux", "x86_64")


def test_removing_only_pyjpegls_does_not_fix_other_codecs(tmp_path):
    old = ROOT / "artifacts/kaggle_wheels/wheel_manifest.json"
    manifest = json.loads(old.read_text())
    manifest["wheels"] = {n: h for n, h in manifest["wheels"].items() if not n.startswith("pyjpegls-")}
    for name in manifest["wheels"]:
        shutil.copyfile(old.parent / name, tmp_path / name)
    path = tmp_path / "wheel_manifest.json"; path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match=r"pylibjpeg-libjpeg: requires numpy"):
        bootstrap.validate_wheelhouse(path, "cp312", {"numpy": "1.26.4"}, "Linux", "x86_64")


@pytest.mark.parametrize("abi", ["cp310", "cp311", "cp312"])
def test_new_linux_wheelhouse_accepts_numpy126_and_rejects_numpy2(abi):
    path = ROOT / f"artifacts/kaggle_support_numpy126/wheels/{abi}/wheel_manifest.json"
    manifest = bootstrap.validate_wheelhouse(path, abi, {"numpy": "1.26.4"}, "Linux", "x86_64")
    assert len(manifest["wheels"]) == 4
    assert set(m["name"] for m in manifest["package_metadata"].values()) == {
        "pydicom", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"}
    assert manifest["jpeg_ls_decoder"] == "pylibjpeg-libjpeg"
    assert manifest["rle_decoder"] == "pydicom (native)"
    with pytest.raises(RuntimeError, match="NumPy profile incompatible"):
        bootstrap.validate_wheelhouse(path, abi, {"numpy": "2.2.6"}, "Linux", "x86_64")


def test_selection_considers_numpy_and_abi(tmp_path):
    for name, abi, numpy in (("old", "cp312", ">=2,<3"), ("new", "cp312", ">=1.24,<2"), ("other", "cp311", ">=1.24,<2")):
        directory = tmp_path / name; directory.mkdir()
        (directory / "wheel_manifest.json").write_text(json.dumps({"python_abi": abi, "numpy_specifier": numpy}))
    assert bootstrap.select_wheelhouse(tmp_path, "cp312", numpy_version="1.26.4").parent.name == "new"
    assert bootstrap.select_wheelhouse(tmp_path, "cp312", numpy_version="2.2.6").parent.name == "old"
    with pytest.raises(RuntimeError, match="found 0"):
        bootstrap.select_wheelhouse(tmp_path, "cp312", explicit=tmp_path / "old", numpy_version="1.26.4")


def fake_manifest(tmp_path, name, requires_python=None):
    filename = f"{name}-1.0-py3-none-any.whl"
    meta = f"Name: {name}\nVersion: 1.0\n"
    if requires_python: meta += f"Requires-Python: {requires_python}\n"
    wheel = tmp_path / filename
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{name}-1.0.dist-info/METADATA", meta)
    manifest = tmp_path / "wheel_manifest.json"
    manifest.write_text(json.dumps({"python_abi": "cp312", "wheels": {filename: hashlib.sha256(wheel.read_bytes()).hexdigest()}}))
    return manifest


@pytest.mark.parametrize("name", ["numpy", "torch", "pandas", "scipy", "scikit_learn"])
def test_numerical_stack_wheels_are_forbidden(tmp_path, name):
    with pytest.raises(RuntimeError, match="Forbidden offline package"):
        bootstrap.validate_wheelhouse(fake_manifest(tmp_path, name), "cp312", {}, "Linux", "x86_64")


def test_requires_python_checked(tmp_path):
    with pytest.raises(RuntimeError, match="Requires-Python"):
        bootstrap.validate_wheelhouse(fake_manifest(tmp_path, "pydicom", ">=3.13"), "cp312", {}, "Linux", "x86_64")


def test_pip_receives_only_exact_verified_wheel_paths():
    path = ROOT / "artifacts/kaggle_support_numpy126/wheels/cp312/wheel_manifest.json"
    manifest = json.loads(path.read_text())
    command = bootstrap.install_command(path, manifest, "unused")
    assert "--no-index" in command and "--no-deps" in command
    assert "--find-links" not in command
    assert command[-4:] == [str(path.parent / name) for name in manifest["wheels"]]
    assert len(bootstrap.install_command(path, manifest, "unused", synthetic=True)[-1:]) == 1
    assert bootstrap.install_command(path, manifest, "unused", synthetic=True)[-1].endswith("pydicom-3.0.2-py3-none-any.whl")


def test_decoder_preflight_includes_libjpeg_jpeg_ls_and_native_rle():
    report = bootstrap.decoder_preflight()
    assert len(report) == 9
    for uid in ("1.2.840.10008.1.2.4.80", "1.2.840.10008.1.2.4.81"):
        assert report[uid]["required_plugin"] == "pylibjpeg"
    assert report["1.2.840.10008.1.2.5"]["required_plugin"] == "pydicom"


def test_every_notebook_uses_fixed_shared_bootstrap():
    expected = (ROOT / "scripts/notebook_bootstrap.py").read_text()
    for path in [*ROOT.glob("notebooks/*.ipynb"), *ROOT.glob("experiments_v2/notebooks/*.ipynb")]:
        notebook = nbformat.read(path, as_version=4)
        assert notebook.cells[1].source == expected


def test_isolated_numpy126_decoding_without_pyjpegls(tmp_path):
    interpreter = ROOT / ".tmp/numpy126_codec_env/Scripts/python.exe"
    if not interpreter.exists():
        pytest.skip("Prepare the isolated NumPy-1 codec environment described in docs/KAGGLE.md")
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_numpy126_codecs.py"),
                    "--python", str(interpreter), "--output", str(tmp_path / "codec_fixtures")], check=True, capture_output=True, text=True)
    report = json.loads((tmp_path / "codec_fixtures/verification.json").read_text())
    assert report["versions"]["numpy"] == "1.26.4" and not report["pyjpegls_installed"]
    assert len(report["fixtures"]) == 4 and all(row["passed"] for row in report["fixtures"])
    assert not report["genuine_competition_images"]


def test_upload_zip_hashes_and_no_numerical_stack_or_imaging():
    folder = ROOT / "artifacts/kaggle_support_numpy126"
    manifest = json.loads((folder / "support_manifest.json").read_text())
    with zipfile.ZipFile(ROOT / "artifacts/kaggle_support_numpy126.zip") as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == 20
        for name, item in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == item["sha256"]
            assert (folder / name).stat().st_size == item["bytes"]
        assert not any(name.endswith((".dcm", ".csv", ".pt")) for name in archive.namelist())
        assert not any(Path(name).name.startswith(("numpy-", "torch-", "pandas-", "pyjpegls-", "pylibjpeg_rle-")) for name in archive.namelist())


def test_all_prior_models_data_outputs_and_non_bootstrap_notebook_cells_preserved():
    snapshot = ROOT / "artifacts/pre_numpy126_fix_20261009"
    protected = json.loads((snapshot / "protected_files.json").read_text())
    with zipfile.ZipFile(snapshot / "project.zip") as archive:
        for name, digest in protected.items():
            current = (ROOT / name).read_bytes()
            if hashlib.sha256(current).hexdigest() == digest:
                continue
            assert name.endswith(".ipynb") and (name.startswith("notebooks/") or name.startswith("experiments_v2/notebooks/")), name
            original = archive.read(name)
            assert hashlib.sha256(original).hexdigest() == digest
            old = nbformat.reads(original.decode(), as_version=4)
            new = nbformat.reads(current.decode(), as_version=4)
            assert old.metadata == new.metadata
            assert len(old.cells) == len(new.cells)
            assert [c.source for i, c in enumerate(old.cells) if i != 1] == [c.source for i, c in enumerate(new.cells) if i != 1]


def test_pinned_requirement_file_matches_built_profiles():
    from importlib import util
    spec = util.spec_from_file_location("numpy126_wheel_builder", ROOT / "scripts/prepare_kaggle_wheels.py")
    builder = util.module_from_spec(spec); spec.loader.exec_module(builder)
    requirements = [line.strip() for line in (ROOT / "requirements-kaggle-numpy126.txt").read_text().splitlines()
                    if line.strip() and not line.startswith("#")]
    assert requirements == builder.PROFILES["numpy126"]["packages"]

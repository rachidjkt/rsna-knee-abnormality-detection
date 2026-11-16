"""Dual NumPy/ABI discovery, runtime provenance and actual Linux codec regressions."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts/kaggle_support_dual_numpy_20261009"
pytestmark = pytest.mark.skipif(not BUNDLE.is_dir(), reason='Optional local Linux wheel bundle is not distributed.')
spec = importlib.util.spec_from_file_location("dual_numpy_bootstrap", ROOT / "scripts/notebook_bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec); spec.loader.exec_module(bootstrap)


class MetadataDistribution:
    def __init__(self, name, version, location):
        self.metadata = {"Name": name}; self.version = version; self.location = location
        self._path = f"{location}/{name}-{version}.dist-info"

    def locate_file(self, path):
        return self.location


@pytest.mark.parametrize("abi,numpy,profile", [
    ("cp310", "1.26.4", "numpy126"), ("cp311", "1.26.4", "numpy126"), ("cp312", "1.26.4", "numpy126"),
    ("cp310", "2.1.3", "numpy2"), ("cp311", "2.1.3", "numpy2"), ("cp312", "2.1.3", "numpy2"), ("cp313", "2.1.3", "numpy2"),
])
def test_combined_support_selects_exactly_one_valid_profile(abi, numpy, profile):
    path = bootstrap.select_wheelhouse(BUNDLE, abi, numpy_version=numpy)
    assert path == BUNDLE / f"wheels/{profile}/{abi}/wheel_manifest.json"
    manifest = bootstrap.validate_wheelhouse(path, abi, {"numpy": numpy}, "Linux", "x86_64",
                {"python_full_version": "3.13.15"} if abi == "cp313" else None)
    assert len(manifest["wheels"]) == 4 and manifest["profile"] == f"{profile}_decode_only"


def test_reported_cp313_numpy2_missing_profile_reproduced():
    with pytest.raises(RuntimeError, match=r"kernel ABI cp313, NumPy 2.1.3; found 0"):
        bootstrap.select_wheelhouse(ROOT / "artifacts/kaggle_support_numpy126", "cp313", numpy_version="2.1.3")


@pytest.mark.parametrize("abi,numpy", [("cp313", "1.26.4"), ("cp314", "2.1.3"), ("cp312", "3.0.0")])
def test_unsupported_environment_fails_without_stack_changes(abi, numpy):
    with pytest.raises(RuntimeError, match="found 0"):
        bootstrap.select_wheelhouse(BUNDLE, abi, numpy_version=numpy)


def test_ambiguous_manifests_fail_and_explicit_override_is_checked(tmp_path):
    for name in ("one", "two"):
        directory = tmp_path / name; directory.mkdir()
        (directory / "wheel_manifest.json").write_text(json.dumps({"python_abi": "cp313", "numpy_specifier": ">=2,<3"}))
    with pytest.raises(RuntimeError, match="found 2"):
        bootstrap.select_wheelhouse(tmp_path, "cp313", numpy_version="2.1.3")
    assert bootstrap.select_wheelhouse(tmp_path, "cp313", explicit=tmp_path / "one", numpy_version="2.1.3").parent.name == "one"
    with pytest.raises(RuntimeError, match="found 0"):
        bootstrap.select_wheelhouse(tmp_path, "cp313", explicit=tmp_path / "one", numpy_version="1.26.4")


def test_imported_numpy_wins_over_stale_or_duplicate_distribution_metadata():
    module = SimpleNamespace(__version__="2.1.3", __file__="/usr/local/lib/python3.13/dist-packages/numpy/__init__.py")
    distributions = [MetadataDistribution("numpy", "2.1.3", "/usr/local/lib/python3.13/dist-packages"),
                     MetadataDistribution("numpy", "1.26.4", "/example/stale-dist-info")]
    # Reproduce the original metadata-only comprehension's last-record-wins bug.
    assert {d.metadata["Name"]: d.version for d in distributions}["numpy"] == "1.26.4"
    installed, runtime = bootstrap.runtime_dependencies({"numpy": module}, distributions)
    assert installed["numpy"] == runtime["numpy"]["imported_version"] == "2.1.3"
    assert runtime["numpy"]["imported_path"] == module.__file__
    assert not runtime["numpy"]["metadata_matches_import"]
    assert [r["version"] for r in runtime["numpy"]["metadata_distributions"]] == ["2.1.3", "1.26.4"]
    path = bootstrap.select_wheelhouse(BUNDLE, "cp313", numpy_version=module.__version__)
    bootstrap.validate_wheelhouse(path, "cp313", installed, "Linux", "x86_64", {"python_full_version": "3.13.15"})


def test_all_imported_core_versions_override_metadata_including_scipy():
    modules = {name: SimpleNamespace(__version__="2.1.3", __file__=f"/runtime/{name}/__init__.py") for name in bootstrap.CORE_DISTRIBUTIONS}
    distributions = [MetadataDistribution(dist, "1.0.0", "/stale") for dist in bootstrap.CORE_DISTRIBUTIONS.values()]
    installed, runtime = bootstrap.runtime_dependencies(modules, distributions)
    assert all(installed[dist] == "2.1.3" for dist in bootstrap.CORE_DISTRIBUTIONS.values())
    assert "scipy" in runtime and "sklearn" in runtime
    assert all(d["dependency_version_source"] == "imported module" for d in runtime.values())


def test_runtime_report_written_even_when_selection_fails(tmp_path, monkeypatch):
    modules = {name: SimpleNamespace(__version__="2.1.3", __file__=f"/runtime/{name}/__init__.py") for name in bootstrap.CORE_DISTRIBUTIONS}
    modules["torch"].amp = SimpleNamespace(GradScaler=lambda *args, **kwargs: None)
    original = bootstrap.importlib.import_module
    monkeypatch.setattr(bootstrap.importlib, "import_module", lambda name, *args: modules[name] if name in modules else original(name, *args))
    monkeypatch.setattr(bootstrap.importlib.metadata, "distributions", lambda: [])
    monkeypatch.setenv("KNEE_SYNTHETIC", "0")
    monkeypatch.setenv("KNEE_INPUT_DIR", str(tmp_path / "empty_inputs"))
    monkeypatch.setenv("KNEE_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.delenv("KNEE_WHEELHOUSE", raising=False)
    with pytest.raises(RuntimeError, match="found 0"):
        bootstrap.bootstrap()
    report = json.loads((tmp_path / "work/environment_runtime.json").read_text())
    assert report["selection_uses_imported_numpy"] and report["core"]["numpy"]["imported_version"] == "2.1.3"
    assert not (tmp_path / "work/offline_deps").exists()


def test_numpy2_wheel_hash_tampering_is_rejected(tmp_path):
    source = BUNDLE / "wheels/numpy2/cp313"
    manifest = json.loads((source / "wheel_manifest.json").read_text())
    for name in manifest["wheels"]:
        shutil.copyfile(source / name, tmp_path / name)
    name = next(n for n in manifest["wheels"] if n.startswith("pylibjpeg_libjpeg-"))
    with (tmp_path / name).open("ab") as file: file.write(b"tampered")
    path = tmp_path / "wheel_manifest.json"; path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        bootstrap.validate_wheelhouse(path, "cp313", {"numpy": "2.1.3"}, "Linux", "x86_64")


def test_no_original_or_combined_codec_wheel_bundles_numpy_metadata():
    manifests = [ROOT / "artifacts/kaggle_wheels/wheel_manifest.json", ROOT / "artifacts/kaggle_wheels_cp313/wheel_manifest.json",
                 *BUNDLE.rglob("wheel_manifest.json")]
    for path in manifests:
        for name in json.loads(path.read_text())["wheels"]:
            with zipfile.ZipFile(path.parent / name) as archive:
                assert not any(n.lower().startswith(("numpy/", "numpy-", "numpy.libs/")) for n in archive.namelist())


def test_dual_support_zip_inventory_and_hashes():
    manifest = json.loads((BUNDLE / "support_manifest.json").read_text())
    assert len(manifest["profiles"]) == 7
    assert manifest["prioritized_user_reported_runtime"]["imported_numpy"] == "2.1.3"
    with zipfile.ZipFile(BUNDLE.with_suffix(".zip")) as archive:
        assert archive.testzip() is None and len(archive.namelist()) == 41
        for name, item in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == item["sha256"]
            assert hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == item["sha256"]
        assert not any(Path(n).name.startswith(("numpy-", "torch-", "pandas-", "scipy-", "scikit_learn-")) for n in archive.namelist())
        assert not any(n.endswith((".dcm", ".csv", ".pt")) for n in archive.namelist())


def test_dual_fix_preserves_all_prior_models_results_and_support_assets():
    snapshot = ROOT / "artifacts/pre_dual_numpy_fix_20261009"
    with zipfile.ZipFile(snapshot / "project.zip") as archive:
        for name, digest in json.loads((snapshot / "protected_files.json").read_text()).items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest: continue
            assert name.endswith(".ipynb") and name.startswith(("notebooks/", "experiments_v2/notebooks/")), name
            original = archive.read(name)
            assert hashlib.sha256(original).hexdigest() == digest
            old = nbformat.reads(original.decode(), as_version=4); new = nbformat.read(ROOT / name, as_version=4)
            assert old.metadata == new.metadata and len(old.cells) == len(new.cells)
            assert [c.source for i, c in enumerate(old.cells) if i != 1] == [c.source for i, c in enumerate(new.cells) if i != 1]


def test_numpy2_requirements_match_profile_pins():
    requirements = [n for n in (ROOT / "requirements-kaggle-numpy2.txt").read_text().splitlines() if n and not n.startswith("#")]
    assert requirements == json.loads((BUNDLE / "wheels/numpy2/cp313/wheel_manifest.json").read_text())["packages"]


@pytest.mark.parametrize("profile,numpy,abi,python", [("numpy126", "1.26.4", "cp312", "3.12.3"), ("numpy2", "2.1.3", "cp313", "3.13.15")])
def test_linux_actual_binary_profile_install_and_synthetic_decode(tmp_path, profile, numpy, abi, python):
    if not (ROOT / f".tmp/linux_codec_{profile}/pyvenv.cfg").exists() or not shutil.which("wsl"):
        pytest.skip("Prepare the isolated WSL environments described in docs/KAGGLE.md")
    def linux(path):
        absolute = Path(path).resolve().as_posix()
        return "/mnt/" + absolute[0].lower() + absolute[2:]
    command = ["wsl", "-d", "Ubuntu-24.04", "--", linux(ROOT / f".tmp/linux_codec_{profile}/bin") + "/python",
               linux(ROOT / "scripts/verify_linux_codec_profile.py"), "--wheelhouse", linux(BUNDLE / f"wheels/{profile}/{abi}"),
               "--fixtures", linux(ROOT / "outputs/numpy126_codec_fixtures"), "--output", linux(tmp_path / profile),
               "--expected-numpy", numpy, "--expected-python", python]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    report = json.loads((tmp_path / profile / "verification.json").read_text())
    assert report["numpy"] == numpy and report["python_abi"] == abi and report["numpy_preserved"]
    assert not report["pyjpegls_installed"] and not report["genuine_mri"]
    assert len(report["decoders"]) == 9 and all(f["passed"] for f in report["fixtures"])

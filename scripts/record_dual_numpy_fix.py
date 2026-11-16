"""Record the dual-runtime correction independently without replacing prior evidence."""
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    output = ROOT / "reports/dual_numpy_verification.json"
    if output.exists():
        raise RuntimeError("Evidence exists; preserve it")
    snapshot = ROOT / "artifacts/pre_dual_numpy_fix_20261009"
    protected = read("artifacts/pre_dual_numpy_fix_20261009/protected_files.json")
    notebooks = []
    with zipfile.ZipFile(snapshot / "project.zip") as archive:
        for name, digest in protected.items():
            current = ROOT / name
            if sha(current) == digest: continue
            assert name.endswith(".ipynb") and name.startswith(("notebooks/", "experiments_v2/notebooks/")), name
            original = archive.read(name); assert hashlib.sha256(original).hexdigest() == digest
            old = nbformat.reads(original.decode(), as_version=4); new = nbformat.read(current, as_version=4)
            assert old.metadata == new.metadata and len(old.cells) == len(new.cells)
            assert [c.source for i, c in enumerate(old.cells) if i != 1] == [c.source for i, c in enumerate(new.cells) if i != 1]
            assert new.cells[1].source == (ROOT / "scripts/notebook_bootstrap.py").read_text()
            notebooks.append({"path": name, "before_sha256": digest, "after_sha256": sha(current),
                              "original_archived": True, "changed_cell": "shared bootstrap only"})
        previous = [n for n in archive.namelist() if n.startswith("reports/") and n not in {
            "reports/TECHNICAL_REPORT.md", "reports/VERIFICATION.md", "reports/KAGGLE_READINESS_AUDIT.md"}]
        for name in previous:
            assert sha(ROOT / name) == hashlib.sha256(archive.read(name)).hexdigest(), name
    suite = ET.parse(ROOT / "reports/dual_numpy_tests.xml").getroot().find("testsuite")
    tests = {k: int(suite.attrib[k]) for k in ("tests", "failures", "errors", "skipped")}
    assert tests["tests"] == 114 and tests["failures"] == tests["errors"] == tests["skipped"] == 0
    linux = {}
    for name, python, numpy in (("linux313", "3.13.15", "2.1.3"), ("linux126", "3.12.3", "1.26.4")):
        report = read(f"outputs/dual_numpy_{name}_fixtures/verification.json")
        assert report["python"] == python and report["numpy"] == numpy
        assert report["numpy_preserved"] and not report["genuine_mri"]
        assert len(report["decoders"]) == 9 and all(f["passed"] for f in report["fixtures"])
        linux[name] = report
    notebook_runs = {name: read(f"outputs/dual_numpy_{name}_notebooks/verification.json") for name in ("v1", "v2")}
    for name, report in notebook_runs.items():
        assert report["synthetic"] and report["networking"] == "blocked" and not report["kaggle_remote_execution"]
        assert len([k for k in report if k.endswith("cells_seconds")]) == (4 if name == "v1" else 2)
    runtime = read("outputs/dual_numpy_v1_notebooks/train/environment_runtime.json")
    bundle = ROOT / "artifacts/kaggle_support_dual_numpy_20261009"
    support = read("artifacts/kaggle_support_dual_numpy_20261009/support_manifest.json")
    for name, entry in support["files"].items():
        assert sha(bundle / name) == entry["sha256"]
    zip_path = bundle.with_suffix(".zip")
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        for name, entry in support["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == entry["sha256"]
    with zipfile.ZipFile(ROOT / "artifacts/pre_numpy126_fix_20261009/project.zip") as archive:
        original_bootstrap = archive.read("scripts/notebook_bootstrap.py")
    assert b'installed = {d.metadata["Name"]: d.version' in original_bootstrap
    assert b'installed["numpy"]' not in original_bootstrap
    result = {"date": "2026-10-09", "prioritized_runtime": support["prioritized_user_reported_runtime"],
              "runtime_provenance": "user-verified Kaggle values; no remote action by this session",
              "prior_numpy126_error": {"value_source": "distribution metadata, not imported module",
                  "original_bootstrap_sha256": hashlib.sha256(original_bootstrap).hexdigest(),
                  "prior_genuinely_imported_version": "unknown; original imported version/path not supplied",
                  "numpy_distribution_bundled_in_codec_wheels": False,
                  "cause_of_prior_metadata_value": "not established; stale/duplicate records or another kernel remain possible"},
              "tests": tests, "junit_seconds": float(suite.attrib["time"]),
              "preservation": {"protected_files": len(protected), "unchanged_files": len(protected) - len(notebooks),
                               "notebooks": notebooks, "historical_reports_unchanged": len(previous),
                               "archive_sha256": sha(snapshot / "project.zip")},
              "linux_fixture_runs": linux, "synthetic_notebook_runs": notebook_runs,
              "local_runtime_numpy_provenance": runtime["core"]["numpy"],
              "linux_wheel_manifests": {p["manifest"]: read("artifacts/kaggle_support_dual_numpy_20261009/" + p["manifest"]) for p in support["profiles"]},
              "bootstrap_sha256": sha(ROOT / "scripts/notebook_bootstrap.py"),
              "upload": {"path": zip_path.relative_to(ROOT).as_posix(), "bytes": zip_path.stat().st_size,
                         "sha256": sha(zip_path), "files": len(support["files"]) + 1},
              "genuine_mri_decoding": "not verified; updated smoke must pass on Kaggle",
              "kaggle_stack_execution": "not executed by this session",
              "real_model_performance": "not measured"}
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    files = [ROOT / p for p in (
        "scripts/notebook_bootstrap.py", "scripts/prepare_kaggle_wheels.py", "scripts/package_dual_numpy_support.py",
        "scripts/prepare_linux_codec_tests.py", "scripts/verify_linux_codec_profile.py", "scripts/record_dual_numpy_fix.py",
        "tests/test_dual_numpy_bootstrap.py", "requirements-kaggle-numpy2.txt",
        "docs/KAGGLE.md", "RESUME.md", "PROGRESS.md", "TASKS.md", "experiments_v2/README.md",
        "reports/DUAL_NUMPY_KAGGLE_FIX.md", "reports/TECHNICAL_REPORT.md", "reports/VERIFICATION.md",
        "reports/KAGGLE_READINESS_AUDIT.md", "reports/dual_numpy_tests.xml", "reports/dual_numpy_verification.json")]
    files += sorted(ROOT.glob("notebooks/*.ipynb")) + sorted(ROOT.glob("experiments_v2/notebooks/*.ipynb"))
    (ROOT / "reports/dual_numpy_manifest.json").write_text(json.dumps({
        p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in files}, indent=2))
    print({"tests": tests, "protected_files": len(protected), "regenerated_bootstrap_cells": len(notebooks),
           "linux_cp313_numpy213_fixtures_passed": True, "real_mri_verified": False})


if __name__ == "__main__":
    main()

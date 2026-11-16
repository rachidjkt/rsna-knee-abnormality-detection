"""Record measured compatibility evidence separately; preserve old verification results."""
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    return json.loads((ROOT / relative).read_text())


def main():
    output = ROOT / "reports/numpy126_verification.json"
    if output.exists():
        raise RuntimeError("Evidence already exists; do not replace previous results")
    snapshot = ROOT / "artifacts/pre_numpy126_fix_20261009"
    protected = read("artifacts/pre_numpy126_fix_20261009/protected_files.json")
    regenerated = []
    with zipfile.ZipFile(snapshot / "project.zip") as archive:
        for name, digest in protected.items():
            current = ROOT / name
            if sha(current) == digest:
                continue
            if not name.endswith(".ipynb") or not name.startswith(("notebooks/", "experiments_v2/notebooks/")):
                raise RuntimeError(f"Protected model/data/result changed: {name}")
            before = archive.read(name)
            assert hashlib.sha256(before).hexdigest() == digest
            old = nbformat.reads(before.decode(), as_version=4)
            new = nbformat.read(current, as_version=4)
            assert old.metadata == new.metadata and len(old.cells) == len(new.cells)
            assert [c.source for i, c in enumerate(old.cells) if i != 1] == [c.source for i, c in enumerate(new.cells) if i != 1]
            assert new.cells[1].source == (ROOT / "scripts/notebook_bootstrap.py").read_text()
            regenerated.append({"path": name, "before_sha256": digest, "after_sha256": sha(current),
                                "changed_cell": "shared bootstrap only", "original_bytes_archived": True})
        # Historical scientific/verification records remain intact. Only the
        # living technical report and verification index receive appendices.
        prior_reports = [n for n in archive.namelist() if n.startswith("reports/") and n not in {
            "reports/TECHNICAL_REPORT.md", "reports/VERIFICATION.md", "reports/KAGGLE_READINESS_AUDIT.md"}]
        for name in prior_reports:
            assert sha(ROOT / name) == hashlib.sha256(archive.read(name)).hexdigest(), name
    suite = ET.parse(ROOT / "reports/numpy126_tests.xml").getroot().find("testsuite")
    tests = {k: int(suite.attrib[k]) for k in ("tests", "failures", "errors", "skipped")}
    assert tests["failures"] == tests["errors"] == tests["skipped"] == 0
    fixtures = read("outputs/numpy126_codec_fixtures/verification.json")
    assert fixtures["versions"]["numpy"] == "1.26.4" and not fixtures["pyjpegls_installed"]
    assert all(f["passed"] for f in fixtures["fixtures"]) and not fixtures["genuine_competition_images"]
    v1 = read("outputs/numpy126_v1_notebooks/verification.json")
    v2 = read("outputs/numpy126_v2_notebooks/verification.json")
    assert v1["networking"] == v2["networking"] == "blocked"
    assert v1["synthetic"] and v2["synthetic"]
    assert len([k for k in v1 if k.endswith("cells_seconds")]) == 4
    assert len([k for k in v2 if k.endswith("cells_seconds")]) == 2
    bundle = ROOT / "artifacts/kaggle_support_numpy126"
    support = read("artifacts/kaggle_support_numpy126/support_manifest.json")
    for name, entry in support["files"].items():
        assert sha(bundle / name) == entry["sha256"], name
    result = {"date": "2026-10-09", "reported_kaggle_failure": "pyjpegls requires numpy >=2.0; installed 1.26.4",
              "tests": tests, "junit_seconds": float(suite.attrib["time"]),
              "preservation": {"protected_files": len(protected), "unchanged_files": len(protected) - len(regenerated),
                               "regenerated_notebooks": regenerated, "historical_reports_unchanged": len(prior_reports),
                               "archive_sha256": sha(snapshot / "project.zip")},
              "bootstrap_sha256": sha(ROOT / "scripts/notebook_bootstrap.py"),
              "isolated_numpy126_windows_fixtures": fixtures,
              "v1_synthetic_notebook_execution": v1, "v2_synthetic_notebook_execution": v2,
              "linux_wheel_manifests": {abi: read(f"artifacts/kaggle_support_numpy126/wheels/{abi}/wheel_manifest.json")
                                       for abi in ("cp310", "cp311", "cp312")},
              "upload_zip": {"path": "artifacts/kaggle_support_numpy126.zip", "bytes": (ROOT / "artifacts/kaggle_support_numpy126.zip").stat().st_size,
                             "sha256": sha(ROOT / "artifacts/kaggle_support_numpy126.zip")},
              "genuine_mri_decoding": "not verified; rerun regenerated notebook on Kaggle",
              "linux_binary_execution": "not executed on this Windows workstation",
              "real_model_performance": "not measured", "kaggle_actions_by_this_session": "none"}
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    paths = [ROOT / p for p in ("scripts/notebook_bootstrap.py", "scripts/prepare_kaggle_wheels.py", "scripts/package_numpy126_support.py",
                               "scripts/verify_numpy126_codecs.py", "scripts/record_numpy126_fix.py", "tests/test_numpy126_bootstrap.py",
                               "tests/test_v2.py", "requirements-kaggle-numpy126.txt", "docs/KAGGLE.md", "PROGRESS.md", "RESUME.md", "TASKS.md", "experiments_v2/README.md",
                               "reports/NUMPY126_CODEC_FIX.md", "reports/TECHNICAL_REPORT.md", "reports/VERIFICATION.md", "reports/KAGGLE_READINESS_AUDIT.md",
                               "reports/numpy126_tests.xml", "reports/numpy126_verification.json")]
    paths += sorted(ROOT.glob("notebooks/*.ipynb")) + sorted(ROOT.glob("experiments_v2/notebooks/*.ipynb"))
    (ROOT / "reports/numpy126_manifest.json").write_text(json.dumps({p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size,
                                                                    "sha256": sha(p)} for p in paths}, indent=2))
    print({"tests": tests, "protected_files": len(protected), "notebooks_regenerated_bootstrap_only": len(regenerated),
           "genuine_mri_decoding_verified": False})


if __name__ == "__main__":
    main()

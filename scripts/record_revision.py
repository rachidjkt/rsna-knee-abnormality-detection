"""Record this audit's measured evidence without replacing prior verification artifacts."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    saved = read("artifacts/pre_label_audit_20261009/preserved_input_hashes.json")
    for name, digest in saved.items():
        path = Path("experiments/study_split.json") if name == "study_split.json" else Path("data") / name
        if sha(path) != digest:
            raise ValueError(f"Preserved input changed: {name}")
    suite = ET.parse("reports/revision_tests.xml").getroot().find("testsuite")
    notebook_check = read("outputs/final_audit_notebooks/verification.json")
    current = read("notebooks/kaggle_train.ipynb")["metadata"]["knee_ai_source_sha256"]
    sources = {p.relative_to(Path("src")).as_posix(): p.read_text(encoding="utf-8") for p in sorted(Path("src/knee_ai").rglob("*.py"))}
    if hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest() != current:
        raise ValueError("Regenerate notebooks from current source")
    if notebook_check["source_sha256"] != current:
        raise ValueError("Verification must execute the shipped source")
    result = {"date": "2026-10-09", "tests": {k: int(suite.attrib[k]) for k in ["tests", "failures", "errors", "skipped"]},
              "junit_seconds": float(suite.attrib["time"]), "preserved_official_input_hashes": saved,
              "preservation_archive_sha256": sha("artifacts/pre_label_audit_20261009/project.zip"),
              "labels": read("reports/label_availability.json"), "gpu_tensor_profile": read("reports/gpu_profile.json"),
              "notebook_synthetic_execution": notebook_check,
              "real_shape_synthetic_smoke": read("outputs/real_shape_fixture_smoke/smoke_diagnostics.json"),
              "synthetic_end_to_end": read("outputs/label_audit_smoke/smoke_results.json"),
              "real_mri_training": "not executed", "kaggle_remote_execution": "not executed",
              "linux_wheel_binary_installation": "not executed", "hidden_test_runtime": "not measured",
              "comparison": "Previous synthetic controlled comparison preserved; no repeat needed for unchanged experiment logic."}
    Path("reports/revision_verification.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    files = [p for directory in ["src", "scripts", "tests", "configs", "notebooks", "docs", "reports"]
             for p in Path(directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and p.suffix in {".py", ".json", ".md", ".ipynb", ".png", ".xml", ".csv"} and p.name != "revision_manifest.json"]
    files += [Path(n) for n in ["README.md", "AGENTS.md", "RESUME.md", "PROGRESS.md", "TASKS.md", ".gitignore", "pyproject.toml"]]
    manifest = {p.as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(files)}
    Path("reports/revision_manifest.json").write_text(json.dumps(manifest, indent=2))
    print({"tests": result["tests"], "preserved_csvs_and_split": True, "source_sha256": current, "manifest_files": len(manifest)})


if __name__ == "__main__":
    main()

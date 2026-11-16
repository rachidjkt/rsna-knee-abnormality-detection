"""Aggregate actual run artifacts into a portable, non-patient verification record."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    xml = ET.parse("reports/tests.xml")
    suite = xml.getroot().find("testsuite")
    notebooks = read("outputs/handoff_notebooks/verification.json")
    current = read("notebooks/kaggle_train.ipynb")["metadata"]["knee_ai_source_sha256"]
    if notebooks["source_sha256"] != current:
        raise ValueError("Notebook verification must match the shipped embedded source")
    result = {"date": "2026-10-09", "timezone": "America/Toronto",
              "tests": {key: int(suite.attrib[key]) for key in ["tests", "failures", "errors", "skipped"]},
              "test_seconds": float(suite.attrib["time"]),
              "preprocessing_version": "geometry-fixed-reference-percentile-letterbox-v2",
              "cpu_synthetic": read("outputs/verified_cpu/summary.json"),
              "gpu_synthetic": read("outputs/verified_gpu/summary.json"),
              "gpu_inference_synthetic": read("outputs/verified_gpu/submission.diagnostics.json"),
              "cpu_controlled_synthetic": read("outputs/verified_comparison/comparison.json"),
              "pretrained_notebook_synthetic": notebooks,
              "pretrained_notebook_training": read("outputs/handoff_notebooks/train/baseline/summary.json"),
              "real_mri_training": "not executed", "real_slice_experiment": "not executed",
              "kaggle_remote_execution": "not executed", "kaggle_submission": "not made",
              "warnings": ["All model metrics here are synthetic engineering measurements, not medical performance.",
                           "Linux wheelhouse installation and hidden-test runtime remain unverified."]}
    Path("reports/verification_results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    environment = read("reports/environment.json")
    lines = ["# Direct dependency pins from the tested Windows Python 3.12 environment.",
             "# CUDA wheel index is required only when reconstructing these specific local torch builds.",
             "# This is not a Kaggle install recipe or a complete transitive lock.",
             "--extra-index-url https://download.pytorch.org/whl/cu128"]
    lines += [f"{name}=={version}" for name, version in environment["dependencies"].items() if version]
    Path("requirements-tested.txt").write_text("\n".join(lines) + "\n")
    files = [path for directory in ["src", "scripts", "tests", "configs", "notebooks", "docs", "reports"]
             for path in Path(directory).rglob("*") if path.is_file() and "__pycache__" not in path.parts
             and path.suffix in [".py", ".json", ".md", ".ipynb", ".png", ".xml"]
             and path.name != "FILE_MANIFEST.json"]
    files += [Path(name) for name in ["README.md", "AGENTS.md", "PROGRESS.md", "TASKS.md", "RESUME.md", ".gitignore", "pyproject.toml", "requirements.txt", "requirements-tested.txt"]]
    manifest = {path.as_posix(): {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in sorted(files)}
    Path("reports/FILE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print({"tests": result["tests"], "files": len(manifest), "source_sha256": current})


if __name__ == "__main__":
    main()

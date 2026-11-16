"""Package only source, configs and notebooks; never include data or credentials."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/kaggle_bundle")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    paths = [p for p in (root / "src").rglob("*.py")]
    paths += list((root / "configs").glob("*.json")) + list((root / "notebooks").glob("*.ipynb"))
    paths += list((root / "scripts").glob("*.py")) + list((root / "tests").glob("*.py"))
    paths += list((root / "docs").glob("*.md")) + list((root / "reports").glob("*.md"))
    paths += list((root / "reports/figures").glob("*.png"))
    paths += [root / "reports" / name for name in ["label_availability.csv", "label_availability.json", "gpu_profile.json",
              "revision_tests.xml", "revision_verification.json", "revision_manifest.json"] if (root / "reports" / name).exists()]
    paths += [root / name for name in ["pyproject.toml", "requirements.txt", "requirements-tested.txt", "README.md", "AGENTS.md", "PROGRESS.md", "TASKS.md", "RESUME.md"]]
    manifest = {}
    with zipfile.ZipFile(output / "knee_ai_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(root).as_posix())
            manifest[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for notebook in (root / "notebooks").glob("*.ipynb"):
        shutil.copy2(notebook, output / notebook.name)
    print(f"Bundle prepared at {output}; no competition metadata or images included.")


if __name__ == "__main__":
    main()

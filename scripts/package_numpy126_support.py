"""Create a new, validated software/weights-only Kaggle upload ZIP; preserve old assets."""
import hashlib
import importlib.util
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    bundle = ROOT / "artifacts/kaggle_support_numpy126"
    archive_path = ROOT / "artifacts/kaggle_support_numpy126.zip"
    if archive_path.exists():
        raise RuntimeError("Upload ZIP already exists; preserve it and use a fresh versioned destination")
    spec = importlib.util.spec_from_file_location("support_bootstrap", ROOT / "scripts/notebook_bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec); spec.loader.exec_module(bootstrap)
    for abi in ("cp310", "cp311", "cp312"):
        bootstrap.validate_wheelhouse(bundle / f"wheels/{abi}/wheel_manifest.json", abi,
                                    {"numpy": "1.26.4"}, "Linux", "x86_64")
    weights = ROOT / "artifacts/pretrained/resnet18-f37072fd.pth"
    expected = json.loads((weights.parent / "weights_manifest.json").read_text())["sha256"]
    if hashlib.sha256(weights.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Original pretrained weights checksum mismatch")
    (bundle / "pretrained").mkdir(exist_ok=True)
    for name in (weights.name, "weights_manifest.json"):
        shutil.copyfile(weights.parent / name, bundle / "pretrained" / name)
    shutil.copyfile(ROOT / "requirements-kaggle-numpy126.txt", bundle / "requirements-kaggle-numpy126.txt")
    (bundle / "UPLOAD_README.md").write_text(
        "# NumPy 1.26 Kaggle support\n\n"
        "Upload artifacts/kaggle_support_numpy126.zip as a NEW PRIVATE Kaggle dataset named knee-support-numpy126.\n"
        "Keep the extracted wheels/cp310, cp311, cp312 subfolders and pretrained subfolder.\n"
        "Attach the official competition and this dataset to the regenerated notebooks/kaggle_smoke.ipynb.\n"
        "Detach previous codec/weight support datasets from the NEW notebook; leave their dataset versions intact.\n"
        "GPU on (T4 or P100), Internet off; fresh session; Save Version / Save & Run All.\n"
        "No pip upgrade or base numerical-stack installation is needed. Python ABI selection is automatic.\n"
        "JPEG-LS uses libjpeg 2.1.0; JPEG 2000 uses openjpeg 2.2.1; RLE uses native pydicom.\n"
        "No pyjpegls, RLE plugin, NumPy, torch, pandas or sklearn wheels are included.\n"
        "Local fixture verification is not evidence of successful competition MRI decoding.\n"
        "See docs/KAGGLE.md in the project for exact success checks and troubleshooting.\n", encoding="utf-8")
    files = sorted(p for p in bundle.rglob("*") if p.is_file() and p.name != "support_manifest.json")
    allowed = {".whl", ".json", ".pth", ".txt", ".md"}
    if any(p.suffix not in allowed for p in files):
        raise RuntimeError("Unexpected artifact in upload folder")
    manifest = {"purpose": "software and public weights only; no competition images, metadata or credentials",
                "numpy_version": "1.26.4", "python_abis": ["cp310", "cp311", "cp312"],
                "files": {p.relative_to(bundle).as_posix(): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in files}}
    (bundle / "support_manifest.json").write_text(json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(bundle).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for relative, entry in manifest["files"].items():
            assert hashlib.sha256(archive.read(relative)).hexdigest() == entry["sha256"], relative
    print(json.dumps({"zip": str(archive_path), "bytes": archive_path.stat().st_size,
                      "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(), "files": len(manifest["files"]) + 1}, indent=2))


if __name__ == "__main__":
    main()

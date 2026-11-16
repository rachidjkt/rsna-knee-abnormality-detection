"""Package seven ABI/NumPy codec profiles and public weights; preserve old support ZIPs."""
import argparse
import hashlib
import importlib.util
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/kaggle_support_dual_numpy_20261009")
    args = parser.parse_args()
    folder = ROOT / args.output
    zip_path = folder.with_suffix(".zip")
    if zip_path.exists():
        raise RuntimeError("Support ZIP exists; preserve it and choose a fresh output")
    folder.mkdir(parents=True, exist_ok=True)
    for abi in ("cp310", "cp311", "cp312"):
        source = ROOT / f"artifacts/kaggle_support_numpy126/wheels/{abi}"
        destination = folder / f"wheels/numpy126/{abi}"
        if destination.exists():
            raise RuntimeError("Copied NumPy-1 profile already exists; choose a fresh output")
        shutil.copytree(source, destination)
    spec = importlib.util.spec_from_file_location("dual_numpy_bootstrap", ROOT / "scripts/notebook_bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec); spec.loader.exec_module(bootstrap)
    profiles = []
    for profile, abis, numpy in (("numpy126", ("cp310", "cp311", "cp312"), "1.26.4"),
                                ("numpy2", ("cp310", "cp311", "cp312", "cp313"), "2.1.3")):
        for abi in abis:
            relative = f"wheels/{profile}/{abi}/wheel_manifest.json"
            path = folder / relative
            if not path.exists():
                raise RuntimeError(f"First build {relative} using prepare_kaggle_wheels.py")
            manifest = bootstrap.validate_wheelhouse(path, abi, {"numpy": numpy}, "Linux", "x86_64",
                                {"python_full_version": "3.13.15"} if abi == "cp313" else None)
            selected = bootstrap.select_wheelhouse(folder, abi, numpy_version=numpy)
            assert selected == path
            profiles.append({"manifest": relative, "python_abi": abi, "numpy_specifier": manifest["numpy_specifier"],
                             "metadata_validated_against_numpy": numpy})
    assert len(profiles) == 7
    weights = ROOT / "artifacts/pretrained/resnet18-f37072fd.pth"
    expected = json.loads((weights.parent / "weights_manifest.json").read_text())["sha256"]
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == expected
    (folder / "pretrained").mkdir(exist_ok=True)
    for name in (weights.name, "weights_manifest.json"):
        shutil.copyfile(weights.parent / name, folder / "pretrained" / name)
    for name in ("requirements-kaggle-numpy126.txt", "requirements-kaggle-numpy2.txt"):
        shutil.copyfile(ROOT / name, folder / name)
    (folder / "UPLOAD_README.md").write_text(
        "# Dual NumPy Kaggle support, 2026-10-09\n\n"
        "Upload kaggle_support_dual_numpy_20261009.zip as a NEW PRIVATE Kaggle dataset knee-support-dual-numpy-20261009.\n"
        "Preserve extracted wheels/numpy2/cp313, other wheels subfolders and pretrained.\n"
        "Prioritized actual runtime: Linux x86_64, Python 3.13.15 (cp313), imported NumPy 2.1.3.\n"
        "NumPy-2 profiles also cover cp310/cp311/cp312; NumPy-1 profiles cover cp310/cp311/cp312.\n"
        "The notebook selects ONE profile by actual Python ABI and IMPORTED NumPy version.\n"
        "No NumPy/torch/pandas/SciPy/sklearn replacements, installation or downgrades are required.\n"
        "Manifest verification_numpy is a metadata-check target, not an installed-runtime measurement.\n"
        "Reimport canonical notebooks/kaggle_smoke.ipynb; updating only inputs leaves old embedded setup.\n"
        "Attach official competition + this dataset only. Detach old codec/weight support inputs; keep their datasets intact.\n"
        "Python; GPU T4 x2 or P100; Internet off; fresh session/default overrides; Save Version / Save & Run All.\n"
        "Inspect environment_runtime.json (written before selection) and environment_preflight.json.\n"
        "For current kernel require cp313, imported NumPy 2.1.3 unchanged, numpy2_decode_only, core preservation true.\n"
        "Require actual-image usable slices, finite loss, 4 train/2 val, 1 epoch, synthetic=false and reload agreement.\n"
        "Do not claim competition decoding success until this genuine Kaggle run passes.\n"
        "No code dataset, MRI, official CSVs, credentials or trained checkpoints are included.\n", encoding="utf-8")
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    # Exact inventory protects the upload from accidental unrelated files.
    allowed = {"UPLOAD_README.md", "requirements-kaggle-numpy126.txt", "requirements-kaggle-numpy2.txt",
               "pretrained/resnet18-f37072fd.pth", "pretrained/weights_manifest.json"}
    for profile in profiles:
        path = folder / profile["manifest"]
        allowed.add(profile["manifest"])
        allowed.update((path.parent / n).relative_to(folder).as_posix() for n in json.loads(path.read_text())["wheels"])
    if {p.relative_to(folder).as_posix() for p in files} != allowed:
        raise RuntimeError("Unexpected files in Kaggle support folder")
    manifest = {"purpose": "codec software and public pretrained weights only",
                "prioritized_user_reported_runtime": {"python": "3.13.15", "python_abi": "cp313", "imported_numpy": "2.1.3",
                    "metadata_numpy": "2.1.3", "numpy_import_path": "/usr/local/lib/python3.13/dist-packages/numpy/__init__.py"},
                "profiles": profiles,
                "files": {p.relative_to(folder).as_posix(): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in files}}
    (folder / "support_manifest.json").write_text(json.dumps(manifest, indent=2))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(folder).as_posix())
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        for name, entry in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == entry["sha256"]
    print(json.dumps({"zip": str(zip_path), "files": len(files) + 1, "profiles": len(profiles),
                      "bytes": zip_path.stat().st_size, "sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()

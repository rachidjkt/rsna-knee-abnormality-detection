"""Build pinned software-only Linux codec profiles without replacing Kaggle's core stack."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "numpy126": {"numpy_specifier": ">=1.24,<2.0", "verification_numpy": "1.26.4",
                 "packages": ["pydicom==3.0.2", "pylibjpeg==2.1.0", "pylibjpeg-libjpeg==2.1.0", "pylibjpeg-openjpeg==2.2.1"]},
    "numpy2": {"numpy_specifier": ">=2,<3", "verification_numpy": "2.1.3",
               "packages": ["pydicom==3.0.2", "pylibjpeg==2.1.0", "pylibjpeg-libjpeg==2.4.0", "pylibjpeg-openjpeg==2.6.0"]},
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-version", choices=["310", "311", "312", "313"], default="312")
    parser.add_argument("--profile", choices=list(PROFILES), default="numpy126")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.profile == "numpy126" and args.python_version == "313":
        parser.error("NumPy 1.26 does not support Python 3.13; keep the actual kernel stack and choose its matching profile")
    profile = PROFILES[args.profile]
    output = Path(args.output or f"artifacts/kaggle_support_{args.profile}/wheels/cp{args.python_version}")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Output must be empty: preserve existing wheelhouses and choose a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pip", "download", "--dest", str(output),
               "--platform", "manylinux2014_x86_64", "--platform", "manylinux_2_28_x86_64",
               "--python-version", args.python_version, "--implementation", "cp", "--abi", f"cp{args.python_version}",
               "--only-binary=:all:", "--no-deps", "--index-url", "https://pypi.org/simple", *profile["packages"]]
    subprocess.run(command, check=True)
    wheels, metadata = {}, {}
    for path in sorted(output.glob("*.whl")):
        wheels[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with zipfile.ZipFile(path) as archive:
            entry = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            meta = Parser().parsestr(archive.read(entry).decode())
        metadata[path.name] = {"name": meta["Name"], "version": meta["Version"],
                              "requires_python": meta["Requires-Python"], "requires_dist": meta.get_all("Requires-Dist", [])}
    manifest = {"python_abi": f"cp{args.python_version}", "platform": "Linux x86_64",
                "profile": f"{args.profile}_decode_only", "numpy_specifier": profile["numpy_specifier"],
                "verification_numpy": profile["verification_numpy"], "packages": profile["packages"],
                "wheels": wheels, "package_metadata": metadata,
                "jpeg_ls_decoder": "pylibjpeg-libjpeg", "rle_decoder": "pydicom (native)"}
    if len(wheels) != len(profile["packages"]):
        raise RuntimeError("Incomplete or ambiguous wheel download")
    manifest_path = output / "wheel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    spec = importlib.util.spec_from_file_location("codec_bootstrap", ROOT / "scripts/notebook_bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec); spec.loader.exec_module(bootstrap)
    bootstrap.validate_wheelhouse(manifest_path, manifest["python_abi"], {"numpy": profile["verification_numpy"]}, "Linux", "x86_64")
    print(f"Prepared and dependency/hash-validated {len(wheels)} wheels for {manifest['python_abi']}, NumPy {profile['verification_numpy']}")


if __name__ == "__main__":
    main()

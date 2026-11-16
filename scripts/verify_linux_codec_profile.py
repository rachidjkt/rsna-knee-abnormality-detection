"""Exercise verified Linux wheel installation and decoding of existing synthetic fixtures.

Run inside a separate Linux venv containing NumPy, packaging and pip, but no
codec packages. Does not install or change NumPy; fixtures are never real MRI.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path


def blocked(*args, **kwargs):
    raise RuntimeError("Networking forbidden in Linux codec fixture verification")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--fixtures", default="outputs/numpy126_codec_fixtures")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-numpy", required=True)
    parser.add_argument("--expected-python")
    args = parser.parse_args()
    assert platform.system() == "Linux" and platform.machine() == "x86_64"
    assert not importlib.util.find_spec("jpeg_ls"), "pyjpegls must be absent"
    assert not importlib.util.find_spec("rle"), "external RLE must be absent"
    import numpy as np
    assert np.__version__ == args.expected_numpy
    if args.expected_python:
        assert platform.python_version() == args.expected_python
    before = (id(np), np.__version__, np.__file__)
    folder = Path(args.output).resolve(); folder.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("linux_codec_bootstrap", root / "scripts/notebook_bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec); spec.loader.exec_module(bootstrap)
    installed, runtime = bootstrap.runtime_dependencies({"numpy": np})
    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    manifest_path = Path(args.wheelhouse).resolve() / "wheel_manifest.json"
    manifest = bootstrap.validate_wheelhouse(manifest_path, abi, installed,
                    marker_environment={"python_full_version": platform.python_version()})
    socket.create_connection = blocked; socket.socket.connect = blocked
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    destination = folder / "offline_deps"
    subprocess.run(bootstrap.install_command(manifest_path, manifest, destination), check=True)
    sys.path.insert(0, str(destination)); importlib.invalidate_caches()
    import pydicom
    from pydicom.pixels import pixel_array
    from pydicom.uid import JPEGLSLossless, JPEGLSNearLossless, JPEG2000Lossless, RLELossless
    decoders = bootstrap.decoder_preflight()
    fixture_folder = Path(args.fixtures).resolve()
    truth = np.load(fixture_folder / "truth.npy")
    results = []
    for name, uid, plugin, tolerance in (
        ("jpeg_ls_lossless", JPEGLSLossless, "pylibjpeg", 0),
        ("jpeg_ls_near_lossless", JPEGLSNearLossless, "pylibjpeg", 1),
        ("jpeg2000_lossless", JPEG2000Lossless, "pylibjpeg", 0),
        ("rle_lossless", RLELossless, "pydicom", 0),
    ):
        pixels = pixel_array(fixture_folder / (name + ".dcm"), decoding_plugin=plugin)
        error = int(np.abs(pixels.astype(np.int32) - truth.astype(np.int32)).max())
        assert error <= tolerance
        results.append({"fixture": name, "transfer_syntax": str(uid), "plugin": plugin,
                        "max_absolute_pixel_error": error, "passed": True})
    assert before == (id(importlib.import_module("numpy")), np.__version__, np.__file__)
    report = {"synthetic": True, "genuine_mri": False, "platform": platform.platform(),
              "python": platform.python_version(), "python_abi": abi, "numpy": np.__version__,
              "numpy_runtime": runtime["numpy"], "tested_numerical_modules": ["numpy"],
              "numpy_preserved": True, "pyjpegls_installed": False, "external_rle_installed": False,
              "profile": manifest["profile"], "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "install_mode": "exact hash-verified files, --no-index --no-deps --target",
              "versions": {p: importlib.metadata.version(p) for p in ("pydicom", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg")},
              "decoders": decoders, "fixtures": results, "networking": "socket guard; pip offline only",
              "limitations": "Isolated WSL codec test; no Kaggle GPU, torch/pandas/SciPy/sklearn execution or real MRI."}
    (folder / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

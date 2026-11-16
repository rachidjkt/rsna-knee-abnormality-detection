"""Prepare separate Linux decoder-test venvs from cached software wheels only.

Requires the project-local uv executable and isolated Python 3.13.15, previously
downloaded for testing. Does not alter any system or Kaggle numerical stack.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["numpy126", "numpy2"], required=True)
    args = parser.parse_args()
    assert platform.system() == "Linux"
    root = Path(__file__).resolve().parents[1]
    uv = root / ".tmp/linux_uv/uv"
    cache = root / ".tmp/uv_cache"
    env = root / f".tmp/linux_codec_{args.profile}"
    if env.exists():
        raise RuntimeError("Test venv already exists; preserve it")
    python = root / ".tmp/linux_python/cpython-3.13.15-linux-x86_64-gnu/bin/python3.13" if args.profile == "numpy2" else Path(sys.executable)
    subprocess.run([str(uv), "venv", "--python", str(python), "--cache-dir", str(cache), str(env)], check=True)
    tools = root / "artifacts/dual_numpy_linux_test_tools"
    numpy_folder = tools if args.profile == "numpy2" else root / "artifacts/dual_numpy_linux126_test_tools"
    wheels = [next(numpy_folder.glob("numpy-*.whl")), next(tools.glob("packaging-*.whl")), next(tools.glob("pip-*.whl"))]
    subprocess.run([str(uv), "pip", "install", "--python", str(env / "bin/python"), "--cache-dir", str(cache),
                    "--no-index", "--no-deps", *map(str, wheels)], check=True)
    (env / "test_software_manifest.json").write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in wheels}, indent=2))


if __name__ == "__main__":
    main()

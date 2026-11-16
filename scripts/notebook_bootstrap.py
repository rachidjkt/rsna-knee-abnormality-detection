"""Embedded offline Kaggle setup. Helpers are also exercised locally in tests."""
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from email.parser import Parser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.markers import default_environment
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, parse_wheel_filename

CODEC_PACKAGES = {"pydicom", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "pylibjpeg-rle", "pyjpegls"}
CORE_DISTRIBUTIONS = {"torch": "torch", "torchvision": "torchvision", "numpy": "numpy", "pandas": "pandas",
                      "scipy": "scipy", "sklearn": "scikit-learn"}


def runtime_dependencies(modules, distributions=None):
    """Use loaded core versions for validation; retain all metadata copies as evidence.

    importlib.metadata can see duplicate/stale dist-info records on sys.path.
    Neither an attached wheel's Requires-Dist nor such a record proves which
    binary module Python actually imported. Never infer runtime NumPy from them.
    """
    installed, records = {}, {canonicalize_name(name): [] for name in CORE_DISTRIBUTIONS.values()}
    for distribution in distributions if distributions is not None else importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        name = canonicalize_name(name)
        installed[name] = distribution.version
        if name in records:
            records[name].append({"version": distribution.version, "location": str(distribution.locate_file("")),
                                  "metadata_path": str(getattr(distribution, "_path", "unknown"))})
    diagnostics = {}
    for module_name, module in modules.items():
        name = canonicalize_name(CORE_DISTRIBUTIONS[module_name])
        version = str(module.__version__)
        installed[name] = version
        diagnostics[module_name] = {"imported_version": version, "imported_path": str(module.__file__),
                                    "metadata_distributions": records[name],
                                    "metadata_matches_import": bool(records[name]) and all(r["version"] == version for r in records[name]),
                                    "dependency_version_source": "imported module"}
    return installed, diagnostics


def select_wheelhouse(root, abi, explicit=None, numpy_version=None):
    paths = [Path(explicit) / "wheel_manifest.json"] if explicit else sorted(Path(root).rglob("wheel_manifest.json"))
    matches = []
    for path in paths:
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        # Historical wheelhouses contain NumPy-2 codecs. New profiles declare their range.
        numpy_range = manifest.get("numpy_specifier", ">=2,<3")
        if manifest["python_abi"] == abi and (numpy_version is None or numpy_version in SpecifierSet(numpy_range)):
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"Attach exactly one compatible wheelhouse for kernel ABI {abi}, NumPy {numpy_version}; found {len(matches)}. See docs/KAGGLE.md")
    return matches[0]


def validate_wheelhouse(manifest_path, abi, installed, system=None, machine=None, marker_environment=None):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["python_abi"] != abi:
        raise RuntimeError("Wheelhouse ABI mismatch")
    if (system or platform.system()) != "Linux" or (machine or platform.machine()).lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("This codec wheelhouse requires Linux x86_64")
    available = {canonicalize_name(k): v for k, v in installed.items()}
    if manifest.get("numpy_specifier") and available.get("numpy", "0") not in SpecifierSet(manifest["numpy_specifier"]):
        raise RuntimeError("Wheelhouse NumPy profile incompatible; do not replace Kaggle's numerical stack")
    environment = default_environment()
    environment.update(python_version=f"3.{abi[3:]}", python_full_version=f"3.{abi[3:]}.0",
                       sys_platform="linux", platform_system="Linux", platform_machine="x86_64", extra="")
    environment.update(marker_environment or {})
    metadata = []
    packages = set()
    for filename, digest in manifest["wheels"].items():
        if Path(filename).name != filename or not filename.endswith(".whl"):
            raise RuntimeError("Unsafe wheel manifest filename")
        path = manifest_path.parent / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Wheel checksum mismatch: {filename}")
        _, _, _, tags = parse_wheel_filename(filename)
        if not any(tag.interpreter == "py3" and tag.platform == "any" or
                   tag.interpreter == abi and tag.abi == abi and "manylinux" in tag.platform and tag.platform.endswith("x86_64") for tag in tags):
            raise RuntimeError(f"Wheel tags incompatible with {abi}: {filename}")
        with zipfile.ZipFile(path) as archive:
            entries = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            if len(entries) != 1:
                raise RuntimeError(f"Wheel metadata invalid: {filename}")
            meta = Parser().parsestr(archive.read(entries[0]).decode())
        name = canonicalize_name(meta["Name"])
        if name not in CODEC_PACKAGES:
            raise RuntimeError(f"Forbidden offline package {name}; numerical-stack replacements are not allowed")
        if name in packages:
            raise RuntimeError(f"Duplicate offline package {name}")
        packages.add(name)
        if meta.get("Requires-Python") and environment["python_full_version"] not in SpecifierSet(meta["Requires-Python"]):
            raise RuntimeError(f"Incompatible Requires-Python for {name}: {meta['Requires-Python']}")
        available[name] = meta["Version"]
        metadata.append(meta)
    for meta in metadata:
        for spec in meta.get_all("Requires-Dist", []):
            requirement = Requirement(spec)
            if requirement.marker and not requirement.marker.evaluate(environment):
                continue
            version = available.get(canonicalize_name(requirement.name))
            if version is None or version not in requirement.specifier:
                raise RuntimeError(f"Incompatible offline dependency {meta['Name']}: requires {spec}, installed {version}. Do not replace Kaggle's numerical stack.")
    return manifest


def install_command(manifest_path, manifest, destination, synthetic=False):
    # Install only checksum-verified files, never unchecked neighbours or resolved dependencies.
    wheels = [str(Path(manifest_path).parent / name) for name in manifest["wheels"]
              if not synthetic or canonicalize_name(parse_wheel_filename(name)[0]) == "pydicom"]
    return [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--upgrade",
            "--target", str(destination), *wheels]


def decoder_preflight():
    from pydicom.pixels import get_decoder
    from pydicom.uid import (JPEGBaseline8Bit, JPEGExtended12Bit, JPEGLossless, JPEGLosslessSV1,
                            JPEGLSLossless, JPEGLSNearLossless, JPEG2000Lossless, JPEG2000, RLELossless)
    report = {}
    for uid in (JPEGBaseline8Bit, JPEGExtended12Bit, JPEGLossless, JPEGLosslessSV1,
                JPEGLSLossless, JPEGLSNearLossless, JPEG2000Lossless, JPEG2000, RLELossless):
        decoder = get_decoder(uid)
        required = "pydicom" if uid == RLELossless else "pylibjpeg"
        if required not in decoder.available_plugins:
            raise RuntimeError(f"Required {required} decoder unavailable for {uid.name}: {decoder.missing_dependencies}")
        report[str(uid)] = {"name": uid.name, "required_plugin": required, "available_plugins": list(decoder.available_plugins)}
    return report


def bootstrap():
    work = Path(os.environ.get("KNEE_WORK_DIR", "/kaggle/working"))
    inputs = Path(os.environ.get("KNEE_INPUT_DIR", "/kaggle/input"))
    work.mkdir(parents=True, exist_ok=True)
    synthetic = os.environ.get("KNEE_SYNTHETIC", "0") == "1"
    modules = {name: importlib.import_module(name) for name in CORE_DISTRIBUTIONS}
    installed, runtime = runtime_dependencies(modules)
    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    early_report = {"python": sys.version.split()[0], "python_abi": abi, "platform": platform.platform(),
                    "synthetic": synthetic, "selection_uses_imported_numpy": True, "core": runtime}
    # Written before selection/install: retain evidence even when setup fails.
    (work / "environment_runtime.json").write_text(json.dumps(early_report, indent=2))
    print({"python": early_report["python"], "python_abi": abi, "numpy_runtime": runtime["numpy"]})
    torch = modules["torch"]
    if not hasattr(torch.amp, "GradScaler"):
        raise RuntimeError("This pipeline requires torch.amp.GradScaler; use a current Kaggle GPU image")
    torch.amp.GradScaler("cuda", enabled=False)
    core_before = {name: (id(module), str(module.__version__), str(module.__file__)) for name, module in modules.items()}
    codec_modules = ["pydicom"] if synthetic else ["pydicom", "pylibjpeg", "libjpeg", "openjpeg"]
    missing = importlib.util.find_spec("pydicom") is None
    manifest_path = None
    # Real runs always install the attached pinned profile, regardless of image defaults.
    if not synthetic or missing:
        manifest_path = select_wheelhouse(inputs, abi, os.environ.get("KNEE_WHEELHOUSE"), modules["numpy"].__version__)
        manifest = validate_wheelhouse(manifest_path, abi, installed,
                                      marker_environment={"python_full_version": sys.version.split()[0]})
        destination = work / "offline_deps" / hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]
        subprocess.run(install_command(manifest_path, manifest, destination, synthetic), check=True)
        sys.path.insert(0, str(destination))
        for name in list(sys.modules):
            if name.split(".")[0] in codec_modules:
                del sys.modules[name]
        importlib.invalidate_caches()
    for name in codec_modules:
        importlib.import_module(name)
    import pydicom
    from packaging.version import Version
    if not Version("3.0") <= Version(pydicom.__version__) < Version("4"):
        raise RuntimeError("pydicom >=3,<4 required; attach a compatible offline wheelhouse")
    core_after = {name: (id(importlib.import_module(name)), str(module.__version__), str(module.__file__)) for name, module in modules.items()}
    if core_before != core_after:
        raise RuntimeError("Offline codec installation changed the numerical stack")
    decoders = {} if synthetic else decoder_preflight()
    report = {"python": sys.version.split()[0], "python_abi": abi, "platform": platform.platform(),
              "versions": {name: str(module.__version__) for name, module in modules.items()},
              "pydicom": pydicom.__version__, "cuda": torch.cuda.is_available(), "synthetic": synthetic,
              "offline_install_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path else None,
              "offline_profile": json.loads(manifest_path.read_text()).get("profile", "legacy_numpy2") if manifest_path else None,
              "numerical_stack_preserved": core_before == core_after, "decoders": decoders,
              "selection_uses_imported_numpy": True, "runtime_dependencies": runtime,
              "codec_versions": {name: str(importlib.import_module(name).__version__) for name in codec_modules},
              "gpu": None}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        report["gpu"] = {"name": torch.cuda.get_device_name(), "free_mb": free / 1024**2, "total_mb": total / 1024**2}
    (work / "environment_preflight.json").write_text(json.dumps(report, indent=2))
    print(report)
    if not synthetic and not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator for this real-data workflow")
    return work, inputs, synthetic


if __name__ == "__main__":
    WORK, INPUT, SYNTHETIC = bootstrap()
    import torch, numpy as np, pandas as pd, pydicom

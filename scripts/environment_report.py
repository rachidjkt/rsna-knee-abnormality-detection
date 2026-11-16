"""Aggregate environment diagnostics without credential values or patient data."""
import ctypes
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path

import torch


def main():
    versions = {}
    for package in ["torch", "torchvision", "numpy", "pandas", "scikit-learn", "pydicom", "pytest",
                    "matplotlib", "nbformat", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "pylibjpeg-rle", "pyjpegls"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    memory_gb = None
    if platform.system() == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                        ("available_extended", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            memory_gb = status.total_physical / 1024**3
    disk = shutil.disk_usage(".")
    result = {"python": platform.python_version(), "platform": platform.platform(), "dependencies": versions,
              "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None,
              "physical_memory_gb": memory_gb, "free_disk_gb": disk.free / 1024**3,
              "kaggle_credentials_file_present": (Path.home() / ".kaggle/kaggle.json").exists()}
    Path("reports/environment.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

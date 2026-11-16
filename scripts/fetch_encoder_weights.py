"""Fetch official, publicly available torchvision ResNet18 weights (software only)."""
import hashlib
import json
from pathlib import Path

import torch


def main():
    root = Path("artifacts/pretrained")
    root.mkdir(parents=True, exist_ok=True)
    output = root / "resnet18-f37072fd.pth"
    url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    if not output.exists():
        torch.hub.download_url_to_file(url, str(output), hash_prefix="f37072fd", progress=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    if not digest.startswith("f37072fd"):
        raise ValueError("Weight checksum does not match the official torchvision URL hash")
    manifest = {"architecture": "resnet18", "weights": "IMAGENET1K_V1", "url": url,
                "sha256": digest, "bytes": output.stat().st_size,
                "code_license": "torchvision BSD-3-Clause; see upstream LICENSE",
                "eligibility": "Not established by this script; verify competition rules and weight terms before use.",
                "usage": "Mount this directory as a Kaggle dataset for training. Inference uses the complete saved checkpoint."}
    (root / "weights_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

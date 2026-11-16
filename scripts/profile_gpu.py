"""Measure baseline tensor compute/memory with synthetic inputs; no MRI performance claim."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from knee_ai.models.multislice import MultiSliceClassifier
from knee_ai.training.engine import masked_bce
from knee_ai.utils.reproducibility import seed_everything, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/gpu_profile.json")
    parser.add_argument("--weights", default="artifacts/pretrained/resnet18-f37072fd.pth")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this measurement")
    seed_everything(42)
    device = torch.device("cuda")
    model = MultiSliceClassifier("resnet18", encoder_weights=args.weights, freeze_encoder=True).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    images = torch.rand(2, 24, 1, 224, 224, device=device)
    mask = torch.ones(2, 24, dtype=torch.bool, device=device)
    labels = torch.randint(0, 2, (2, 12), device=device).float()
    scaler = torch.amp.GradScaler("cuda")

    def step():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            loss = masked_bce(model(images, mask), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    step()  # warm up allocations and kernels before measuring
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    training_seconds = (time.perf_counter() - start) / 3
    training_peak = torch.cuda.max_memory_allocated() / 1024**2
    training_reserved = torch.cuda.max_memory_reserved() / 1024**2
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda"):
        for _ in range(3):
            probabilities = model(images, mask, encode_chunk_size=32).float().sigmoid()
    torch.cuda.synchronize()
    inference_seconds = (time.perf_counter() - start) / 3
    assert torch.isfinite(probabilities).all()
    report = {"synthetic_inputs": True, "genuine_mri": False, "device": torch.cuda.get_device_name(),
              "torch": str(torch.__version__), "shape": list(images.shape), "amp": True, "encoder_frozen": True,
              "weights_sha256": sha256_file(args.weights), "input_tensor_mb": images.numel() * images.element_size() / 1024**2,
              "training_seconds_per_batch": training_seconds, "training_peak_allocated_mb": training_peak,
              "training_peak_reserved_mb": training_reserved, "inference_seconds_per_batch": inference_seconds,
              "inference_peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
              "limitation": "Tensor compute only. Excludes DICOM decoding, Kaggle I/O, cache, Linux codecs, and hidden-test throughput. Frozen encoder only."}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

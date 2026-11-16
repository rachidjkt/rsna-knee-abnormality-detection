"""Actual synthetic DICOM -> training -> checkpoint -> test inference, no network."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee_ai.data.metadata import audit_metadata
from knee_ai.data.synthetic import create_synthetic
from knee_ai.inference.predict import infer
from knee_ai.training.engine import TrainConfig, train
from knee_ai.utils.reproducibility import configure_logging, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encoder", default="tiny")
    args = parser.parse_args()
    configure_logging()
    root = Path(args.output)
    create_synthetic(root / "data", train_studies=16, labelled_count=12, test_studies=3, slices=6)
    audit_metadata(root / "data", root / "metadata_audit.json")
    config = TrainConfig(data_dir=str(root / "data"), image_root=str(root / "data/train_images"),
                         output_dir=str(root / "train"), split_path=str(root / "split.json"), encoder=args.encoder,
                         device=args.device, slices_per_series=2, image_size=48, epochs=2, batch_size=2,
                         cache_dir=str(root / "cache"), synthetic=True)
    trained = train(config)
    predicted = infer(root / "train/best.pt", root / "data", root / "data/test_images",
                      root / "submission.csv", device=args.device)
    write_json(root / "smoke_results.json", {"training": trained, "inference": predicted})
    print(json.dumps({"synthetic": True, "train_epochs": trained["completed_epochs"], "inferred_studies": predicted["studies"],
                      "train_seconds": trained["seconds"], "infer_seconds": predicted["seconds"]}, indent=2))


if __name__ == "__main__":
    main()

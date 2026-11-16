from __future__ import annotations

import argparse
import json
from pathlib import Path

from knee_ai.data.metadata import audit_metadata, load_train
from knee_ai.data.splits import save_split
from knee_ai.data.synthetic import create_synthetic
from knee_ai.evaluation.experiment import compare_conditions
from knee_ai.inference.predict import infer
from knee_ai.training.engine import TrainConfig, train
from knee_ai.training.smoke import smoke_train
from knee_ai.utils.reproducibility import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Study-level knee MRI research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--data-dir", default="data")
    audit.add_argument("--output", default="reports/metadata_audit.json")
    split = commands.add_parser("split")
    split.add_argument("--data-dir", default="data")
    split.add_argument("--output", default="experiments/study_split.json")
    split.add_argument("--seed", type=int, default=42)
    synthetic = commands.add_parser("synthetic")
    synthetic.add_argument("--output", default="outputs/synthetic_data")
    synthetic.add_argument("--seed", type=int, default=123)
    for name in ("train", "compare", "smoke-real"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        if name == "compare":
            command.add_argument("--output", default="outputs/comparison")
            command.add_argument("--seeds", type=int, nargs="+", default=[42])
    inference = commands.add_parser("infer")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--data-dir", required=True)
    inference.add_argument("--image-root", required=True)
    inference.add_argument("--output", default="outputs/submission.csv")
    inference.add_argument("--device", default="auto")
    inference.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    configure_logging()
    if args.command == "audit":
        result = audit_metadata(args.data_dir, args.output)
        print(json.dumps({k: result[k] for k in ["train_studies", "train_series", "fully_labelled_studies", "unlabelled_studies"]}, indent=2))
    elif args.command == "split":
        split = save_split(load_train(Path(args.data_dir) / "train.csv"), args.output, args.seed)
        print(f"Saved {len(split['train_ids'])} train / {len(split['val_ids'])} validation studies")
    elif args.command == "synthetic":
        print(json.dumps(create_synthetic(args.output, seed=args.seed), indent=2))
    elif args.command in ("train", "compare", "smoke-real"):
        config = TrainConfig.from_dict(json.loads(Path(args.config).read_text()))
        if args.command == "compare":
            result = compare_conditions(config, args.output, args.seeds)
        else:
            result = smoke_train(config) if args.command == "smoke-real" else train(config)
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(infer(args.checkpoint, args.data_dir, args.image_root, args.output, args.device, args.batch_size), indent=2))


if __name__ == "__main__":
    main()

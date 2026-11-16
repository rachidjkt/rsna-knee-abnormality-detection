"""Execute every code cell in generated notebooks against local synthetic mounts.

Runs real Python cells in an isolated namespace; no Jupyter frontend or Kaggle
service is involved. Networking is blocked by a socket guard. The .ipynb format
and cell syntax are validated too. This is not a real Kaggle run.
"""
import argparse
import json
import os
import socket
import time
from pathlib import Path

import nbformat


def blocked_network(*args, **kwargs):
    raise RuntimeError("Network access forbidden in notebook verification")


def execute(path: Path) -> float:
    start = time.perf_counter()
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"{path.name}:cell{index}", "exec"), namespace)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="outputs/synthetic_data")
    parser.add_argument("--output", default="outputs/notebook_verification")
    parser.add_argument("--skip-experiment", action="store_true", help="Skip previously verified comparison; execute changed train/infer/smoke flows")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(KNEE_SYNTHETIC="1", KNEE_DATA_DIR=str(Path(args.data_dir).resolve()),
                      KNEE_INPUT_DIR=str(output / "inputs"), KNEE_WORK_DIR=str(output / "train"))
    socket.create_connection = blocked_network
    socket.socket.connect = blocked_network
    results = {}
    results["training_cells_seconds"] = execute(Path("notebooks/kaggle_train.ipynb"))
    os.environ.update(KNEE_CHECKPOINT=str(output / "train/baseline/best.pt"), KNEE_WORK_DIR=str(output / "inference"))
    results["inference_cells_seconds"] = execute(Path("notebooks/kaggle_inference.ipynb"))
    if not args.skip_experiment:
        os.environ["KNEE_WORK_DIR"] = str(output / "comparison")
        results["experiment_cells_seconds"] = execute(Path("notebooks/kaggle_slice_experiment.ipynb"))
    os.environ["KNEE_WORK_DIR"] = str(output / "smoke")
    results["smoke_cells_seconds"] = execute(Path("notebooks/kaggle_smoke.ipynb"))
    results.update(synthetic=True, networking="blocked", kaggle_remote_execution=False,
                   source_sha256=nbformat.read("notebooks/kaggle_train.ipynb", as_version=4).metadata.knee_ai_source_sha256)
    (output / "verification.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

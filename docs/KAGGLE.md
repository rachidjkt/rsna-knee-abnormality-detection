# Offline Kaggle notebooks

This portfolio edition contains four baseline notebooks in `notebooks/`: smoke, train, inference, and slice-selection comparison. They embed the current `src/knee_ai` source for offline execution. Rebuild with `python scripts/build_notebooks.py` after package source changes.

For a genuine run, an authorized user must attach the competition inputs, separately acquired torchvision ResNet18 weights with provenance, and a Linux codec support wheelhouse compatible with the kernel's Python ABI and NumPy. No wheelhouse, dataset, or weights are published here. `scripts/prepare_kaggle_wheels.py`, `scripts/prepare_linux_codec_tests.py`, and the NumPy support packaging scripts are optional environment-preparation utilities; they may download dependencies when deliberately run. They were retained from the existing workflow and were not run during portfolio preparation.

Run the synthetic local smoke first. In Kaggle, run a smoke notebook in a fresh offline session before full training. Use the training notebook output as the inference input. Set paths through the notebook's documented KNEE_* variables; inspect `scripts/notebook_bootstrap.py` for ABI/profile validation. Do not automatically submit an output. The synthetic training checkpoint is diagnostic, not a medical model.

Historical scored checkpoints may have different preprocessing versions or source. This edition rejects incompatible checkpoints. Use their exact original source rather than disabling compatibility checks. Full Kaggle GPU execution, hidden-test runtime, fresh Linux binary installation, and competition eligibility of external weights remain unverified in this preparation.

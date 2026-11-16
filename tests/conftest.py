from pathlib import Path

import pytest
import torch

from knee_ai.data.synthetic import create_synthetic


@pytest.fixture(scope="session")
def fixtures(tmp_path_factory) -> Path:
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("synthetic")
    create_synthetic(root, train_studies=12, labelled_count=8, test_studies=2, slices=5)
    return root

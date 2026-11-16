import json

import pandas as pd
import torch

from knee_ai.constants import STUDY_ID, TARGETS
from knee_ai.inference.predict import infer, load_model
from knee_ai.training.engine import TrainConfig, train


def test_train_checkpoint_infer_end_to_end(fixtures, tmp_path):
    config = TrainConfig(data_dir=str(fixtures), image_root=str(fixtures / "train_images"),
                         output_dir=str(tmp_path / "run"), split_path=str(tmp_path / "split.json"),
                         encoder="tiny", image_size=32, slices_per_series=2, batch_size=2,
                         epochs=2, device="cpu", synthetic=True)
    result = train(config)
    assert result["completed_epochs"] == 2
    assert result["train_studies"] + result["val_studies"] == 8
    assert result["synthetic"] is True
    a, checkpoint = load_model(tmp_path / "run/best.pt", torch.device("cpu"))
    b, _ = load_model(tmp_path / "run/best.pt", torch.device("cpu"))
    image = torch.rand(1, 3, 1, 32, 32)
    mask = torch.ones(1, 3, dtype=torch.bool)
    torch.testing.assert_close(a(image, mask), b(image, mask), atol=0, rtol=0)
    assert checkpoint["targets"] == TARGETS
    predictions = []
    for name in ("first", "second"):
        summary = infer(tmp_path / "run/best.pt", fixtures, fixtures / "test_images",
                        tmp_path / f"{name}.csv", "cpu")
        assert summary["studies"] == 2
        predictions.append(pd.read_csv(tmp_path / f"{name}.csv"))
    pd.testing.assert_frame_equal(*predictions)
    assert predictions[0][TARGETS].notna().all().all()

import numpy as np
import pytest
import torch

from knee_ai.constants import STUDY_ID, TARGETS
from knee_ai.data.metadata import load_template
from knee_ai.evaluation.metrics import multilabel_metrics, paired_bootstrap
from knee_ai.inference.predict import write_submission
from knee_ai.models.multislice import MultiSliceClassifier
from knee_ai.training.engine import masked_bce


@pytest.mark.parametrize("encoder", ["tiny", "resnet18"])
def test_model_shape_gradient_and_mask_invariance(encoder):
    model = MultiSliceClassifier(encoder, dropout=0).eval()
    images = torch.rand(2, 4, 1, 32, 32)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool)
    logits = model(images, mask)
    assert logits.shape == (2, 12)
    replaced = images.clone()
    replaced[~mask] = 10000
    torch.testing.assert_close(model(replaced, mask, 2), logits, atol=1e-5, rtol=1e-5)
    labels = torch.randint(0, 2, (2, 12)).float()
    masked_bce(logits, labels).backward()
    assert model.head[-1].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="at least one"):
        model(images, torch.zeros_like(mask))


def test_feature_aggregation_equals_mean_slice_features():
    model = MultiSliceClassifier("tiny", dropout=0).eval()
    images = torch.rand(1, 3, 1, 32, 32)
    all_valid = model(images, torch.ones(1, 3, dtype=torch.bool))
    separate = torch.cat([model(images[:, i:i + 1], torch.ones(1, 1, dtype=torch.bool)) for i in range(3)])
    torch.testing.assert_close(all_valid, separate.mean(0, keepdim=True))


def test_loss_masks_unknown_labels():
    logits = torch.zeros(1, 12, requires_grad=True)
    y = torch.zeros(1, 12)
    y[0, 0] = float("nan")
    loss = masked_bce(logits, y)
    assert float(loss.detach()) == pytest.approx(np.log(2))
    loss.backward()
    assert logits.grad[0, 0] == 0
    with pytest.raises(ValueError):
        masked_bce(logits, torch.full_like(y, float("nan")))


def test_perfect_and_undefined_auc():
    y = np.tile([[0], [1], [0], [1]], (1, 12))
    p = y * .8 + .1
    result = multilabel_metrics(y, p)
    assert result["macro_auc_all_12"] == 1
    assert result["defined_classes"] == 12
    y[:, 0] = 0
    result = multilabel_metrics(y, p)
    assert result["macro_auc_all_12"] is None
    assert result["macro_auc_defined"] == 1
    assert result["defined_classes"] == 11
    assert result["per_class"]["ACL"]["roc_auc"] is None


def test_missing_metric_labels_and_bad_probabilities():
    y = np.full((3, 12), np.nan)
    result = multilabel_metrics(y, np.full((3, 12), .5))
    assert result["defined_classes"] == 0
    with pytest.raises(ValueError):
        multilabel_metrics(y, np.full((3, 12), 1.1))


def test_paired_bootstrap_resamples_studies():
    y = np.tile(np.arange(20)[:, None] % 2, (1, 12))
    a = y * .8 + .1
    result = paired_bootstrap(y, a, 1 - a, draws=30)
    assert result["delta_central_minus_uniform_ci95"] == [-1, -1]


def test_submission_order_and_validation(fixtures, tmp_path):
    template = load_template(fixtures / "sample_submission.csv")
    ids = template[STUDY_ID].tolist()[::-1]
    p = np.vstack([np.full(12, .2), np.full(12, .7)])
    result = write_submission(template, ids, p, tmp_path / "submission.csv")
    assert result.columns.tolist() == [STUDY_ID, *TARGETS]
    assert result.iloc[0][TARGETS].to_numpy(dtype=float).tolist() == pytest.approx([.7] * 12)
    for bad_ids, bad_p in [(ids[:1], p[:1]), (ids * 2, np.tile(p, (2, 1))), (ids, p * 10)]:
        with pytest.raises(ValueError):
            write_submission(template, bad_ids, bad_p, tmp_path / "bad.csv")

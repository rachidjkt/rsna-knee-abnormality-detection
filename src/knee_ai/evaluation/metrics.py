from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from knee_ai.constants import TARGETS


def multilabel_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = .5) -> dict:
    y, p = np.asarray(labels), np.asarray(probabilities)
    if y.ndim != 2 or y.shape != p.shape or y.shape[1] != len(TARGETS) or not len(y):
        raise ValueError("Expected matching, nonempty [N,12] arrays")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Predictions must be finite probabilities in [0,1]")
    known = np.isfinite(y)
    if np.isinf(y).any() or not np.isin(y[known], [0, 1]).all() or not 0 <= threshold <= 1:
        raise ValueError("Invalid labels or threshold")
    aucs, per_class = [], {}
    for i, target in enumerate(TARGETS):
        observed = known[:, i]
        truth, pred = y[observed, i], p[observed, i]
        auc = float(roc_auc_score(truth, pred)) if len(np.unique(truth)) == 2 else None
        if auc is not None:
            aucs.append(auc)
        per_class[target] = {"roc_auc": auc, "known": int(observed.sum()),
                             "positive": int(truth.sum()),
                             "negative": int((truth == 0).sum()), "missing": int((~observed).sum()),
                             "prevalence": float(truth.mean()) if len(truth) else None,
                             "confusion_matrix_tn_fp_fn_tp": confusion_matrix(truth, pred >= threshold, labels=[0, 1]).ravel().tolist() if len(truth) else None,
                             "prediction_min": float(p[:, i].min()), "prediction_max": float(p[:, i].max()),
                             "prediction_mean": float(p[:, i].mean()), "prediction_std": float(p[:, i].std())}
    return {"macro_auc_all_12": float(np.mean(aucs)) if len(aucs) == 12 else None,
            "macro_auc_defined": float(np.mean(aucs)) if aucs else None,
            "defined_classes": len(aucs), "total_classes": 12, "threshold": threshold,
            "per_class": per_class}


def paired_bootstrap(labels: np.ndarray, uniform: np.ndarray, central: np.ndarray,
                     seed: int = 123, draws: int = 1000) -> dict:
    """Resample whole held-out studies; require all twelve AUCs for a valid replicate."""
    if draws < 1:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        index = rng.integers(0, len(labels), len(labels))
        a = multilabel_metrics(labels[index], uniform[index])["macro_auc_all_12"]
        b = multilabel_metrics(labels[index], central[index])["macro_auc_all_12"]
        if a is not None and b is not None:
            differences.append(b - a)
    return {"unit": "study", "draws": draws, "valid_draws": len(differences),
            "delta_central_minus_uniform_ci95": np.percentile(differences, [2.5, 97.5]).tolist() if differences else None,
            "limitation": "Very small labelled holdouts can make most bootstrap draws undefined; not a clinical claim."}

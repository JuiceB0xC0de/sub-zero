from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class AxisFitResult:
    corporate_axis: torch.Tensor
    neutral_midpoint_projection: float
    classifier_coef: torch.Tensor
    classifier_accuracy: float


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def fit_corporate_axis(
    corporate_acts: torch.Tensor,
    neutral_acts: torch.Tensor,
    authentic_acts: torch.Tensor,
) -> AxisFitResult:
    """Fit corp-vs-auth axis via multinomial logistic regression when available."""
    c = corporate_acts.detach().float().cpu().numpy()
    n = neutral_acts.detach().float().cpu().numpy()
    a = authentic_acts.detach().float().cpu().numpy()

    x = np.concatenate([c, n, a], axis=0)
    y = np.concatenate(
        [
            np.full(c.shape[0], 1, dtype=np.int64),
            np.full(n.shape[0], 0, dtype=np.int64),
            np.full(a.shape[0], -1, dtype=np.int64),
        ],
        axis=0,
    )

    coef = None
    accuracy = 0.0

    try:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(multi_class="multinomial", max_iter=1000, random_state=42)
        clf.fit(x, y)
        pred = clf.predict(x)
        accuracy = float((pred == y).mean())
        classes = list(clf.classes_)
        c_idx = classes.index(1)
        a_idx = classes.index(-1)
        coef = clf.coef_[c_idx] - clf.coef_[a_idx]
    except Exception:
        # Fallback: mean-difference axis if sklearn is unavailable.
        coef = c.mean(axis=0) - a.mean(axis=0)
        score = x @ _normalize(coef)
        # rough ternary accuracy with midpoint heuristics
        y_hat = np.where(score > 0.2, 1, np.where(score < -0.2, -1, 0))
        accuracy = float((y_hat == y).mean())

    axis = _normalize(coef)

    neutral_mean = n.mean(axis=0)
    neutral_projection = float(np.dot(neutral_mean, axis))

    return AxisFitResult(
        corporate_axis=torch.from_numpy(axis).float(),
        neutral_midpoint_projection=neutral_projection,
        classifier_coef=torch.from_numpy(np.asarray(coef)).float(),
        classifier_accuracy=accuracy,
    )


def project_on_axis(x: torch.Tensor, axis: torch.Tensor, midpoint: float = 0.0) -> torch.Tensor:
    axis = axis.to(device=x.device, dtype=x.dtype)
    return x @ axis - midpoint

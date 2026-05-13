import numpy as np


def ace(
    preds:     np.ndarray,             # [B, T, H, C, Q]
    targets:   np.ndarray,             # [B, T, H, C]
    quantiles: np.ndarray,             # [Q]  e.g. [0.1, 0.25, 0.5, 0.75, 0.9]
    mask:      np.ndarray | None = None,
) -> float:
    """
    Average Coverage Error (ACE).
    For each quantile q, measures |empirical_coverage - q|, then averages.
    """
    errors = []
    for i, q in enumerate(quantiles):
        pred_q = preds[..., i]                          # [B, T, H, C]
        covered = (targets <= pred_q).astype(float)     # [B, T, H, C]

        if mask is not None:
            empirical = (covered * mask).sum() / max(mask.sum(), 1)
        else:
            empirical = covered.mean()

        errors.append(abs(empirical - q))

    return float(np.mean(errors))
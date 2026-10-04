"""4th-down yards-gained model (nfl4th fd_model, xgboost multi:softprob)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl4th_py import artifacts

GAINS = np.arange(-10, 66)  # class k <-> gain k - 10


def gain_probs(prepared: pd.DataFrame) -> np.ndarray:
    """P(yards gained = g) for g in GAINS, shape (n_plays, 76).

    Strength features are `posteam_spread`, `total_line` and `posteam_total`, read
    from `prepared` as-is, so the caller decides whether they hold pregame lines or
    market-implied values.
    """
    return artifacts.predict("fd_model", prepared)


def first_down_prob(prepared: pd.DataFrame) -> np.ndarray:
    probs = gain_probs(prepared)
    need = prepared["ydstogo"].to_numpy()[:, None]
    return (probs * (GAINS[None, :] >= need)).sum(axis=1)

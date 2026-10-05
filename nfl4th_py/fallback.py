"""Fallback to the original nfl4th for plays the market engine can't decide.

Used only for plays where the market-valued engine gives no recommendation (outcome
states outside the response model's support, or an unusable winner market). It runs the
original nfl4th calculation: win-probability valuation with pregame betting lines and
nfl4th's own field goal model. This is the only production use of `wp_baseline`, and
`decision.py` imports this module lazily, so the market path never loads the WP models
unless a play needs the fallback (enforced in tests/test_quarantine.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl4th_py.models import wp_baseline

ORIGINAL_COLUMNS = ["go_boost", "first_down_prob", "wp_fail", "wp_succeed", "go_wp", "fg_make_prob",
                    "miss_fg_wp", "make_fg_wp", "fg_wp", "punt_wp"]


def nfl4th_recommendation(df: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Original nfl4th columns plus `recommendation` (highest WP option) and `gap` (WP
    margin over the next-best option). `games` must carry the betting lines."""
    base = df[[c for c in df.columns if c not in ("kicker_player_id", "wind", "temp")]]
    out = wp_baseline.add_4th_probs_baseline(base.reset_index(drop=True), games)
    vals = np.column_stack([out["go_wp"], out["fg_wp"], out["punt_wp"].fillna(-np.inf)]).astype(float)
    order = np.sort(vals, axis=1)
    res = out[ORIGINAL_COLUMNS].copy()
    res["recommendation"] = np.array(["go", "fg", "punt"])[vals.argmax(axis=1)]
    res["gap"] = order[:, -1] - order[:, -2]
    return res

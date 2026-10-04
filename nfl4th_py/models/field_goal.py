"""Field goal make probability (nfl4th mgcv::bam GAM, exported as an exact lookup).

The GAM's inputs are an integer yardline_100 and a 4-level roof/era factor, so the
lookup over every combination reproduces it exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl4th_py import artifacts


def raw_make_prob(yardline_100: np.ndarray, fg_model_roof: np.ndarray) -> np.ndarray:
    lut = artifacts.fg_lookup()
    table = lut.set_index(["fg_model_roof", "yardline_100"])["fg_make_prob"]
    yl = np.asarray(yardline_100)
    if not np.all(np.mod(yl, 1) == 0):
        raise ValueError("FG lookup requires integer yardline_100")
    keys = pd.MultiIndex.from_arrays([np.asarray(fg_model_roof, dtype=object), yl.astype(int)])
    out = table.reindex(keys).to_numpy()
    if np.isnan(out).any():
        raise ValueError("yardline_100 / fg_model_roof outside the exported FG grid")
    return out


def make_prob(prepared: pd.DataFrame) -> np.ndarray:
    """nfl4th's post-processed make probability: linear decay beyond 58 yards
    (yardline_100 > 40) to 0 at 71 yards (yardline_100 >= 53)."""
    yl = prepared["yardline_100"].to_numpy()
    roof = prepared["fg_model_roof"].to_numpy()
    p = raw_make_prob(yl, roof)
    p58 = raw_make_prob(np.full(len(yl), 40), roof)
    p = np.where(yl > 40, (53 - yl) / 13 * p58, p)
    return np.where(yl >= 53, 0.0, p)

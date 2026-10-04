"""2-point conversion model (nfl4th two_pt_model) and XP make probability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl4th_py import artifacts
from nfl4th_py.models import field_goal


def two_point_prob(df: pd.DataFrame) -> np.ndarray:
    x = df.assign(era2=0)  # R forces era2 = 0
    return artifacts.predict("two_pt_model", x)


def xp_prob(df: pd.DataFrame) -> np.ndarray:
    return field_goal.raw_make_prob(np.full(len(df), 15), df["fg_model_roof"].to_numpy())

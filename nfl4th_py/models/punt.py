"""Empirical punt outcome distribution (nfl4th punt_df).

Rows give P(yardline_after, muff | yardline_100) for yardline_100 in 31..99, with
yardline_after measured from the punting team's view: 100 = return TD, equal to
yardline_100 (muff 0) = block, muff 1 = punting team recovers at yardline_after.
There are no rows at yardline_100 <= 30, where nfl4th never punts.
"""

from __future__ import annotations

import pandas as pd

from nfl4th_py import artifacts


def punt_distribution() -> pd.DataFrame:
    return artifacts.punt_table()

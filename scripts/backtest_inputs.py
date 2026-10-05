"""Turn Phase 4 dataset rows into decision-engine inputs (play frame + market snapshot).

Used by the Phase 8 backtest and tests. Prices are the pre-snap trade VWAPs, so this is
the information available just before each snap.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from nfl4th_py.models import kicker

CACHE = Path(__file__).resolve().parents[1] / "data_cache"


@cache
def _kicker_lookup() -> pd.DataFrame:
    return kicker.team_kickers(pd.read_parquet(CACHE / "kicks_2014_2025.parquet"))


@cache
def _weather() -> pd.DataFrame:
    w = pd.read_parquet(CACHE / "pbp_2025.parquet", columns=["game_id", "temp", "wind"])
    return w.groupby("game_id").first()


def engine_inputs(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    plays = pd.DataFrame(
        {
            "home_team": d["home_team"].to_numpy(),
            "away_team": d["away_team"].to_numpy(),
            "posteam": d["posteam"].to_numpy(),
            "type": np.where(d["season_type"] == "REG", "reg", "post"),
            "season": d["season"].to_numpy(),
            "qtr": d["qtr"].to_numpy(),
            "quarter_seconds_remaining": d["quarter_seconds_remaining"].to_numpy(),
            "ydstogo": d["ydstogo"].to_numpy(),
            "yardline_100": d["yardline_100"].to_numpy(),
            "score_differential": d["score_differential"].to_numpy(),
            "home_opening_kickoff": d["home_opening_kickoff"].to_numpy(),
            "posteam_timeouts_remaining": d["posteam_timeouts_remaining"].to_numpy(),
            "defteam_timeouts_remaining": d["defteam_timeouts_remaining"].to_numpy(),
        }
    )
    lk = _kicker_lookup()
    plays["kicker_player_id"] = [kicker.kicker_for(lk, g, t, s, w) for g, t, s, w in
                                 zip(d["game_id"], d["posteam"], d["season"], d["week"])]
    wx = _weather()
    plays["wind"] = d["game_id"].map(wx["wind"]).to_numpy(dtype=float)
    plays["temp"] = d["game_id"].map(wx["temp"]).to_numpy(dtype=float)
    snap = pd.DataFrame(
        {
            "price": d["pre_price"].to_numpy(),
            "bid_ask": d["candle_spread"].to_numpy(),
            "quote_age_s": d["pre_last_age_s"].to_numpy(),
            "n_trades": d["pre_n_trades"].to_numpy(),
            "halted": d["flag_pre_stale"].to_numpy(),
            "kalshi_spread_line": d["kalshi_spread_line"].to_numpy(),
            "kalshi_total_line": d["kalshi_total_line"].to_numpy(),
            "kalshi_spread_quality": d["kalshi_spread_quality"].to_numpy(),
            "kalshi_total_quality": d["kalshi_total_quality"].to_numpy(),
        }
    )
    return plays, snap

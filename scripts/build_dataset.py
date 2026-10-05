"""Build the Phase 4 dataset (2025 plays aligned to Kalshi prices).

Inputs (from scripts/backfill_kalshi.py): data_cache/{pbp_2025,markets_2025,schedule_2025}.parquet
and data_cache/kalshi/{trades,candles}/<series>/<game_id>.parquet.

Outputs:
  data_cache/kalshi_pregame_2025.parquet  per-game Kalshi pregame spread/total (Phase 3)
  data_cache/dataset/plays_2025.parquet   one row per play (see DATA_DICTIONARY.md)

Usage: python scripts/build_dataset.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nfl4th_py import dataset
from nfl4th_py.market import alignment, pregame

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data_cache"
KALSHI = CACHE / "kalshi"
SERIES = ("KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL")

PBP_COLS = [
    "game_id", "play_id", "season", "week", "season_type", "game_date", "home_team", "away_team",
    "posteam", "defteam", "roof", "home_opening_kickoff", "qtr", "down", "ydstogo", "yardline_100",
    "goal_to_go", "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
    "posteam_score", "defteam_score", "score_differential", "total_home_score", "total_away_score",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining", "play_type", "play_type_nfl", "desc",
    "yards_gained", "touchdown", "td_team", "pass_touchdown", "rush_touchdown", "return_touchdown",
    "interception", "fumble_lost", "safety", "penalty", "penalty_team", "penalty_yards",
    "first_down", "fourth_down_converted", "fourth_down_failed", "third_down_converted",
    "third_down_failed", "field_goal_result", "kick_distance", "extra_point_result",
    "two_point_conv_result", "two_point_attempt", "punt_blocked", "punt_inside_twenty", "touchback", "aborted_play",
    "qb_kneel", "qb_spike", "time_of_day", "end_clock_time", "spread_line", "total_line",
    "ep", "epa", "wp", "vegas_wp", "vegas_home_wp", "wpa",
]

log = logging.getLogger("build_dataset")


def _load(kind: str, series: str, gid: str) -> pd.DataFrame:
    path = KALSHI / kind / series / f"{gid}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def candle_quotes(candles: pd.DataFrame, home_ticker: str, snaps: np.ndarray) -> pd.DataFrame:
    """Home-market 1-minute candle quote (closed at or before each snap): bid, ask, age."""
    c = candles[candles["ticker"] == home_ticker].dropna(subset=["yes_bid_close", "yes_ask_close"])
    c = c.sort_values("end_period_ts")
    if c.empty:
        return pd.DataFrame({"candle_bid": np.nan, "candle_ask": np.nan, "candle_age_s": np.nan}, index=range(len(snaps)))
    ends = c["end_period_ts"].to_numpy(float)
    i = np.searchsorted(ends, snaps, side="right") - 1
    ok = i >= 0
    j = np.where(ok, i, 0)
    return pd.DataFrame(
        {
            "candle_bid": np.where(ok, c["yes_bid_close"].to_numpy()[j], np.nan),
            "candle_ask": np.where(ok, c["yes_ask_close"].to_numpy()[j], np.nan),
            "candle_age_s": np.where(ok, snaps - ends[j], np.nan),
        }
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pbp = pd.read_parquet(CACHE / "pbp_2025.parquet", columns=PBP_COLS)
    markets = pd.read_parquet(CACHE / "markets_2025.parquet")
    sched = pd.read_parquet(CACHE / "schedule_2025.parquet").set_index("game_id", drop=False)
    plays = dataset.prepare_plays(pbp)
    kickoff = plays.groupby("game_id")["snap_ts"].min()

    games = sorted(set(markets["game_id"]) & set(plays["game_id"]))
    strengths, parts = [], []
    for n, gid in enumerate(games, 1):
        mk = markets[markets["game_id"] == gid]
        candles = {s: _load("candles", s, gid) for s in SERIES}
        trades = _load("trades", "KXNFLGAME", gid)
        if trades.empty or any(not (KALSHI / "candles" / s / f"{gid}.parquet").exists() for s in SERIES):
            log.warning("%s: Kalshi data missing, skipped", gid)
            continue
        g = sched.loc[gid]
        strengths.append(pregame.game_strength(g, mk, candles, int(kickoff[gid])))

        win = mk[mk["series_ticker"] == "KXNFLGAME"].set_index("team")["ticker"]
        home_t, away_t = win[g["home_team"]], win[g["away_team"]]
        series = alignment.home_price_series(trades, home_t, away_t)
        gp = dataset.align_game(plays[plays["game_id"] == gid], series)
        cq = candle_quotes(candles["KXNFLGAME"], home_t, gp["snap_ts"].to_numpy(float))
        cq.index = gp.index
        gp = pd.concat([gp, cq], axis=1)
        gp["home_ticker"], gp["away_ticker"] = home_t, away_t
        parts.append(gp)
        if n % 25 == 0:
            log.info("%d/%d games", n, len(games))

    strength = pd.DataFrame(strengths)
    strength.to_parquet(CACHE / "kalshi_pregame_2025.parquet")
    d = dataset.finalize(pd.concat(parts, ignore_index=True))
    s = strength.set_index("game_id")
    home_spread = d["game_id"].map(s["kalshi_spread_line"])
    total = d["game_id"].map(s["kalshi_total_line"])
    d["kalshi_spread_line"] = home_spread
    d["kalshi_total_line"] = total
    d["kalshi_posteam_spread"] = np.where(d["is_home_pos"], home_spread, -home_spread)
    d["kalshi_posteam_total"] = (total + d["kalshi_posteam_spread"]) / 2
    d["kalshi_spread_quality"] = d["game_id"].map(s["kalshi_spread_quality"])
    d["kalshi_total_quality"] = d["game_id"].map(s["kalshi_total_quality"])
    # candle quote converted to the posteam's view (home market quote)
    d["candle_spread"] = d["candle_ask"] - d["candle_bid"]
    d["candle_mid_pos"] = np.where(
        d["is_home_pos"], (d["candle_bid"] + d["candle_ask"]) / 2, 1 - (d["candle_bid"] + d["candle_ask"]) / 2
    )
    out = CACHE / "dataset"
    out.mkdir(exist_ok=True)
    d.to_parquet(out / "plays_2025.parquet")
    log.info("wrote %d plays from %d games (%d usable)", len(d), d["game_id"].nunique(), int(d["usable"].sum()))


if __name__ == "__main__":
    main()

"""Input validation and feature engineering (port of R/helpers.R)."""

from __future__ import annotations

import numpy as np
import pandas as pd

SCHEDULES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"

REQUIRED_4TH_COLUMNS = [
    "home_team",
    "away_team",
    "posteam",
    "type",
    "season",
    "qtr",
    "quarter_seconds_remaining",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "home_opening_kickoff",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
]

# columns dropped before the schedule join so an nflfastR frame doesn't collide
DROP_COLS = [
    "game_id",
    "week",
    "model_roof",
    "roof",
    "era3",
    "era4",
    "home_total",
    "away_total",
    "total_line",
    "spread_line",
    "retractable",
    "dome",
    "outdoors",
]

_TEAM_FIXES = {
    "JAC": "JAX",
    "STL": "LA",
    "SL": "LA",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "SD": "LAC",
    "OAK": "LV",
}


def team_name_fn(s: pd.Series) -> pd.Series:
    return s.replace(_TEAM_FIXES)


def build_games(schedules: pd.DataFrame) -> pd.DataFrame:
    """Port of get_games_file(): nflverse schedules -> the per-game columns nfl4th joins."""
    g = schedules.loc[schedules["season"] > 2013].copy()
    roof = g["roof"].where(g["roof"].isin(["closed", "dome", "outdoors", "open", "retractable"]))
    g["type"] = np.where(g["game_type"] == "REG", "reg", "post")
    g["model_roof"] = np.select(
        [roof.isin(["open", "closed"]) | roof.isna(), roof == "retractable", roof == "dome"],
        ["retractable", "retractable", "dome"],
        "outdoors",
    )
    g["era0"] = 0
    g["era1"] = 0
    g["era2"] = 0
    g["era3"] = ((g["season"] > 2013) & (g["season"] <= 2017)).astype(int)
    g["era4"] = (g["season"] > 2017).astype(int)
    fg_roof = (roof == "outdoors").astype(int).astype(str)
    fg_era = (g["season"] >= 2020).astype(int).astype(str)
    g["fg_model_roof"] = fg_roof + fg_era
    g["home_total"] = (g["total_line"] + g["spread_line"]) / 2
    g["away_total"] = (g["total_line"] - g["spread_line"]) / 2
    g["retractable"] = (g["model_roof"] == "retractable").astype(int)
    g["dome"] = (g["model_roof"] == "dome").astype(int)
    g["outdoors"] = (g["model_roof"] == "outdoors").astype(int)
    g["roof"] = g["model_roof"]
    g["home_team"] = team_name_fn(g["home_team"])
    g["away_team"] = team_name_fn(g["away_team"])
    cols = [
        "game_id", "season", "type", "week", "away_team", "home_team", "espn",
        "fg_model_roof", "model_roof", "roof", "era0", "era1", "era2", "era3", "era4",
        "home_total", "away_total", "total_line", "spread_line", "retractable", "dome",
        "outdoors",
    ]
    return g[cols].reset_index(drop=True)


def load_games(url: str = SCHEDULES_URL) -> pd.DataFrame:
    return build_games(pd.read_csv(url, low_memory=False))


def validate_4th_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_4TH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"add_4th_probs input is missing columns: {missing}")


def prepare_df(df: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Port of prepare_df(): join the schedule and derive clock/strength columns."""
    out = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    out = out.merge(games, on=["home_team", "away_team", "type", "season"], how="left")
    qtr = out["qtr"]
    hok = out["home_opening_kickoff"]
    out["home_receive_2h_ko"] = np.select(
        [(qtr <= 2) & (hok == 1), (qtr <= 2) & (hok == 0)], [-1, 1], 0
    )
    out["down"] = 4
    out["half_seconds_remaining"] = np.where(
        qtr.isin([2, 4]), out["quarter_seconds_remaining"], out["quarter_seconds_remaining"] + 900
    )
    out["game_seconds_remaining"] = np.where(
        qtr <= 2, out["half_seconds_remaining"] + 1800, out["half_seconds_remaining"]
    )
    out["elapsed_share"] = (3600 - out["game_seconds_remaining"]) / 3600
    # home perspective; computed once here and (as in R) never refreshed later
    out["spread_time"] = out["spread_line"] * np.exp(-4 * out["elapsed_share"])
    is_home = out["posteam"] == out["home_team"]
    out["posteam_spread"] = np.where(is_home, out["spread_line"], -out["spread_line"])
    out["posteam_total"] = np.where(is_home, out["home_total"], out["away_total"])
    out["home_timeouts_remaining"] = np.where(
        is_home, out["posteam_timeouts_remaining"], out["defteam_timeouts_remaining"]
    )
    out["away_timeouts_remaining"] = np.where(
        out["posteam"] == out["away_team"],
        out["posteam_timeouts_remaining"],
        out["defteam_timeouts_remaining"],
    )
    out["original_posteam"] = out["posteam"]
    return out

"""Game-state transitions shared by every option (port of flip_team / flip_half)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def other_team(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(df["home_team"] == df["posteam"], df["away_team"], df["home_team"]),
        index=df.index,
    )


def flip_team(df: pd.DataFrame) -> pd.DataFrame:
    """Change of possession: opponent 1st & 10, score negated, 6 seconds off (floored at 0)."""
    out = df.copy()
    out["posteam"] = other_team(df)
    out["score_differential"] = -df["score_differential"]
    out["down"] = 1
    out["ydstogo"] = 10
    out["half_seconds_remaining"] = np.maximum(df["half_seconds_remaining"] - 6, 0)
    out["game_seconds_remaining"] = np.maximum(df["game_seconds_remaining"] - 6, 0)
    return out


def flip_half(df: pd.DataFrame) -> pd.DataFrame:
    """If Q2 has expired, move to the 3rd-quarter kickoff (2nd-half receiver at its 25).

    Matches R exactly: only a clock of exactly 0 triggers it (negative clocks from
    get_go_wp do not), and `home_opening_kickoff == 1` gives the 2nd-half ball to the
    away team.
    """
    out = df.copy()
    eoh = (df["qtr"] == 2) & (df["half_seconds_remaining"] == 0)
    hok = df["home_opening_kickoff"]
    out["posteam"] = np.select(
        [(hok == 1) & eoh, (hok == 0) & eoh], [df["away_team"], df["home_team"]], df["posteam"]
    )
    out["qtr"] = np.where(eoh, 3, df["qtr"])
    out["down"] = np.where(eoh, 1, df["down"])
    out["ydstogo"] = np.where(eoh, 10, df["ydstogo"])
    out["yardline_100"] = np.where(eoh, 75, df["yardline_100"])
    out["half_seconds_remaining"] = np.where(eoh, 1800, df["half_seconds_remaining"])
    out["game_seconds_remaining"] = np.where(eoh, 1800, df["game_seconds_remaining"])
    out["score_differential"] = np.where(
        (out["posteam"] != df["posteam"]) & eoh,
        -df["score_differential"],
        df["score_differential"],
    )
    out["home_receive_2h_ko"] = np.where(eoh, 0, df["home_receive_2h_ko"])
    return out


def defteam_timeouts(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(
            df["posteam"] == df["home_team"],
            df["away_timeouts_remaining"],
            df["home_timeouts_remaining"],
        ),
        index=df.index,
    )


def can_kneel_out(df: pd.DataFrame) -> pd.Series:
    """Team with the ball leads and the defense can't stop the clock (end_game_fn rule)."""
    dto = defteam_timeouts(df)
    gsr = df["game_seconds_remaining"]
    lead = df["score_differential"] > 0
    return lead & (
        ((gsr < 120) & (dto == 0)) | ((gsr < 80) & (dto == 1)) | ((gsr < 40) & (dto == 2))
    )

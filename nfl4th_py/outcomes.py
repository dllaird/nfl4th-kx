"""Expand each 4th-down decision into its possible post-play states and probabilities.

Port of the state logic in R/decision_functions.R, independent of how states are
valued. Every output row carries `play` (the row position in the prepared frame),
`prob`, and a full game state; values are always from `original_posteam`'s view.

R quirks reproduced deliberately (search "R quirk") so the benchmark matches nfl4th.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl4th_py import states
from nfl4th_py.models import conversion, field_goal, punt


def go_outcomes(prepared: pd.DataFrame, gain_probs: np.ndarray | None = None) -> pd.DataFrame:
    """One row per (play, distinct resulting yardline) for going for it.

    Columns added: `prob`, `gain`, `turnover` (failed attempt), `td` (yardline 0).
    A row with `turnover == 1 and td` is the opponent scoring on a backed-up loss.
    """
    if gain_probs is None:
        gain_probs = conversion.gain_probs(prepared)
    n = len(prepared)
    base = prepared.reset_index(drop=True)
    long = pd.DataFrame(
        {
            "play": np.repeat(np.arange(n), len(conversion.GAINS)),
            "gain": np.tile(conversion.GAINS, n),
            "prob": gain_probs.reshape(-1),
        }
    )
    yl = base["yardline_100"].to_numpy()[long["play"]]
    # gains longer than the field are touchdowns; pool their probability
    long["gain"] = np.where(long["gain"] > yl, yl, long["gain"]).astype(int)
    long = long.groupby(["play", "gain"], as_index=False, sort=True)["prob"].sum()
    out = base.iloc[long["play"]].reset_index(drop=True)
    out.insert(0, "play", long["play"].to_numpy())
    out["gain"] = long["gain"].to_numpy()
    out["prob"] = long["prob"].to_numpy()

    out["yardline_100"] = out["yardline_100"] - out["gain"]
    turnover = (out["gain"] < out["ydstogo"]).to_numpy()
    out["turnover"] = turnover.astype(int)
    out["down"] = 1
    out["yardline_100"] = np.where(turnover, 100 - out["yardline_100"], out["yardline_100"])
    out["posteam"] = np.where(turnover, states.other_team(out), out["posteam"])
    out["score_differential"] = np.where(turnover, -out["score_differential"], out["score_differential"])
    td = (out["yardline_100"] == 0).to_numpy()
    out["td"] = td
    out["score_differential"] = np.where(td, out["score_differential"] + 6, out["score_differential"])

    converted_not_td = ~turnover & (out["yardline_100"] > 0).to_numpy()
    for col in ("half_seconds_remaining", "game_seconds_remaining"):
        secs = out[col] - 6 - np.where(converted_not_td, out["runoff"], 0)
        # R quirk: `max(x, 0)` inside a grouped mutate returns one scalar per play,
        # max(all outcomes' clocks, 0), so every outcome gets the largest clock
        # (pre-play - 6, floored at 0). `runoff` is therefore never applied.
        out[col] = np.maximum(secs.groupby(out["play"]).transform("max"), 0)
    out["ydstogo"] = np.where(out["yardline_100"] < 10, out["yardline_100"], 10)
    return out


def td_pat_outcomes(td_rows: pd.DataFrame) -> pd.DataFrame:
    """For each TD row, the three PAT results (0/1/2 points) followed by a touchback
    kickoff: opponent 1st & 10 at its 25. Port of get_2pt_wp's state logic.

    R quirk: posteam_spread/posteam_total stay those of the original posteam even when
    the TD belongs to the opponent; that is what R feeds the 2-pt model.
    """
    from nfl4th_py.models import two_point

    td_rows = td_rows.reset_index(drop=True)
    td_rows = td_rows.assign(
        td_row=np.arange(len(td_rows)),
        prob_2pt=two_point.two_point_prob(td_rows),
        prob_1pt=two_point.xp_prob(td_rows),
    )
    parts = []
    for pts in (0, 1, 2):
        p = td_rows.copy()
        p["pts"] = pts
        p["score_differential"] = -p["score_differential"] - pts
        parts.append(p)
    out = pd.concat(parts, ignore_index=True)
    out["posteam"] = states.other_team(out)
    out["yardline_100"] = 75
    out["down"] = 1
    out["ydstogo"] = 10
    return states.flip_half(out)


def fg_outcomes(prepared: pd.DataFrame) -> pd.DataFrame:
    """Two rows per play: make (opponent at its 25 after kickoff, down 3 more) and miss
    (opponent at the spot of the kick, 92 - yardline_100, clipped to [1, 80])."""
    base = prepared.reset_index(drop=True).assign(play=np.arange(len(prepared)))
    p_make = field_goal.make_prob(base)

    make = states.flip_team(base)
    make["yardline_100"] = 75
    make["score_differential"] = make["score_differential"] - 3
    make["outcome"] = "make"
    make["prob"] = p_make

    miss = states.flip_team(base)
    spot = (100 - base["yardline_100"]) - 8
    miss["yardline_100"] = np.clip(spot, 1, 80)
    miss["outcome"] = "miss"
    miss["prob"] = 1 - p_make

    out = pd.concat([make, miss], ignore_index=True)
    return states.flip_half(out)


def punt_outcomes(prepared: pd.DataFrame) -> pd.DataFrame:
    """One row per punt-table outcome. Plays with yardline_100 <= 30 have no rows and
    get a single row with prob NaN, matching R's NA punt_wp."""
    base = prepared.reset_index(drop=True).assign(play=np.arange(len(prepared)))
    table = punt.punt_distribution().rename(columns={"pct": "prob"})
    merged = base.merge(table, on="yardline_100", how="left")
    out = states.flip_team(merged)
    ya = merged["yardline_after"]
    muff = merged["muff"] == 1
    ret_td = ya == 100
    keep_ball = (ret_td | muff).to_numpy()
    out["yardline_100"] = 100 - ya
    out["posteam"] = np.where(keep_ball, merged["original_posteam"], out["posteam"])
    out["yardline_100"] = np.where(muff, 100 - out["yardline_100"], out["yardline_100"])
    out["yardline_100"] = np.where(ret_td, 75, out["yardline_100"])
    sd = out["score_differential"].astype(float)
    # R's if_else propagates NA conditions (no punt row) into the score
    sd = np.where(ya.isna(), np.nan, np.where(ret_td, -sd - 7, sd))
    sd = np.where(merged["muff"].isna(), np.nan, np.where(muff, -sd, sd))
    out["score_differential"] = sd
    out["ydstogo"] = np.where(out["yardline_100"] < 10, out["yardline_100"], out["ydstogo"])
    out["muff"] = merged["muff"]
    out["yardline_after"] = ya
    return states.flip_half(out)

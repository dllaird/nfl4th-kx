"""BENCHMARK ONLY: the original nfl4th win-probability valuation.

Reproduces R/apply_win_prob.R (the mean of nfl4th's home-team WP model and nflfastR's
spread-adjusted WP model) and the original add_4th_probs()/add_2pt_probs() built on it.
It exists so the market-based approach can be compared against nfl4th.

Never import this from the production decision path; tests/test_quarantine.py fails
if anything outside the allow-list does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl4th_py import artifacts, outcomes, prep, states
from nfl4th_py.models import conversion, field_goal

_EP_WEIGHTS = np.array([7, -7, 3, -3, 2, -2, 0])  # TD, oTD, FG, oFG, S, oS, none


def expected_points(df: pd.DataFrame) -> np.ndarray:
    """nflfastR::calculate_expected_points. Eras come from season, `home` from posteam.

    R quirk: nfl4th passes the pre-play posteam/defteam timeouts here even after a
    change of possession.
    """
    season = df["season"]
    down = df["down"]
    x = pd.DataFrame(
        {
            "half_seconds_remaining": df["half_seconds_remaining"],
            "yardline_100": df["yardline_100"],
            "home": (df["posteam"] == df["home_team"]).astype(float),
            "retractable": (df["roof"] == "retractable").astype(float),
            "dome": (df["roof"] == "dome").astype(float),
            "outdoors": (df["roof"] == "outdoors").astype(float),
            "ydstogo": df["ydstogo"],
            "era0": (season <= 2001).astype(float),
            "era1": ((season > 2001) & (season <= 2005)).astype(float),
            "era2": ((season > 2005) & (season <= 2013)).astype(float),
            "era3": ((season > 2013) & (season <= 2017)).astype(float),
            "era4": (season > 2017).astype(float),
            "down1": np.where(down.isna(), np.nan, down == 1),
            "down2": np.where(down.isna(), np.nan, down == 2),
            "down3": np.where(down.isna(), np.nan, down == 3),
            "down4": np.where(down.isna(), np.nan, down == 4),
            "posteam_timeouts_remaining": df["posteam_timeouts_remaining"],
            "defteam_timeouts_remaining": df["defteam_timeouts_remaining"],
        },
        index=df.index,
    )
    probs = artifacts.predict("ep_model", x)
    return probs @ _EP_WEIGHTS


def calculate_win_probability(df: pd.DataFrame) -> np.ndarray:
    """WP of `original_posteam` in each state: mean of the two models."""
    is_home = (df["posteam"] == df["home_team"]).to_numpy()
    sd = df["score_differential"].to_numpy(dtype=float)
    home_sd = np.where(is_home, sd, -sd)
    ep = expected_points(df)
    # R quirk: elapsed_share / spread_time are the stale pre-play values from prepare_df
    elapsed = df["elapsed_share"].to_numpy(dtype=float)
    home_x = pd.DataFrame(
        {
            "home_receive_2h_ko": df["home_receive_2h_ko"].to_numpy(dtype=float),
            "spread_time": df["spread_time"].to_numpy(dtype=float),
            "home_posteam": is_home.astype(float),
            "half_seconds_remaining": df["half_seconds_remaining"].to_numpy(dtype=float),
            "game_seconds_remaining": df["game_seconds_remaining"].to_numpy(dtype=float),
            "Diff_Time_Ratio": home_sd / np.exp(-4 * elapsed),
            "home_score_differential": home_sd,
            "home_ep": np.where(is_home, ep, -ep),
            "ydstogo": df["ydstogo"].to_numpy(dtype=float),
            "home_yardline_100": np.where(
                is_home, df["yardline_100"], 100 - df["yardline_100"].to_numpy(dtype=float)
            ),
            # R quirk: away timeouts never reach this model
            "home_timeouts_remaining": df["home_timeouts_remaining"].to_numpy(dtype=float),
        }
    )
    home_wp = artifacts.predict("home_wp_model", home_x)
    orig_is_away = (df["original_posteam"] == df["away_team"]).to_numpy()
    home_wp = np.where(orig_is_away, 1 - home_wp, home_wp)

    qtr = df["qtr"].to_numpy()
    hok = df["home_opening_kickoff"].to_numpy()
    is_away = (df["posteam"] == df["away_team"]).to_numpy()
    receive_2h = ((qtr <= 2) & (hok == 1) & is_away) | ((qtr <= 2) & (hok == 0) & is_home)
    gsr = df["game_seconds_remaining"].to_numpy(dtype=float)
    fresh_elapsed = (3600 - gsr) / 3600
    spread = df["spread_line"].to_numpy(dtype=float)
    pos_spread = np.where(is_home, spread, -spread)
    fast_x = pd.DataFrame(
        {
            "receive_2h_ko": receive_2h.astype(float),
            "spread_time": pos_spread * np.exp(-4 * fresh_elapsed),
            "home": is_home.astype(float),
            "half_seconds_remaining": df["half_seconds_remaining"].to_numpy(dtype=float),
            "game_seconds_remaining": gsr,
            "Diff_Time_Ratio": sd / np.exp(-4 * fresh_elapsed),
            "score_differential": sd,
            "down": df["down"].to_numpy(dtype=float),
            "ydstogo": df["ydstogo"].to_numpy(dtype=float),
            "yardline_100": df["yardline_100"].to_numpy(dtype=float),
            "posteam_timeouts_remaining": np.where(
                is_away, df["away_timeouts_remaining"], df["home_timeouts_remaining"]
            ).astype(float),
            "defteam_timeouts_remaining": np.where(
                is_home, df["away_timeouts_remaining"], df["home_timeouts_remaining"]
            ).astype(float),
        }
    )
    fast_wp = artifacts.predict("wp_spread_model", fast_x)
    flipped = (df["posteam"] != df["original_posteam"]).to_numpy()
    fast_wp = np.where(flipped, 1 - fast_wp, fast_wp)
    return (home_wp + fast_wp) / 2


def _end_game(st: pd.DataFrame, wp: np.ndarray) -> np.ndarray:
    """end_game_fn: the team with the ball can kneel out the clock -> WP 0."""
    return np.where(states.can_kneel_out(st).to_numpy(), 0.0, wp)


def _pat_values(td_rows: pd.DataFrame) -> pd.DataFrame:
    """Per TD row: wp_0/1/2, conversion probs, wp_go1/wp_go2 and wp_td = max of the two."""
    pat = outcomes.td_pat_outcomes(td_rows)
    pat["wp"] = calculate_win_probability(pat)
    wide = pat.pivot(index="td_row", columns="pts", values="wp")
    first = pat.loc[pat["pts"] == 0].set_index("td_row")
    res = pd.DataFrame(
        {
            "wp_0": wide[0],
            "wp_1": wide[1],
            "wp_2": wide[2],
            "conv_2pt": first["prob_2pt"],
            "conv_1pt": first["prob_1pt"],
        }
    ).sort_index()
    res["wp_go2"] = res["conv_2pt"] * res["wp_2"] + (1 - res["conv_2pt"]) * res["wp_0"]
    res["wp_go1"] = res["conv_1pt"] * res["wp_1"] + (1 - res["conv_1pt"]) * res["wp_0"]
    res["wp_td"] = np.where(res["wp_go1"] > res["wp_go2"], res["wp_go1"], res["wp_go2"])
    return res


def go_values(prepared: pd.DataFrame) -> pd.DataFrame:
    pre = outcomes.go_outcomes(prepared)
    td = pre["td"].to_numpy()
    o = states.flip_half(pre)
    wp = calculate_win_probability(o)
    if td.any():
        # PAT values use the state before flip_half; R then substitutes them only where
        # the row still sits at yardline 0 after flip_half
        wp_td = np.full(len(o), np.nan)
        wp_td[td] = _pat_values(pre.loc[td])["wp_td"].to_numpy()
        use_td = (o["yardline_100"] == 0).to_numpy()
        wp = np.where(use_td, wp_td, wp)
    turnover = o["turnover"].to_numpy() == 1
    kneel = states.can_kneel_out(o).to_numpy()
    in_field = (o["yardline_100"] > 0).to_numpy()
    wp = np.where(kneel & ~turnover & in_field, 1.0, wp)
    wp = np.where(kneel & turnover, 0.0, wp)
    o["wp"] = wp

    prob = o["prob"].to_numpy()
    grp = o.assign(pw=prob * wp)
    res = pd.DataFrame(index=pd.RangeIndex(len(prepared), name="play"))
    res["go_wp"] = grp.groupby("play")["pw"].sum()
    by = grp.groupby(["play", "turnover"])[["prob", "pw"]].sum()
    succ = by.xs(0, level="turnover") if 0 in by.index.get_level_values(1) else None
    fail = by.xs(1, level="turnover") if 1 in by.index.get_level_values(1) else None
    res["first_down_prob"] = succ["prob"] if succ is not None else np.nan
    res["wp_succeed"] = succ["pw"] / succ["prob"] if succ is not None else np.nan
    res["wp_fail"] = fail["pw"] / fail["prob"] if fail is not None else np.nan
    return res


def fg_values(prepared: pd.DataFrame) -> pd.DataFrame:
    o = outcomes.fg_outcomes(prepared)
    o["wp"] = _end_game(o, calculate_win_probability(o))
    make = o.loc[o["outcome"] == "make"].set_index("play")
    miss = o.loc[o["outcome"] == "miss"].set_index("play")
    res = pd.DataFrame(
        {"fg_make_prob": make["prob"], "make_fg_wp": make["wp"], "miss_fg_wp": miss["wp"]}
    ).sort_index()
    res["fg_wp"] = res["fg_make_prob"] * res["make_fg_wp"] + (1 - res["fg_make_prob"]) * res["miss_fg_wp"]
    return res


def punt_values(prepared: pd.DataFrame) -> pd.DataFrame:
    o = outcomes.punt_outcomes(prepared)
    o["wp"] = _end_game(o, calculate_win_probability(o))
    pw = o["prob"] * o["wp"]
    total = pw.groupby(o["play"]).sum()
    # sum() with any NA is NA in R (plays inside the 31 have no punt rows)
    total[pw.isna().groupby(o["play"]).any()] = np.nan
    return total.rename("punt_wp").to_frame()


def add_4th_probs_baseline(df: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Original nfl4th add_4th_probs() (Path A input: a `type` column is required)."""
    prep.validate_4th_input(df)
    data = df.copy()
    if "runoff" not in data.columns:
        data["runoff"] = 0
    prepared = prep.prepare_df(data, games)
    if len(prepared) != len(df):
        raise ValueError("schedule join changed the number of rows; check team/type/season keys")
    res = go_values(prepared).join(fg_values(prepared)).join(punt_values(prepared))
    punt_prob = res["punt_wp"].fillna(0)
    res["go_boost"] = 100 * (res["go_wp"] - np.fmax(res["fg_wp"], punt_prob))
    cols = [
        "go_boost", "first_down_prob", "wp_fail", "wp_succeed", "go_wp", "fg_make_prob",
        "miss_fg_wp", "make_fg_wp", "fg_wp", "punt_wp",
    ]
    out = df.reset_index(drop=True).copy()
    for c in cols:
        out[c] = res[c].to_numpy()
    return out


def add_2pt_probs_baseline(df: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Original nfl4th add_2pt_probs(): values the PAT decision after a TD just scored."""
    data = df.copy()
    for c in ("ydstogo", "yardline_100"):
        if c not in data.columns:
            data[c] = np.nan
    prepared = prep.prepare_df(data, games)
    res = _pat_values(prepared.reset_index(drop=True))
    out = df.reset_index(drop=True).copy()
    for c in ("wp_0", "wp_1", "wp_2", "conv_1pt", "conv_2pt", "wp_go1", "wp_go2"):
        out[c] = res[c].to_numpy()
    return out


__all__ = [
    "add_2pt_probs_baseline",
    "add_4th_probs_baseline",
    "calculate_win_probability",
    "conversion",
    "field_goal",
]


# --------------------------------------------------------------------------------------
# Benchmark for the market-response model (Phase 5): value arbitrary observed states with
# the nfl4th WP models, using an implied spread solved so the pre-play WP matches the
# pre-play market price.


def _model_roof(roof: pd.Series) -> pd.Series:
    return roof.where(roof.isin(["dome", "outdoors"]), "retractable")


def states_frame(d: pd.DataFrame, which: str) -> pd.DataFrame:
    """nfl4th-style frame for the pre-play ('pre') or post-play ('post') state of Phase 4
    dataset rows. WP is always for the pre-play posteam (original_posteam). Kickoff
    states are valued as nfl4th does: receiving team 1st & 10 at its 25."""
    p = "" if which == "pre" else "post_"
    f = pd.DataFrame(index=d.index)
    f["home_team"], f["away_team"] = d["home_team"], d["away_team"]
    f["original_posteam"] = d["posteam"]
    f["posteam"] = d[f"{p}posteam"]
    f["season"] = d["season"]
    f["roof"] = _model_roof(d["roof"])
    f["home_opening_kickoff"] = d["home_opening_kickoff"]
    f["qtr"] = d[f"{p}qtr"]
    f["down"] = d[f"{p}down"]
    f["ydstogo"] = d[f"{p}ydstogo"]
    f["yardline_100"] = d[f"{p}yardline_100"]
    f["score_differential"] = d[f"{p}score_differential"]
    f["half_seconds_remaining"] = d[f"{p}half_seconds_remaining"]
    f["game_seconds_remaining"] = d[f"{p}game_seconds_remaining"]
    f["posteam_timeouts_remaining"] = d[f"{p}posteam_timeouts_remaining"]
    f["defteam_timeouts_remaining"] = d[f"{p}defteam_timeouts_remaining"]
    is_home = f["posteam"] == f["home_team"]
    f["home_timeouts_remaining"] = np.where(is_home, f["posteam_timeouts_remaining"], f["defteam_timeouts_remaining"])
    f["away_timeouts_remaining"] = np.where(is_home, f["defteam_timeouts_remaining"], f["posteam_timeouts_remaining"])
    kind = d[f"{p}kind"] if which == "post" else d["kind"]
    ko = (kind == "kickoff").to_numpy()
    f.loc[ko, "down"], f.loc[ko, "ydstogo"], f.loc[ko, "yardline_100"] = 1, 10, 75
    f["kind"] = kind
    f["elapsed_share"] = (3600 - f["game_seconds_remaining"]) / 3600
    f["home_receive_2h_ko"] = np.select(
        [(f["qtr"] <= 2) & (f["home_opening_kickoff"] == 1), (f["qtr"] <= 2) & (f["home_opening_kickoff"] == 0)],
        [-1, 1], 0,
    )
    # columns the 2-pt model reads for PAT states (set per spread in value_states)
    f["era2"], f["era3"], f["era4"] = 0, 0, 1
    f["outdoors"] = (f["roof"] == "outdoors").astype(int)
    f["dome"] = (f["roof"] == "dome").astype(int)
    f["retractable"] = (f["roof"] == "retractable").astype(int)
    f["fg_model_roof"] = (f["roof"] == "outdoors").astype(int).astype(str) + "1"
    return f


def value_states(f: pd.DataFrame, home_spread: np.ndarray, total: np.ndarray) -> np.ndarray:
    """WP of original_posteam in each state for a given home spread line and total.
    PAT states use nfl4th's TD valuation (best of kicking and going for 2)."""
    g = f.copy()
    g["spread_line"] = home_spread
    g["total_line"] = total
    g["spread_time"] = g["spread_line"] * np.exp(-4 * g["elapsed_share"])
    is_home = g["posteam"] == g["home_team"]
    g["posteam_spread"] = np.where(is_home, g["spread_line"], -g["spread_line"])
    g["posteam_total"] = (g["total_line"] + g["posteam_spread"]) / 2
    wp = calculate_win_probability(g)
    pat = (g["kind"] == "pat").to_numpy()
    if pat.any():
        wp[pat] = _pat_values(g.loc[pat])["wp_td"].to_numpy()
    return wp


def solve_home_spread(pre: pd.DataFrame, price: np.ndarray, total: np.ndarray, lo: float = -40.0, hi: float = 40.0, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Bisection for the home spread making the pre-play WP equal the market price.
    Returns (spread, matched) where matched is False if the price is outside the WP
    range reachable within [lo, hi] (the spread is then the nearer bound)."""
    a, b = np.full(len(pre), lo), np.full(len(pre), hi)
    fa = value_states(pre, a, total) - price
    fb = value_states(pre, b, total) - price
    matched = np.sign(fa) != np.sign(fb)
    nearer_lo = np.abs(fa) < np.abs(fb)
    for _ in range(iters):
        m = (a + b) / 2
        fm = value_states(pre, m, total) - price
        left = np.sign(fm) == np.sign(fa)
        a, fa = np.where(left, m, a), np.where(left, fm, fa)
        b = np.where(left, b, m)
    s = (a + b) / 2
    s = np.where(matched, s, np.where(nearer_lo, lo, hi))
    return s, matched

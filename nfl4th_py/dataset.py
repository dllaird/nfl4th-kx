"""Phase 4 training dataset: every 2025 play aligned to Kalshi winner-market prices.

One row per snap-type play (run, pass, punt, field goal, kickoff, extra point, kneel,
spike, and penalty no-plays; timeouts excluded). Each row holds the pre-play state, the play outcome, the
post-play state (the next play's pre-play state), the winner price just before the
snap and once settled after the play (just before the next snap), Kalshi pregame
strength, and quality / alignment flags. See DATA_DICTIONARY.md.

All prices are P(pre-play posteam wins) + 0.5 P(tie), since ties settle at $0.50.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl4th_py.market import alignment

PLAY_TYPES = ["run", "pass", "punt", "field_goal", "kickoff", "extra_point", "qb_kneel", "qb_spike", "no_play"]

STATE_COLS = [
    "posteam", "defteam", "qtr", "down", "ydstogo", "yardline_100", "goal_to_go",
    "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
    "posteam_score", "defteam_score", "score_differential", "total_home_score", "total_away_score",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
]


@dataclass(frozen=True)
class AlignConfig:
    pre_window_s: float = 5.0  # VWAP window ending at the snap
    post_window_s: float = 5.0  # VWAP window ending `post_gap_s` before the next snap
    post_gap_s: float = 1.0
    settle_min_s: float = 25.0  # play end -> next snap; shorter = market may not have settled
    stale_s: float = 60.0  # no trade within this long before a reference time = stale


def state_kind(play_type: pd.Series, down: pd.Series, two_point: pd.Series, yardline_100: pd.Series) -> pd.Series:
    """What kind of snap a state is: `scrimmage` (has a down), `pat` (extra point or
    2-point try) or `kickoff`. Penalty no-plays without a down are classified by spot
    (PAT/2-pt spots are inside the 20; kickoffs are from the kicking team's 35 or so)."""
    kind = np.where(
        down.notna(),
        "scrimmage",
        np.where(
            (play_type == "extra_point") | (two_point.fillna(0) == 1),
            "pat",
            np.where(play_type == "kickoff", "kickoff", np.where(yardline_100 < 20, "pat", "kickoff")),
        ),
    )
    return pd.Series(kind, index=play_type.index)


def _ts(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, utc=True, format="ISO8601")
    return t.map(lambda x: x.timestamp() if pd.notna(x) else np.nan).astype(float)


def prepare_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Snap-type plays with pre and post state. Post state = the next snap-type play's
    pre-play state in the same game (NaN after the game's last play)."""
    p = pbp[pbp["play_type"].isin(PLAY_TYPES)].copy()
    # timeouts are logged as no_play with out-of-order timestamps; they are not plays
    p = p[~p["desc"].fillna("").str.startswith("Timeout")]
    p = p.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    p["snap_ts"] = _ts(p["time_of_day"])
    p["end_ts"] = _ts(p["end_clock_time"])
    nxt = p.groupby("game_id")
    for c in STATE_COLS:
        p[f"post_{c}"] = nxt[c].shift(-1)
    p["kind"] = state_kind(p["play_type"], p["down"], p["two_point_attempt"], p["yardline_100"])
    p["post_kind"] = nxt["kind"].shift(-1)
    p["next_snap_ts"] = nxt["snap_ts"].shift(-1)
    p["next_play_id"] = nxt["play_id"].shift(-1)
    p["prev_end_ts"] = nxt["end_ts"].shift(1)
    p["is_home_pos"] = p["posteam"] == p["home_team"]
    p["post_same_posteam"] = p["post_posteam"] == p["posteam"]
    # receive_2h_ko: posteam gets the 2nd-half kickoff (1st half only)
    first_def = p.dropna(subset=["defteam"]).groupby("game_id")["defteam"].first()
    p["receive_2h_ko"] = ((p["qtr"] <= 2) & (p["posteam"] == p["game_id"].map(first_def))).astype(int)
    desc = p["desc"].fillna("")
    p["flag_replay"] = desc.str.contains(r"REVERSED|challenged|replay official|Replay Official", regex=True)
    p["flag_reversed"] = desc.str.contains("REVERSED")
    p["flag_penalty"] = p["penalty"].fillna(0).eq(1)
    p["flag_no_play"] = p["play_type"].eq("no_play")
    p["flag_missing_end"] = p["end_ts"].isna()
    # whole-second snap times (most PATs, ~25% of punts/FGs, pre-snap penalties) look
    # hand-entered; kept but flagged
    p["flag_coarse_time"] = ~p["time_of_day"].fillna("").str.contains(".", regex=False)
    prev_snap = nxt["snap_ts"].shift(1)
    p["flag_nonmonotonic_time"] = (p["snap_ts"] <= prev_snap) | (p["next_snap_ts"] <= p["snap_ts"])
    p["flag_qtr_change"] = p["post_qtr"].ne(p["qtr"]) & p["post_qtr"].notna()
    p["flag_last_play"] = p["next_snap_ts"].isna()
    return p


def align_game(plays: pd.DataFrame, series: pd.DataFrame, cfg: AlignConfig = AlignConfig()) -> pd.DataFrame:
    """Add pre/post winner prices (from the home-price trade series) to one game's plays."""
    out = []
    for r in plays.itertuples(index=False):
        pre = alignment.price_at(series, r.snap_ts, cfg.pre_window_s)
        pre_last, pre_last_age = alignment.last_trade_price(series, r.snap_ts)
        # the post window must start after the play ended (or at least after the snap)
        floor = r.end_ts if not np.isnan(r.end_ts) else r.snap_ts
        if np.isnan(r.next_snap_ts):
            post = alignment.WindowPrice(None, 0, 0.0, None, None)
            post_last, post_last_age = None, None
        else:
            t_post = r.next_snap_ts - cfg.post_gap_s
            post = alignment.price_at(series, t_post, cfg.post_window_s, t_start_min=floor)
            post_last, post_last_age = alignment.last_trade_price(series, t_post)
            if post_last_age is not None and t_post - post_last_age < floor:
                post_last, post_last_age = None, None
        out.append(
            (
                pre.price, pre.n_trades, pre.volume, pre.age_s, pre.dispersion, pre_last, pre_last_age,
                post.price, post.n_trades, post.volume, post.age_s, post.dispersion, post_last, post_last_age,
            )
        )
    cols = [
        "pre_home_vwap", "pre_n_trades", "pre_volume", "pre_age_s", "pre_dispersion", "pre_home_last", "pre_last_age_s",
        "post_home_vwap", "post_n_trades", "post_volume", "post_age_s", "post_dispersion", "post_home_last", "post_last_age_s",
    ]
    res = pd.DataFrame(out, columns=cols, index=plays.index)
    return pd.concat([plays, res], axis=1)


def finalize(df: pd.DataFrame, cfg: AlignConfig = AlignConfig()) -> pd.DataFrame:
    """Choose a price per side (VWAP, else last trade within `stale_s`), convert to the
    pre-play posteam's perspective, and set quality flags."""
    d = df.copy()
    for side in ("pre", "post"):
        vwap, last, age = d[f"{side}_home_vwap"], d[f"{side}_home_last"], d[f"{side}_last_age_s"]
        use_last = vwap.isna() & last.notna() & age.le(cfg.stale_s)
        home = vwap.where(vwap.notna(), last.where(use_last))
        d[f"{side}_price_source"] = np.select(
            [vwap.notna(), use_last], ["trades_vwap", "last_trade"], "none"
        )
        d[f"{side}_price"] = np.where(d["is_home_pos"], home, 1 - home)
        d[f"flag_{side}_stale"] = d[f"{side}_last_age_s"].isna() | d[f"{side}_last_age_s"].gt(cfg.stale_s)
    # post state seen from the pre-play posteam (the team whose price we track)
    d["post_pre_pos_has_ball"] = d["post_posteam"] == d["posteam"]
    d["post_score_differential_pre_view"] = np.where(
        d["post_pre_pos_has_ball"], d["post_score_differential"], -d["post_score_differential"]
    )
    settle = d["next_snap_ts"] - d["end_ts"].fillna(d["snap_ts"])
    d["settle_time_s"] = settle
    d["flag_unsettled"] = settle.lt(cfg.settle_min_s)
    eps = 1e-3
    for side in ("pre", "post"):
        p = d[f"{side}_price"].clip(eps, 1 - eps)
        d[f"{side}_logit"] = np.log(p / (1 - p))
    d["usable"] = (
        d["pre_price"].notna()
        & d["post_price"].notna()
        & ~d["flag_pre_stale"]
        & ~d["flag_nonmonotonic_time"]
        & ~d["flag_unsettled"]
    )
    return d

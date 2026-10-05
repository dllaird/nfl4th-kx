"""4th-down decision engine valued by Kalshi market prices (Phase 6).

For each 4th down:
  * outcome probabilities come from the ported nfl4th components: the conversion model
    (team strength = Kalshi pregame implied spread and total), the FG make model and the
    punt outcome table;
  * every resulting state (each converted yardline, TD, failed attempt, safety, FG
    make/miss, each punt outcome) is valued with the Phase 5 transition model, anchored
    on the live pre-snap winner price;
  * EV(option) = sum(prob * predicted post-play price). The option with the highest EV is
    recommended, with the gap and its uncertainty.

There is no fallback to a model-based WP. If the market, the ladders or the model's
support are not good enough, the play gets no recommendation and a reason.

Production states differ from nfl4th's on purpose, because the market model can value
the states that actually follow:
  * TD -> the PAT state (instead of nfl4th's PAT decision + touchback approximation)
  * FG make -> the kickoff state (instead of a touchback)
  * punt return TD -> the opponent's PAT state
  * loss into the team's own end zone -> safety (nfl4th treats it as a turnover or TD)
  * `runoff` is applied to converted non-TD outcomes (it is silently ignored in nfl4th)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from nfl4th_py import outcomes, prep
from nfl4th_py.market import pregame
from nfl4th_py.market.response import TransitionModel, logit
from nfl4th_py.models import field_goal, punt

MODEL_DIR = Path(__file__).parent / "data" / "response"
OPTIONS = ("go", "fg", "punt")


@dataclass(frozen=True)
class DecisionConfig:
    min_price: float = 0.03  # outside [min, max] the response model loses to "no move"
    max_price: float = 0.97
    max_quote_age_s: float = 30.0  # winner-market price must be this fresh
    max_bid_ask: float = 0.04  # winner-market bid/ask width
    # trades in the pre-snap window (historical) or resting book levels (live); off by
    # default because quote age and bid/ask width already catch thin markets
    min_trades: int = 0
    min_ladder_quality: float = 0.2  # Kalshi pregame spread and total fits
    # support = percentile of training-data typicality (see response.Support); out-of-fold
    # error roughly doubles below 0.05 and triples below 0.005
    min_support: float = 0.01  # probability-weighted support of an option's outcomes
    min_outcome_support: float = 0.002  # any outcome with prob >= 5% must have at least this
    max_qtr: int = 4  # OT is outside nfl4th's components (clock logic); refuse
    # end of regulation: onside kicks, kneel-downs and last-play states are rare in the
    # data and the response model misprices them (e.g. "down 1, kicking off, 9 s left");
    # the support measure does not catch this because it treats 20 s and 120 s as close
    q4_refuse_last_seconds: float = 120.0
    # game seconds used until the next snap, by outcome: medians measured on 2025 plays
    # (DECISION.md). nfl4th uses 6 s for everything, which misprices conversions, after
    # which the clock usually keeps running (~38 s).
    clock_convert: float = 38.0
    clock_convert_last2min: float = 14.0  # last 2 minutes of a half
    clock_fail: float = 5.0
    clock_td: float = 5.0
    clock_fg: float = 4.0
    clock_punt: float = 9.0


@cache
def response_model() -> TransitionModel:
    return TransitionModel.load(MODEL_DIR)


# --------------------------------------------------------------------------------------
# state construction


def _team_view(o: pd.DataFrame) -> pd.DataFrame:
    """nfl4th-style outcome rows (posteam, yardline_100, score_differential from the
    current posteam) -> response-model post columns from the original posteam's view."""
    has = (o["posteam"] == o["original_posteam"]).to_numpy()
    p = pd.DataFrame(index=o.index)
    p["post_has_ball"] = has.astype(int)
    p["post_kind"] = "scrimmage"
    p["post_fp"] = np.where(has, o["yardline_100"], 100 - o["yardline_100"])
    p["post_score_diff"] = np.where(has, o["score_differential"], -o["score_differential"])
    p["post_down"] = 1.0
    p["post_ydstogo"] = o["ydstogo"].astype(float)
    p["post_qtr"] = o["qtr"].astype(float)
    p["post_game_seconds_remaining"] = o["game_seconds_remaining"].astype(float)
    p["post_half_seconds_remaining"] = o["half_seconds_remaining"].astype(float)
    p["post_team_timeouts"] = o["posteam_timeouts_remaining_orig"]
    p["post_opp_timeouts"] = o["defteam_timeouts_remaining_orig"]
    return p


def _set_kind(p: pd.DataFrame, mask: np.ndarray, kind: str, team_has_ball: np.ndarray, yardline: float) -> None:
    """Make rows PAT or kickoff states with the training encoding: the team at the PAT is
    the scorer, the team 'with the ball' at a kickoff is the receiver; down missing,
    distance 0."""
    m = mask
    p.loc[m, "post_kind"] = kind
    p.loc[m, "post_has_ball"] = team_has_ball[m].astype(int)
    p.loc[m, "post_fp"] = np.where(team_has_ball[m], yardline, 100 - yardline)
    p.loc[m, "post_down"] = np.nan
    p.loc[m, "post_ydstogo"] = 0.0


def _end_of_half(p: pd.DataFrame, second_half_receiver_is_team: np.ndarray) -> None:
    """Q2 clock expired: the next snap is the 3rd-quarter kickoff."""
    eoh = ((p["post_qtr"] == 2) & (p["post_half_seconds_remaining"] <= 0)).to_numpy()
    if eoh.any():
        p.loc[eoh, "post_qtr"] = 3.0
        p.loc[eoh, "post_game_seconds_remaining"] = 1800.0
        p.loc[eoh, "post_half_seconds_remaining"] = 1800.0
        _set_kind(p, eoh, "kickoff", second_half_receiver_is_team, 35.0)


def _apply_clock(p: pd.DataFrame, pre: pd.DataFrame, used: np.ndarray) -> None:
    """Post-state clock = pre-play clock - `used` seconds (per row), floored at 0, and
    stopped at the two-minute warning when a Q2/Q4 play crosses it."""
    qsr0 = pre["quarter_seconds_remaining"].to_numpy(float)
    half0 = pre["half_seconds_remaining"].to_numpy(float)
    game0 = pre["game_seconds_remaining"].to_numpy(float)
    q = pre["qtr"].to_numpy()
    used = np.minimum(used, half0)
    crosses = np.isin(q, [2, 4]) & (qsr0 > 120) & (qsr0 - used < 120)
    used = np.where(crosses, qsr0 - 120, used)
    p["post_half_seconds_remaining"] = np.maximum(half0 - used, 0)
    p["post_game_seconds_remaining"] = np.maximum(game0 - used, 0)


def _clock_used(outcome: np.ndarray, pre: pd.DataFrame, runoff: np.ndarray, cfg: DecisionConfig) -> np.ndarray:
    late = np.isin(pre["qtr"].to_numpy(), [2, 4]) & (pre["quarter_seconds_remaining"].to_numpy(float) <= 120)
    convert = np.where(late, cfg.clock_convert_last2min, cfg.clock_convert)
    table = {"convert": convert, "td": cfg.clock_td, "fail": cfg.clock_fail, "safety": cfg.clock_fail,
             "make": cfg.clock_fg, "miss": cfg.clock_fg, "normal": cfg.clock_punt, "blocked": cfg.clock_punt,
             "muff": cfg.clock_punt, "return_td": cfg.clock_punt}
    used = np.array([np.asarray(table[o]) if np.ndim(table[o]) == 0 else table[o][i]
                     for i, o in enumerate(outcome)], dtype=float)
    # nfl4th's `runoff` input: extra clock after a non-TD conversion (e.g. a leading team
    # bleeding the clock); applied on top of nfl4th's 6 s, so it overrides when larger
    return np.where(outcome == "convert", np.maximum(used, 6 + runoff), used)


def go_states(prepared: pd.DataFrame) -> pd.DataFrame:
    o = outcomes.go_outcomes(prepared, r_quirks=False)
    o["posteam_timeouts_remaining_orig"] = o["posteam_timeouts_remaining"]
    o["defteam_timeouts_remaining_orig"] = o["defteam_timeouts_remaining"]
    p = _team_view(o)
    yl_after = (o["yardline_100"]).to_numpy()
    turnover = o["turnover"].to_numpy() == 1
    td = (~turnover) & (o["td"].to_numpy())
    # a loss behind the team's own goal line: nfl4th flips this to a turnover at
    # yardline <= 0 for the opponent; it is a safety (2 points, team free-kicks)
    safety = turnover & (yl_after <= 0)
    p.loc[td, "post_score_diff"] = o.loc[td, "score_differential"].to_numpy()
    _set_kind(p, td, "pat", np.ones(len(p), bool), 15.0)
    sd0 = o["score_differential"].to_numpy()  # opponent view after the flip (+6 if yl == 0)
    team_sd = np.where(yl_after == 0, -(sd0 - 6), -sd0) - 2
    p.loc[safety, "post_score_diff"] = team_sd[safety]
    _set_kind(p, safety, "kickoff", np.zeros(len(p), bool), 20.0)
    p["outcome"] = np.where(td, "td", np.where(safety, "safety", np.where(turnover, "fail", "convert")))
    p["prob"] = o["prob"].to_numpy()
    p["play"] = o["play"].to_numpy()
    p["gain"] = o["gain"].to_numpy()
    return p


def fg_states(prepared: pd.DataFrame) -> pd.DataFrame:
    base = prepared.reset_index(drop=True)
    pm = field_goal.make_prob(base)
    rows = []
    for outcome, prob in (("make", pm), ("miss", 1 - pm)):
        o = base.copy()
        o["play"] = np.arange(len(base))
        o["half_seconds_remaining"] = np.maximum(o["half_seconds_remaining"] - 6, 0)
        o["game_seconds_remaining"] = np.maximum(o["game_seconds_remaining"] - 6, 0)
        o["posteam_timeouts_remaining_orig"] = o["posteam_timeouts_remaining"]
        o["defteam_timeouts_remaining_orig"] = o["defteam_timeouts_remaining"]
        if outcome == "make":
            o["score_differential"] = o["score_differential"] + 3
            o["ydstogo"] = 0
        else:
            # opponent ball at the spot of the kick, never worse than its 20
            o["posteam"] = np.where(o["posteam"] == o["home_team"], o["away_team"], o["home_team"])
            o["score_differential"] = -o["score_differential"]
            o["yardline_100"] = np.clip((100 - base["yardline_100"]) - 8, 1, 80)
            o["ydstogo"] = np.where(o["yardline_100"] < 10, o["yardline_100"], 10)
        p = _team_view(o)
        if outcome == "make":
            _set_kind(p, np.ones(len(p), bool), "kickoff", np.zeros(len(p), bool), 35.0)
        p["outcome"], p["prob"], p["play"] = outcome, prob, o["play"].to_numpy()
        rows.append(p)
    return pd.concat(rows, ignore_index=True)


def punt_states(prepared: pd.DataFrame) -> pd.DataFrame:
    base = prepared.reset_index(drop=True).assign(play=lambda x: np.arange(len(x)))
    table = punt.punt_distribution()
    m = base.merge(table, on="yardline_100", how="inner")  # no rows inside the 31: no punt
    m["half_seconds_remaining"] = np.maximum(m["half_seconds_remaining"] - 6, 0)
    m["game_seconds_remaining"] = np.maximum(m["game_seconds_remaining"] - 6, 0)
    m["posteam_timeouts_remaining_orig"] = m["posteam_timeouts_remaining"]
    m["defteam_timeouts_remaining_orig"] = m["defteam_timeouts_remaining"]
    ya = m["yardline_after"].to_numpy()
    muff = m["muff"].to_numpy() == 1
    ret_td = ya == 100
    o = m.copy()
    keep = muff
    o["posteam"] = np.where(keep, m["posteam"], np.where(m["posteam"] == m["home_team"], m["away_team"], m["home_team"]))
    o["yardline_100"] = np.where(keep, ya, 100 - ya)
    o["score_differential"] = np.where(keep, m["score_differential"], -m["score_differential"])
    o["ydstogo"] = np.where(o["yardline_100"] < 10, o["yardline_100"], 10)
    p = _team_view(o)
    # return TD: the opponent is at its PAT, up 6 more
    p.loc[ret_td, "post_score_diff"] = (m["score_differential"].to_numpy() - 6)[ret_td]
    _set_kind(p, ret_td, "pat", np.zeros(len(p), bool), 15.0)
    blocked = (~muff) & (ya == m["yardline_100"].to_numpy())
    p["outcome"] = np.where(ret_td, "return_td", np.where(muff, "muff", np.where(blocked, "blocked", "normal")))
    p["prob"] = m["pct"].to_numpy()
    p["play"] = m["play"].to_numpy()
    return p


# --------------------------------------------------------------------------------------
# valuation


def _pre_columns(prepared: pd.DataFrame, pre_price: np.ndarray) -> pd.DataFrame:
    hok = prepared["home_opening_kickoff"].to_numpy()
    is_home = (prepared["posteam"] == prepared["home_team"]).to_numpy()
    # posteam receives the 2nd-half kickoff if it did not receive the opening one
    receives_2h = np.where(is_home, hok == 0, hok == 1)
    pre = pd.DataFrame(
        {
            "pre_logit": logit(pre_price),
            "qtr": prepared["qtr"].to_numpy(float),
            "game_seconds_remaining": prepared["game_seconds_remaining"].to_numpy(float),
            "half_seconds_remaining": prepared["half_seconds_remaining"].to_numpy(float),
            "down": 4.0,
            "ydstogo": prepared["ydstogo"].to_numpy(float),
            "yardline_100": prepared["yardline_100"].to_numpy(float),
            "score_differential": prepared["score_differential"].to_numpy(float),
            "posteam_timeouts_remaining": prepared["posteam_timeouts_remaining"].to_numpy(float),
            "defteam_timeouts_remaining": prepared["defteam_timeouts_remaining"].to_numpy(float),
            "receive_2h_ko": ((prepared["qtr"].to_numpy() <= 2) & receives_2h).astype(int),
            "kind": "scrimmage",
            "kalshi_posteam_spread": prepared["posteam_spread"].to_numpy(float),
            "kalshi_total_line": prepared["total_line"].to_numpy(float),
        }
    )
    pre["receives_2h"] = receives_2h
    pre["quarter_seconds_remaining"] = prepared["quarter_seconds_remaining"].to_numpy(float)
    pre["runoff"] = prepared["runoff"].fillna(0).to_numpy(float)
    return pre


def value_option(
    states: pd.DataFrame, pre: pd.DataFrame, model: TransitionModel, cfg: DecisionConfig = DecisionConfig()
) -> pd.DataFrame:
    """Set each outcome's clock, attach the pre-play columns, handle end of half/game,
    predict each outcome."""
    s = states.copy()
    rows_pre = pre.iloc[s["play"].to_numpy()].reset_index(drop=True)
    s = s.reset_index(drop=True)
    used = _clock_used(s["outcome"].to_numpy(), rows_pre, rows_pre["runoff"].to_numpy(), cfg)
    _apply_clock(s, rows_pre, used)
    _end_of_half(s, pre["receives_2h"].to_numpy()[s["play"].to_numpy()])
    t = pd.concat([pre.iloc[s["play"].to_numpy()].reset_index(drop=True), s.reset_index(drop=True)], axis=1)
    pred = model.predict(t)
    out = s.reset_index(drop=True).copy()
    out["price"] = pred["price"].to_numpy()
    out["sd_logit"] = pred["sd_logit"].to_numpy()
    out["support"] = pred["support"].to_numpy()
    # game over: the result is known
    over = ((t["post_qtr"] >= 4) & (t["post_game_seconds_remaining"] <= 0)).to_numpy()
    decided = over & (t["post_score_diff"].to_numpy() != 0)
    out.loc[decided, "price"] = (t.loc[decided, "post_score_diff"] > 0).astype(float).to_numpy()
    out.loc[decided, "sd_logit"] = 0.0
    out.loc[decided, "support"] = 1.0
    # tied at the end of regulation -> OT, which the model barely supports
    out.loc[over & ~decided, "support"] = 0.0
    p = out["price"].to_numpy()
    out["sd_price"] = p * (1 - p) * out["sd_logit"].to_numpy()  # delta method
    return out


def summarize(valued: pd.DataFrame, n_plays: int, option: str) -> pd.DataFrame:
    v = valued.assign(pw=valued["prob"] * valued["price"], psd=valued["prob"] * valued["sd_price"],
                      psup=valued["prob"] * valued["support"])
    g = v.groupby("play")
    res = pd.DataFrame(index=pd.RangeIndex(n_plays, name="play"))
    tot = g["prob"].sum()
    res[f"{option}_ev"] = g["pw"].sum() / tot
    # outcome errors share one model and one game, so they are treated as fully
    # correlated within an option (a conservative sd)
    res[f"{option}_sd"] = g["psd"].sum() / tot
    res[f"{option}_support"] = g["psup"].sum() / tot
    big = v[v["prob"] >= 0.05]
    res[f"{option}_min_support"] = big.groupby("play")["support"].min()
    return res


# --------------------------------------------------------------------------------------
# market checks and the public entry point

SNAPSHOT_COLUMNS = {
    "price": "P(posteam wins) + 0.5 P(tie): winner-market mid or pre-snap trade VWAP",
    "bid_ask": "winner-market bid/ask width (NaN if unknown)",
    "quote_age_s": "seconds since the price was observed",
    "n_trades": "trades (historical) or resting levels (live) behind the price",
    "halted": "market paused / inactive",
    "kalshi_spread_line": "Kalshi pregame implied home margin",
    "kalshi_total_line": "Kalshi pregame implied total",
    "kalshi_spread_quality": "ladder fit quality 0-1",
    "kalshi_total_quality": "ladder fit quality 0-1",
}


def _market_reasons(snap: pd.DataFrame, prepared: pd.DataFrame, cfg: DecisionConfig) -> list[list[str]]:
    reasons: list[list[str]] = [[] for _ in range(len(snap))]

    def add(mask, text):
        for i in np.flatnonzero(np.asarray(mask, bool)):
            reasons[i].append(text)

    price = snap["price"]
    add(price.isna(), "no winner-market price")
    add(~price.isna() & ~price.between(cfg.min_price, cfg.max_price),
        f"winner price outside [{cfg.min_price}, {cfg.max_price}]")
    add(snap["quote_age_s"].isna() | (snap["quote_age_s"] > cfg.max_quote_age_s),
        f"winner price stale (> {cfg.max_quote_age_s:.0f}s)")
    add(snap["bid_ask"].notna() & (snap["bid_ask"] > cfg.max_bid_ask), f"winner market wider than {cfg.max_bid_ask}")
    add(snap["n_trades"].fillna(0) < cfg.min_trades, "winner market too thin")
    add(snap["halted"].fillna(False).astype(bool), "winner market halted")
    add(snap["kalshi_spread_line"].isna() | (snap["kalshi_spread_quality"].fillna(0) < cfg.min_ladder_quality),
        "Kalshi pregame spread ladder unusable")
    add(snap["kalshi_total_line"].isna() | (snap["kalshi_total_quality"].fillna(0) < cfg.min_ladder_quality),
        "Kalshi pregame total ladder unusable")
    add(prepared["qtr"].to_numpy() > cfg.max_qtr, "overtime not supported")
    add((prepared["qtr"].to_numpy() == 4) & (prepared["quarter_seconds_remaining"].to_numpy() <= cfg.q4_refuse_last_seconds),
        f"final {cfg.q4_refuse_last_seconds:.0f}s of regulation not supported")
    return reasons


def add_4th_probs(
    df: pd.DataFrame,
    market_snapshot: pd.DataFrame,
    games: pd.DataFrame | None = None,
    config: DecisionConfig = DecisionConfig(),
    model: TransitionModel | None = None,
    return_outcomes: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """nfl4th's add_4th_probs() valued with Kalshi markets.

    `df`: nfl4th Path A input columns (see prep.REQUIRED_4TH_COLUMNS), optional `runoff`.
    `market_snapshot`: one row per row of `df` with SNAPSHOT_COLUMNS.
    `games`: prep.build_games() schedule table (roof etc.); loaded from nflverse if None.
    Pregame betting lines in `games` are replaced by the Kalshi values in the snapshot.

    Returns `df` plus the original nfl4th columns (WPs are market-implied prices here),
    market fields, per-outcome predictions, EV/uncertainty per option, and
    `recommendation` / `no_recommendation_reason`. With `return_outcomes=True`, also
    returns {option: per-outcome table (play, outcome, prob, state, price, sd, support)}.
    """
    prep.validate_4th_input(df)
    missing = [c for c in SNAPSHOT_COLUMNS if c not in market_snapshot]
    if missing:
        raise ValueError(f"market_snapshot is missing {missing}")
    if len(market_snapshot) != len(df):
        raise ValueError("market_snapshot must have one row per play")
    model = model or response_model()
    data = df.reset_index(drop=True).copy()
    snap = market_snapshot.reset_index(drop=True).copy()
    for c in SNAPSHOT_COLUMNS:
        if c == "halted":
            snap[c] = snap[c].map(lambda v: bool(v) if pd.notna(v) else False)
        else:
            snap[c] = pd.to_numeric(snap[c], errors="coerce").astype(float)
    if "runoff" not in data:
        data["runoff"] = 0
    games = prep.load_games() if games is None else games

    # Kalshi pregame lines replace the betting lines, per game in this frame
    keys = data[["home_team", "away_team", "type", "season"]].copy()
    g = keys.merge(games, on=["home_team", "away_team", "type", "season"], how="left")
    strength = pd.DataFrame(
        {"game_id": g["game_id"], "kalshi_spread_line": snap["kalshi_spread_line"],
         "kalshi_total_line": snap["kalshi_total_line"]}
    ).drop_duplicates("game_id")
    kalshi_games = pregame.with_kalshi_lines(games[games["game_id"].isin(strength["game_id"])], strength)
    prepared = prep.prepare_df(data, kalshi_games)
    if len(prepared) != len(data):
        raise ValueError("schedule join changed the number of rows")

    reasons = _market_reasons(snap, prepared, config)
    ok = np.array([not r for r in reasons])
    n = len(prepared)
    out = data.copy()
    res = pd.DataFrame(index=pd.RangeIndex(n))
    valued = {}
    if ok.any():
        sub = prepared.loc[ok].reset_index(drop=True)
        idx = np.flatnonzero(ok)
        pre = _pre_columns(sub, snap.loc[ok, "price"].to_numpy(float))
        for option, builder in (("go", go_states), ("fg", fg_states), ("punt", punt_states)):
            states = builder(sub)
            if states.empty:  # e.g. no punt rows inside the 31
                continue
            v = value_option(states, pre, model, config)
            v["play"] = idx[v["play"].to_numpy()]
            valued[option] = v
        for option, v in valued.items():
            s = summarize(v, n, option)
            res = res.join(s)
        go = valued["go"]
        conv = go[go["outcome"].isin(["convert", "td"])]
        fail = go[go["outcome"].isin(["fail", "safety"])]
        res["first_down_prob"] = conv.groupby("play")["prob"].sum()
        res["wp_succeed"] = conv.assign(pw=conv["prob"] * conv["price"]).groupby("play")["pw"].sum() / res["first_down_prob"]
        fail_p = fail.groupby("play")["prob"].sum()
        res["wp_fail"] = fail.assign(pw=fail["prob"] * fail["price"]).groupby("play")["pw"].sum() / fail_p
        fg = valued["fg"]
        res["fg_make_prob"] = fg[fg["outcome"] == "make"].set_index("play")["prob"]
        res["make_fg_wp"] = fg[fg["outcome"] == "make"].set_index("play")["price"]
        res["miss_fg_wp"] = fg[fg["outcome"] == "miss"].set_index("play")["price"]
        res["td_wp"] = go[go["outcome"] == "td"].groupby("play")["price"].mean()
        res["go_turnover_wp"] = fail[fail["outcome"] == "fail"].assign(
            pw=lambda x: x["prob"] * x["price"]).groupby("play")["pw"].sum() / fail_p
        punts = valued.get("punt", pd.DataFrame(columns=["outcome", "play", "prob", "price"]))
        for o in ("normal", "blocked", "muff", "return_td"):
            sub_o = punts[punts["outcome"] == o]
            res[f"punt_{o}_prob"] = sub_o.groupby("play")["prob"].sum()
            res[f"punt_{o}_wp"] = sub_o.assign(pw=sub_o["prob"] * sub_o["price"]).groupby("play")["pw"].sum() / res[f"punt_{o}_prob"]

    for c in [f"{o}_{k}" for o in OPTIONS for k in ("ev", "sd", "support", "min_support")]:
        if c not in res:
            res[c] = np.nan
    res["go_wp"], res["fg_wp"], res["punt_wp"] = res["go_ev"], res["fg_ev"], res["punt_ev"]

    # recommendation
    rec, reason, gap, gap_sd, p_best = [], [], [], [], []
    for i in range(n):
        why = list(reasons[i])
        evs = {o: res.at[i, f"{o}_ev"] for o in OPTIONS if pd.notna(res.at[i, f"{o}_ev"])}
        if not why:
            for o in list(evs):
                if res.at[i, f"{o}_support"] < config.min_support or (
                    pd.notna(res.at[i, f"{o}_min_support"]) and res.at[i, f"{o}_min_support"] < config.min_outcome_support
                ):
                    why.append(f"{o}: outcome states outside model support")
        if why or len(evs) < 2:
            rec.append(None)
            reason.append("; ".join(why) if why else "fewer than two options available")
            gap.append(np.nan), gap_sd.append(np.nan), p_best.append(np.nan)
            continue
        order = sorted(evs, key=evs.get, reverse=True)
        best, second = order[0], order[1]
        g_ = evs[best] - evs[second]
        sd = float(np.hypot(res.at[i, f"{best}_sd"], res.at[i, f"{second}_sd"]))
        rec.append(best), reason.append(None), gap.append(g_), gap_sd.append(sd)
        p_best.append(float(norm.cdf(g_ / sd)) if sd > 0 else 1.0)
    res["recommendation"] = rec
    res["no_recommendation"] = res["recommendation"].isna()
    res["no_recommendation_reason"] = reason
    res["ev_gap"] = gap
    res["ev_gap_sd"] = gap_sd
    res["prob_best_option_is_best"] = p_best
    res["go_boost"] = 100 * (res["go_ev"] - np.fmax(res["fg_ev"], res["punt_ev"].fillna(-np.inf)))
    res.loc[~np.isfinite(res["go_boost"]), "go_boost"] = np.nan

    # market fields
    for c in ("price", "bid_ask", "quote_age_s", "n_trades", "halted"):
        out[f"winner_{c}"] = snap[c].to_numpy()
    for c in ("kalshi_spread_line", "kalshi_total_line", "kalshi_spread_quality", "kalshi_total_quality"):
        out[c] = snap[c].to_numpy()
    out["kalshi_posteam_spread"] = prepared["posteam_spread"].to_numpy()
    for c in res.columns:
        out[c] = res[c].to_numpy()
    out.attrs["config"] = asdict(config)
    return (out, valued) if return_outcomes else out


def log_decisions(result: pd.DataFrame, raw_quotes: list[dict] | None, directory: Path | str) -> Path:
    """Append decisions (all returned columns) plus the raw quotes they used to
    `directory/date=YYYY-MM-DD/decisions-<timestamp>.parquet` for later analysis."""
    import json
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    out = result.copy()
    out.attrs = {}
    out["decided_at_utc"] = now.isoformat()
    out["config_json"] = json.dumps(result.attrs.get("config", {}))
    out["raw_quotes_json"] = [json.dumps(q, default=str) for q in (raw_quotes or [None] * len(out))]
    part = Path(directory) / f"date={now:%Y-%m-%d}"
    part.mkdir(parents=True, exist_ok=True)
    path = part / f"decisions-{now:%H%M%S%f}.parquet"
    out.to_parquet(path)
    return path

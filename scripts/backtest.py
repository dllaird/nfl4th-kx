"""Phase 8 decision backtest on 2025 4th downs (out-of-sample).

Each game's 4th downs are valued with the response model from the CV fold that did not
train on that game (data_cache/response_folds). Compares the market engine with the
original nfl4th (wp_baseline, betting lines), checks EV calibration against what the
market did after the play, and measures sensitivity to price, spread and staleness.

Writes reports/backtest_plays.parquet and reports/backtest_summary.json.
Usage: python scripts/backtest.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_inputs import engine_inputs  # noqa: E402

from nfl4th_py import decision, prep  # noqa: E402
from nfl4th_py.market import alignment  # noqa: E402
from nfl4th_py.market.response import TransitionModel  # noqa: E402
from nfl4th_py.models.kicker import KickerFGModel  # noqa: E402
from nfl4th_py.models import wp_baseline  # noqa: E402  (benchmark only)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data_cache"
REPORTS = ROOT / "reports"
log = logging.getLogger("backtest")
OPTS = ["go", "fg", "punt"]


def actual_choice(d: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select([d["play_type"].isin(["run", "pass"]), d["play_type"] == "field_goal", d["play_type"] == "punt"],
                  ["go", "fg", "punt"], None),
        index=d.index,
    )


# kicker model fit on 2014-2024 only, so 2025 FG probabilities are out-of-sample
FG_MODEL = KickerFGModel.load(CACHE / "kicker_model_le2024.json")


def run_engine(f: pd.DataFrame, games: pd.DataFrame, folds: pd.Series, models: dict, snap_override=None,
               config: decision.DecisionConfig = decision.DecisionConfig()) -> pd.DataFrame:
    plays, snap = engine_inputs(f)
    if snap_override is not None:
        snap = snap_override(snap, f)
    out = []
    # games outside training (no Kalshi pregame lines) are refused by the engine anyway
    fold = f["game_id"].map(folds).fillna(0).astype(int).to_numpy()
    for k, model in models.items():
        m = fold == k
        if m.any():
            r = decision.add_4th_probs(plays[m].reset_index(drop=True), snap[m].reset_index(drop=True), games,
                                       model=model, fg_model=FG_MODEL, config=config)
            r.index = f.index[m]
            out.append(r)
    return pd.concat(out).loc[f.index]


def rec_of(df: pd.DataFrame, cols=("go_wp", "fg_wp", "punt_wp")) -> pd.Series:
    v = np.column_stack([df[c].astype(float).fillna(-np.inf) for c in cols])
    r = pd.Series(np.array(OPTS)[v.argmax(1)], index=df.index)
    return r.where(np.isfinite(v.max(1)))


def flip_rate(base: pd.DataFrame, other: pd.DataFrame) -> dict:
    both = base["recommendation"].notna() & other["recommendation"].notna()
    flips = (base.loc[both, "recommendation"] != other.loc[both, "recommendation"]).mean()
    dgap = (other.loc[both, "go_boost"] - base.loc[both, "go_boost"]).abs().mean()
    return {"n": int(both.sum()), "flip_rate": float(flips), "mean_abs_change_go_boost_pts": float(dgap)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    d = pd.read_parquet(CACHE / "dataset" / "plays_2025.parquet")
    f = d[(d["down"] == 4) & d["play_type"].isin(["run", "pass", "punt", "field_goal"])].copy()
    f.index = pd.RangeIndex(len(f))
    games = prep.build_games(pd.read_parquet(CACHE / "schedule_2025.parquet"))
    folds = pd.read_parquet(CACHE / "response_folds" / "folds.parquet").set_index("game_id")["fold"]
    models = {k: TransitionModel.load(CACHE / "response_folds" / str(k)) for k in range(5)}
    f["actual"] = actual_choice(f)
    summary: dict = {"n_4th_downs": int(len(f))}

    # ---- market engine (out-of-sample) and original nfl4th
    eng = run_engine(f, games, folds, models)
    plays, _ = engine_inputs(f)
    orig = wp_baseline.add_4th_probs_baseline(plays.drop(columns=["kicker_player_id", "wind", "temp"]), games)
    f["engine_rec"] = eng["recommendation"]
    f["nfl4th_rec"] = rec_of(orig)
    for o in OPTS:
        f[f"engine_{o}_ev"] = eng[f"{o}_ev"]
        f[f"engine_{o}_sd"] = eng[f"{o}_sd"]
        f[f"nfl4th_{o}_wp"] = orig[f"{o}_wp"]
    for c in ("ev_gap", "ev_gap_sd", "prob_best_option_is_best", "no_recommendation_reason", "go_boost",
              "first_down_prob", "fg_make_prob", "fg_offered"):
        f[f"engine_{c}"] = eng[c]
    f["nfl4th_go_boost"] = orig["go_boost"]

    rec = f["engine_rec"].notna()
    summary["engine_recommendations"] = f["engine_rec"].value_counts().to_dict()
    summary["engine_no_recommendation"] = int((~rec).sum())
    reasons = f.loc[~rec, "engine_no_recommendation_reason"].str.split("; ").explode()
    summary["refusal_reasons"] = reasons.str.replace(r"^(go|fg|punt): ", "", regex=True).value_counts().to_dict()
    summary["nfl4th_recommendations"] = f["nfl4th_rec"].value_counts().to_dict()

    r = f[rec].copy()
    summary["agreement_engine_vs_nfl4th"] = float((r["engine_rec"] == r["nfl4th_rec"]).mean())
    summary["confusion_engine_rows_nfl4th_cols"] = pd.crosstab(r["engine_rec"], r["nfl4th_rec"]).to_dict()
    summary["coach_followed_engine"] = float((r["actual"] == r["engine_rec"]).mean())
    summary["coach_followed_nfl4th"] = float((r["actual"] == r["nfl4th_rec"]).mean())

    def ev(row, prefix, opt, suffix):
        return row[f"{prefix}_{opt}_{suffix}"] if isinstance(opt, str) else np.nan

    r["engine_ev_of_nfl4th_rec"] = [ev(x, "engine", x["nfl4th_rec"], "ev") for _, x in r.iterrows()]
    r["engine_ev_of_engine_rec"] = [ev(x, "engine", x["engine_rec"], "ev") for _, x in r.iterrows()]
    r["nfl4th_wp_of_engine_rec"] = [ev(x, "nfl4th", x["engine_rec"], "wp") for _, x in r.iterrows()]
    r["nfl4th_wp_of_nfl4th_rec"] = [ev(x, "nfl4th", x["nfl4th_rec"], "wp") for _, x in r.iterrows()]
    dis = r[r["engine_rec"] != r["nfl4th_rec"]]
    summary["disagreements"] = {
        "n": int(len(dis)),
        "market_ev_gain_engine_vs_nfl4th_pts_mean": float(100 * (dis["engine_ev_of_engine_rec"] - dis["engine_ev_of_nfl4th_rec"]).mean()),
        "nfl4th_wp_gain_nfl4th_vs_engine_pts_mean": float(100 * (dis["nfl4th_wp_of_nfl4th_rec"] - dis["nfl4th_wp_of_engine_rec"]).mean()),
        "share_engine_gap_within_1sd": float((dis["engine_ev_gap"] < dis["engine_ev_gap_sd"]).mean()),
        "share_nfl4th_margin_lt_1pt": float(
            (dis["nfl4th_wp_of_nfl4th_rec"] - dis["nfl4th_wp_of_engine_rec"]).lt(0.01).mean()),
    }

    # ---- what the market did: EV of the option actually taken vs the settled price
    took = f[f["actual"].notna() & f["usable"]].copy()
    took["engine_ev_actual"] = [x[f"engine_{x['actual']}_ev"] for _, x in took.iterrows()]
    took["nfl4th_wp_actual"] = [x[f"nfl4th_{x['actual']}_wp"] for _, x in took.iterrows()]
    cal = {}
    for name, col in [("engine", "engine_ev_actual"), ("nfl4th", "nfl4th_wp_actual")]:
        t = took[took[col].notna() & took["pre_price"].between(0.03, 0.97)]
        cal[name] = {
            "n": int(len(t)),
            "mean_post_minus_ev_pts": float(100 * (t["post_price"] - t[col]).mean()),
            "mae_pts": float(100 * (t["post_price"] - t[col]).abs().mean()),
            "corr_ev_vs_post": float(np.corrcoef(t[col], t["post_price"])[0, 1]),
            "mean_ev_minus_pre_pts": float(100 * (t[col] - t["pre_price"]).mean()),
        }
        for o in OPTS:
            to = t[t["actual"] == o]
            cal[name][f"{o}_mean_post_minus_ev_pts"] = float(100 * (to["post_price"] - to[col]).mean())
            cal[name][f"{o}_n"] = int(len(to))
    t = took[took["pre_price"].between(0.03, 0.97)]
    cal["market_mean_post_minus_pre_pts"] = float(100 * (t["post_price"] - t["pre_price"]).mean())
    summary["ev_calibration_vs_settled_price"] = cal

    # disagreement plays, with what the market did
    cols = ["game_id", "qtr", "quarter_seconds_remaining", "ydstogo", "yardline_100", "score_differential",
            "engine_rec", "engine_ev_gap", "engine_ev_gap_sd", "nfl4th_rec", "nfl4th_go_boost", "actual",
            "pre_price", "post_price", "desc"]
    dis_out = dis.assign(market_move=dis["post_price"] - dis["pre_price"])[cols + ["market_move"]]
    dis_out.to_parquet(REPORTS / "backtest_disagreements.parquet")
    by_follow = dis_out.dropna(subset=["market_move"]).assign(
        followed=lambda x: np.where(x["actual"] == x["engine_rec"], "engine", np.where(x["actual"] == x["nfl4th_rec"], "nfl4th", "neither")))
    summary["disagreement_market_move_by_choice_pts"] = (
        by_follow.groupby("followed")["market_move"].agg(["mean", "size"]).assign(mean=lambda x: 100 * x["mean"]).to_dict("index"))

    # ---- FG eligibility threshold: how coverage and recommendations change
    fg_thr = {}
    for thr in (0.0, 0.005, 0.01, 0.02, 0.05):
        o = run_engine(f, games, folds, models, config=decision.DecisionConfig(fg_min_make_prob=thr))
        rr = o["recommendation"]
        fg_thr[str(thr)] = {"recommended": int(rr.notna().sum()), "go": int((rr == "go").sum()),
                            "fg": int((rr == "fg").sum()), "punt": int((rr == "punt").sum()),
                            "refused_fg_support": int(o["no_recommendation_reason"].fillna("").str.contains("fg: outcome").sum())}
    summary["fg_min_make_prob_sweep"] = fg_thr
    summary["kicker_known_share"] = float(plays["kicker_player_id"].notna().mean())

    # ---- sensitivity
    base = eng
    sens = {}
    for dp in (-0.01, 0.01):
        o = run_engine(f, games, folds, models, lambda s, _f, dp=dp: s.assign(price=s["price"] + dp))
        sens[f"price_{dp:+.2f}"] = flip_rate(base, o)
    for ds in (-1.0, 1.0):
        def shift(s, ff, ds=ds):
            # +ds points toward the possession team
            sign = np.where(ff["is_home_pos"].to_numpy(), 1.0, -1.0)
            return s.assign(kalshi_spread_line=s["kalshi_spread_line"] + ds * sign)
        o = run_engine(f, games, folds, models, shift)
        sens[f"posteam_spread_{ds:+.0f}"] = flip_rate(base, o)

    # staleness: the price as of s seconds before the snap, from the trade tape
    stale_prices = {s: np.full(len(f), np.nan) for s in (5, 10, 20, 30, 60)}
    for gid, g in f.groupby("game_id"):
        tr = pd.read_parquet(CACHE / "kalshi" / "trades" / "KXNFLGAME" / f"{gid}.parquet")
        series = alignment.home_price_series(tr, g["home_ticker"].iat[0], g["away_ticker"].iat[0])
        for s in stale_prices:
            for i, row in zip(g.index, g.itertuples()):
                w = alignment.price_at(series, row.snap_ts - s, 5.0)
                if w.price is not None:
                    stale_prices[s][i] = w.price if row.is_home_pos else 1 - w.price
    for s, arr in stale_prices.items():
        o = run_engine(f, games, folds, models, lambda sn, _f, arr=arr: sn.assign(price=arr, quote_age_s=0.0))
        fr = flip_rate(base, o)
        fr["flip_rate_per_second"] = fr["flip_rate"] / s
        sens[f"stale_{s}s"] = fr
    summary["sensitivity"] = sens

    # ---- how often markets were unusable (all plays = samples of in-game moments)
    a = d.copy()
    summary["market_usability_all_plays"] = {
        "n_plays": int(len(a)),
        "no_price": float(a["pre_price"].isna().mean()),
        "price_outside_0.03_0.97": float((~a["pre_price"].between(0.03, 0.97)).mean()),
        "stale_gt_30s": float((a["pre_last_age_s"].isna() | (a["pre_last_age_s"] > 30)).mean()),
        "candle_bid_ask_gt_0.04": float((a["candle_spread"] > 0.04).mean()),
        "no_trade_60s_before_snap": float(a["flag_pre_stale"].mean()),
        "games_without_usable_pregame_ladders": int(
            a.groupby("game_id")[["kalshi_spread_line", "kalshi_total_line"]].first().isna().any(axis=1).sum()),
    }

    f.to_parquet(REPORTS / "backtest_plays.parquet")
    (REPORTS / "backtest_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()

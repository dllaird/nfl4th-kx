"""Phase 5: train and evaluate the market-response models.

Validation, always split by game:
  * time holdout   train weeks 1-14 (15% of those games for early stopping), test weeks
                   15-22 (late regular season + playoffs)
  * grouped 5-fold CV over all games, so every play (incl. all 4th downs) gets an
    out-of-sample prediction

Models compared on identical rows:
  persistence   post = pre (the market doesn't move)
  transition    (a) TransitionModel
  value         (b) ValueSurface, re-anchored to the pre-play price
  wp_baseline   nfl4th WP models with the home spread solved to match the pre-play price
                (scrimmage pre-states only; benchmark only)

Writes nfl4th_py/data/response/final (all-season models for Phase 6),
data_cache/response_holdout_models (weeks 1-14 models),
reports/response_oof.parquet and reports/response_metrics.json.

Usage: python scripts/train_response.py [--skip-baseline]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nfl4th_py.market import response as R
from nfl4th_py.models import wp_baseline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_cache" / "dataset" / "plays_2025.parquet"
REPORTS = ROOT / "reports"
MODEL_DIR = ROOT / "nfl4th_py" / "data" / "response"
HOLDOUT_WEEK = 15
IN_RANGE = (0.03, 0.97)

log = logging.getLogger("train_response")


def load() -> pd.DataFrame:
    d = pd.read_parquet(DATA)
    keep = (
        d["usable"]
        & d["kalshi_posteam_spread"].notna()
        & d["kalshi_total_line"].notna()
        & d["post_kind"].notna()
        & d["post_score_differential"].notna()
    )
    d = d[keep].copy()
    d["post_kind"] = d["post_kind"].astype(str)
    d["kind"] = d["kind"].astype(str)
    d["y_logit"] = d["post_logit"]
    return d.reset_index(drop=True)


def value_training_states(all_plays: pd.DataFrame, games: set[str]) -> pd.DataFrame:
    p = all_plays[
        all_plays["game_id"].isin(games)
        & all_plays["pre_price"].notna()
        & ~all_plays["flag_pre_stale"]
        & all_plays["kalshi_posteam_spread"].notna()
        & all_plays["kalshi_total_line"].notna()
    ]
    p = p.assign(kind=p["kind"].astype(str))
    return R.states_from_dataset(p)


def split_val(games: list[str], frac: float = 0.15, seed: int = 7) -> set[str]:
    rng = np.random.default_rng(seed)
    g = np.array(sorted(games))
    return set(rng.choice(g, size=max(1, int(len(g) * frac)), replace=False))


def fit_pair(d: pd.DataFrame, all_plays: pd.DataFrame, train_games: set[str]):
    val_games = split_val(sorted(train_games))
    tr = d[d["game_id"].isin(train_games - val_games)]
    va = d[d["game_id"].isin(val_games)]
    t_tr, t_va = R.transitions_from_dataset(tr), R.transitions_from_dataset(va)
    tm = R.TransitionModel().fit(t_tr, tr["y_logit"].to_numpy(), t_va, va["y_logit"].to_numpy())
    vs = R.ValueSurface().fit(
        value_training_states(all_plays, train_games - val_games), value_training_states(all_plays, val_games)
    )
    return tm, vs


def predict_all(tm: R.TransitionModel, vs: R.ValueSurface, rows: pd.DataFrame) -> pd.DataFrame:
    t = R.transitions_from_dataset(rows)
    tp = tm.predict(t)
    return pd.DataFrame(
        {
            "pred_persistence": rows["pre_logit"].to_numpy(),
            "pred_transition": tp["logit"].to_numpy(),
            "sd_transition": tp["sd_logit"].to_numpy(),
            "support_transition": tp["support"].to_numpy(),
            "pred_value": vs.predict_logit(t),
        },
        index=rows.index,
    )


def baseline_predictions(rows: pd.DataFrame) -> pd.Series:
    """wp_baseline with solved spread; NaN for non-scrimmage pre-states."""
    out = pd.Series(np.nan, index=rows.index)
    sc = rows[rows["kind"] == "scrimmage"]
    if sc.empty:
        return out
    pre = wp_baseline.states_frame(sc, "pre")
    post = wp_baseline.states_frame(sc, "post")
    total = sc["kalshi_total_line"].to_numpy()
    spread, matched = wp_baseline.solve_home_spread(pre, sc["pre_price"].to_numpy(), total)
    wp = wp_baseline.value_states(post, spread, total)
    out.loc[sc.index] = R.logit(wp)
    log.info("baseline: spread solved for %.1f%% of rows", 100 * matched.mean())
    return out


# --------------------------------------------------------------------------------------
# metrics

MODELS = ["persistence", "transition", "value", "wp_baseline"]


def metrics(df: pd.DataFrame, models=MODELS) -> dict:
    # compare every model on identical rows: those where all available models predict
    present = [m for m in models if f"pred_{m}" in df and df[f"pred_{m}"].notna().any()]
    df = df[df[[f"pred_{m}" for m in present]].notna().all(axis=1)]
    y = df["y_logit"].to_numpy()
    yp = df["post_price"].to_numpy()
    out = {"n": int(len(df))}
    sse_persist = np.sum((df["pred_persistence"].to_numpy() - y) ** 2)
    for m in models:
        col = f"pred_{m}"
        if col not in df or df[col].isna().all():
            continue
        ok = df[col].notna().to_numpy()
        e = df[col].to_numpy()[ok] - y[ok]
        ps = R.expit(df[col].to_numpy()[ok])
        persist_e = df["pred_persistence"].to_numpy()[ok] - y[ok]
        out[m] = {
            "n": int(ok.sum()),
            "rmse_logit": float(np.sqrt(np.mean(e**2))),
            "mae_logit": float(np.mean(np.abs(e))),
            "mae_price": float(np.mean(np.abs(ps - yp[ok]))),
            # share of the persistence squared error removed (1 = perfect, 0 = no better
            # than assuming the price doesn't move, < 0 = worse)
            "skill_vs_persistence": float(1 - np.sum(e**2) / max(np.sum(persist_e**2), 1e-12)),
            "ece_price": calibration_error(ps, yp[ok]),
        }
    _ = sse_persist
    if "transition" in present and len(df) >= 20:
        for other in ("wp_baseline", "value", "persistence"):
            if other in present:
                out[f"rmse_diff_transition_minus_{other}"] = bootstrap_rmse_diff(df, "transition", other)
    return out


def bootstrap_rmse_diff(df: pd.DataFrame, a: str, b: str, reps: int = 500, seed: int = 3) -> dict:
    """RMSE(a) - RMSE(b) on logits with a 90% CI from a game-clustered bootstrap."""
    y = df["y_logit"].to_numpy()
    ea2 = (df[f"pred_{a}"].to_numpy() - y) ** 2
    eb2 = (df[f"pred_{b}"].to_numpy() - y) ** 2
    codes, uniq = pd.factorize(df["game_id"])
    sa = np.bincount(codes, ea2)
    sb = np.bincount(codes, eb2)
    n = np.bincount(codes)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(reps):
        g = rng.integers(0, len(uniq), len(uniq))
        w = np.bincount(g, minlength=len(uniq))
        nn = (w * n).sum()
        diffs.append(np.sqrt((w * sa).sum() / nn) - np.sqrt((w * sb).sum() / nn))
    est = float(np.sqrt(ea2.mean()) - np.sqrt(eb2.mean()))
    lo, hi = np.quantile(diffs, [0.05, 0.95])
    return {"diff": est, "ci90": [float(lo), float(hi)]}


def calibration_error(pred: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(pred, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(pred[m].mean() - actual[m].mean())
    return float(err)


def segments(df: pd.DataFrame) -> dict[str, pd.Series]:
    pt = df["play_type"]
    fourth = df["down"] == 4
    seg = {
        "all": pd.Series(True, index=df.index),
        "in_range": df["pre_price"].between(*IN_RANGE),
        "pass": pt == "pass", "run": pt == "run", "punt": pt == "punt", "field_goal": pt == "field_goal",
        "kickoff": pt == "kickoff", "extra_point": pt == "extra_point", "no_play": pt == "no_play",
        "touchdown": df["touchdown"] == 1,
        "turnover": (df["interception"] == 1) | (df["fumble_lost"] == 1),
        "possession_change": ~df["post_pre_pos_has_ball"].astype(bool),
        "4th_go": fourth & pt.isin(["run", "pass"]),
        "4th_go_converted": fourth & (df["fourth_down_converted"] == 1),
        "4th_go_failed": fourth & (df["fourth_down_failed"] == 1),
        "4th_fg": fourth & (pt == "field_goal"),
        "4th_fg_made": fourth & (df["field_goal_result"] == "made"),
        "4th_fg_missed_or_blocked": fourth & pt.eq("field_goal") & df["field_goal_result"].ne("made"),
        "4th_punt": fourth & (pt == "punt"),
        "4th_all": fourth & pt.isin(["run", "pass", "field_goal", "punt"]),
        "src_vwap_both": (df["pre_price_source"] == "trades_vwap") & (df["post_price_source"] == "trades_vwap"),
        "src_any_last_trade": (df["pre_price_source"] == "last_trade") | (df["post_price_source"] == "last_trade"),
        "coarse_time": df["flag_coarse_time"],
        "replay": df["flag_replay"],
        "ot": df["qtr"] == 5,
        "last_2min_close": (df["game_seconds_remaining"] <= 120) & (df["score_differential"].abs() <= 8),
    }
    return {k: v.fillna(False).astype(bool) for k, v in seg.items()}


def report(df: pd.DataFrame) -> dict:
    out = {}
    for name, mask in segments(df).items():
        sub = df[mask & df["pre_price"].between(*IN_RANGE)] if name != "all" else df
        if len(sub) >= 5:
            out[name] = metrics(sub)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    REPORTS.mkdir(exist_ok=True)

    all_plays = pd.read_parquet(DATA)
    d = load()
    games = sorted(d["game_id"].unique())
    log.info("%d transitions from %d games", len(d), len(games))

    # ---- time holdout
    train_games = set(d.loc[d["week"] < HOLDOUT_WEEK, "game_id"])
    tm, vs = fit_pair(d, all_plays, train_games)
    test = d[~d["game_id"].isin(train_games)]
    hold = pd.concat([test, predict_all(tm, vs, test)], axis=1)
    if not args.skip_baseline:
        hold["pred_wp_baseline"] = baseline_predictions(test)
    log.info("holdout: transition best_iteration %d, value best_iteration %d",
             tm.booster.best_iteration, vs.booster.best_iteration)
    tm.save(ROOT / "data_cache" / "response_holdout_models")
    vs.save(ROOT / "data_cache" / "response_holdout_models")

    # ---- grouped 5-fold CV over all games
    rng = np.random.default_rng(11)
    fold_of = dict(zip(games, rng.permutation(np.arange(len(games)) % 5)))
    oof = []
    for k in range(5):
        tr_games = {g for g in games if fold_of[g] != k}
        tmk, vsk = fit_pair(d, all_plays, tr_games)
        tmk.save(ROOT / "data_cache" / "response_folds" / str(k))
        rows = d[d["game_id"].map(fold_of) == k]
        oof.append(pd.concat([rows, predict_all(tmk, vsk, rows)], axis=1))
        log.info("fold %d done (%d rows)", k, len(rows))
    oof = pd.concat(oof)
    pd.Series(fold_of, name="fold").rename_axis("game_id").reset_index().to_parquet(
        ROOT / "data_cache" / "response_folds" / "folds.parquet")
    if not args.skip_baseline:
        oof["pred_wp_baseline"] = baseline_predictions(oof)

    # ---- final models on all games (for Phase 6)
    tmf, vsf = fit_pair(d, all_plays, set(games))
    z = (oof["y_logit"] - oof["pred_transition"]) / oof["sd_transition"]
    inr = oof["pre_price"].between(*IN_RANGE)
    tmf.sd_scale = float(np.quantile(z[inr].abs(), 0.90) / 1.645)
    log.info("sd_scale from out-of-fold residuals: %.3f", tmf.sd_scale)
    for k in range(5):  # fold models share the out-of-fold sd calibration
        fold_dir = ROOT / "data_cache" / "response_folds" / str(k)
        mk = R.TransitionModel.load(fold_dir)
        mk.sd_scale = tmf.sd_scale
        mk.save(fold_dir)
    tmf.save(MODEL_DIR)
    vsf.save(MODEL_DIR)

    keep = [
        "game_id", "play_id", "kind", "post_kind", "week", "play_type", "down", "qtr", "pre_price", "post_price", "y_logit",
        "pre_logit", "pre_price_source", "post_price_source", "flag_coarse_time", "fourth_down_converted",
        "fourth_down_failed", "field_goal_result", "touchdown", "interception", "fumble_lost",
        "post_pre_pos_has_ball", "flag_replay", "game_seconds_remaining", "score_differential", "desc",
    ] + [c for c in oof.columns if c.startswith(("pred_", "sd_", "support_"))]
    oof[keep].to_parquet(REPORTS / "response_oof.parquet")
    hold[[c for c in keep if c in hold]].to_parquet(REPORTS / "response_holdout.parquet")

    res = {"holdout_weeks_15_22": report(hold), "cv5_all_games": report(oof)}
    (REPORTS / "response_metrics.json").write_text(json.dumps(res, indent=1))
    for split, r in res.items():
        print(f"\n== {split}")
        for seg in ["all", "in_range", "4th_all", "4th_go_converted", "4th_go_failed", "4th_fg_made", "4th_fg_missed_or_blocked", "4th_punt"]:
            if seg in r:
                row = {m: round(r[seg][m]["rmse_logit"], 3) for m in MODELS if m in r[seg]}
                d_wp = r[seg].get("rmse_diff_transition_minus_wp_baseline", {})
                ci = f" | trans-wp {d_wp['diff']:+.3f} [{d_wp['ci90'][0]:+.3f},{d_wp['ci90'][1]:+.3f}]" if d_wp else ""
                n_common = r[seg][MODELS[0]]["n"]
                print(f"{seg:26s} n={n_common:6d} {row}{ci}")


if __name__ == "__main__":
    main()

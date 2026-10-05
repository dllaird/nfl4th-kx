"""Fit and validate the kicker-specific FG model (nfl4th_py/models/kicker.py).

1. Tune the kicker priors on 2023 and 2024 field goals, each fit on earlier seasons.
2. Out-of-sample test: fit on 2014-2024, score 2025 field goals against nfl4th's GAM
   (with its 58-yard decay and 0% from the 53 yard line out) and a league-only curve.
3. Save the 2014-2024 model (data_cache, for the 2025 backtest) and the 2014-2025 model
   (nfl4th_py/data, for production).

Writes reports/kicker_metrics.json. Usage: python scripts/train_kicker.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl4th_py import artifacts
from nfl4th_py.models import kicker

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data_cache"
REPORTS = ROOT / "reports"
BANDS = [0, 30, 40, 50, 55, 60, 80]


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def nfl4th_prob(df: pd.DataFrame) -> np.ndarray:
    """nfl4th's FG probability for these kicks (yardline_100 = distance - 17)."""
    lut = artifacts.fg_lookup().set_index(["fg_model_roof", "yardline_100"])["fg_make_prob"]
    yl = np.clip(np.round(df["kick_distance"].to_numpy() - 17), 1, 99).astype(int)
    roof = np.where(df["roof"].eq("outdoors"), "1", "0").astype(object) + np.where(df["season"] >= 2020, "1", "0")
    p = lut.reindex(pd.MultiIndex.from_arrays([roof, yl])).to_numpy()
    p40 = lut.reindex(pd.MultiIndex.from_arrays([roof, np.full(len(yl), 40)])).to_numpy()
    p = np.where(yl > 40, (53 - yl) / 13 * p40, p)
    return np.where(yl >= 53, 0.0, p)


def score(df: pd.DataFrame, preds: dict[str, np.ndarray]) -> dict:
    y = df["made"].to_numpy(float)
    out = {}
    band = pd.cut(df["kick_distance"], BANDS, right=False)
    for name, p in preds.items():
        row = {"n": int(len(y)), "logloss": logloss(p, y), "brier": float(np.mean((p - y) ** 2))}
        bands = {}
        for b in band.cat.categories:
            m = (band == b).to_numpy()
            if m.sum():
                bands[str(b)] = {"n": int(m.sum()), "made": float(y[m].mean()), "pred": float(p[m].mean()),
                                 "logloss": logloss(p[m], y[m])}
        row["bands"] = bands
        row["made_kicks_given_0pct"] = int(((p <= 0) & (y == 1)).sum())
        out[name] = row
    return out


def main() -> None:
    kicks = kicker.prepare_kicks(pd.read_parquet(CACHE / "kicks_2014_2025.parquet"))
    fg = lambda d: d[d["is_xp"] == 0]  # noqa: E731
    report: dict = {}

    # 1. tune priors on two validation seasons (each fit only on earlier seasons); one
    # season alone is too noisy for the leg-strength spread
    grid = []
    for sa, sr in itertools.product([0.1, 0.15, 0.2, 0.3], [0.1, 0.2, 0.3, 0.45]):
        losses = []
        for year in (2023, 2024):
            tr, te = kicks[kicks["season"] < year], fg(kicks[kicks["season"] == year])
            m = kicker.KickerFGModel(sd_accuracy=sa, sd_range=sr).fit(tr)
            p = m.predict(te["kick_distance"], te["kicker_player_id"], te["outdoors"], te["season"], te["wind"], te["temp"])
            losses.append(logloss(p, te["made"].to_numpy()))
        grid.append({"sd_accuracy": sa, "sd_range": sr, "logloss_2023": losses[0], "logloss_2024": losses[1],
                     "logloss_mean": float(np.mean(losses))})
        print({k: round(v, 4) for k, v in grid[-1].items()}, flush=True)
    best = min(grid, key=lambda g: g["logloss_mean"])
    report["tuning"] = {"grid": grid, "best": best}

    # 2. out-of-sample 2025
    tr, te = kicks[kicks["season"] <= 2024], fg(kicks[kicks["season"] == 2025])
    m = kicker.KickerFGModel(sd_accuracy=best["sd_accuracy"], sd_range=best["sd_range"]).fit(tr)
    args = (te["kick_distance"], None, te["outdoors"], te["season"], te["wind"], te["temp"])
    preds = {
        "kicker_model": m.predict(te["kick_distance"], te["kicker_player_id"], *args[2:]),
        "league_curve": m.predict(*args),
        "nfl4th_gam": nfl4th_prob(te),
    }
    report["holdout_2025"] = score(te, preds)
    m.save(CACHE / "kicker_model_le2024.json")

    # 3. production fit on everything
    mf = kicker.KickerFGModel(sd_accuracy=best["sd_accuracy"], sd_range=best["sd_range"]).fit(kicks)
    mf.save()
    dist = np.arange(20, 81, 5)
    report["league_curve_outdoors_2025"] = {int(d): float(p) for d, p in zip(dist, mf.predict(dist, None, 1, 2025))}
    report["league_curve_indoors_2025"] = {int(d): float(p) for d, p in zip(dist, mf.predict(dist, None, 0, 2025))}
    active = mf.kickers[(mf.kickers["last_season"] == 2025) & (mf.kickers["fg_attempts"] >= 40)].copy()
    for d_ in (50, 60, 65, 70):
        active[f"p{d_}"] = mf.predict(np.full(len(active), d_), active.index.to_numpy(), 0, 2025)
    active = active.sort_values("p60", ascending=False)
    report["active_kickers_indoors_2025"] = active[["name", "fg_attempts", "u", "v", "p50", "p60", "p65", "p70"]].round(3).reset_index().to_dict(orient="records")
    (REPORTS / "kicker_metrics.json").write_text(json.dumps(report, indent=1))

    print("\nbest priors:", best)
    for name, r in report["holdout_2025"].items():
        print(f"{name:13s} logloss {r['logloss']:.4f} brier {r['brier']:.4f} made-given-0%: {r['made_kicks_given_0pct']}")
        print("   ", {b: (v["n"], round(v["made"], 3), round(v["pred"], 3)) for b, v in r["bands"].items()})
    print("league curve outdoors:", {k: round(v, 3) for k, v in report["league_curve_outdoors_2025"].items()})
    print(active[["name", "fg_attempts", "p50", "p60", "p65", "p70"]].round(3).head(6).to_string())
    print(active[["name", "fg_attempts", "p50", "p60", "p65", "p70"]].round(3).tail(4).to_string())


if __name__ == "__main__":
    main()

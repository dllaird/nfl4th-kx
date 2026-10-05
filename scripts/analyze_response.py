"""Diagnostics for the Phase 5 response models (out-of-fold predictions).

Prints: uncertainty coverage, error by support decile, price calibration table, edge-case
segments and pre-price monotonicity. Usage: python scripts/analyze_response.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nfl4th_py.market.response import expit

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    o = pd.read_parquet(ROOT / "reports" / "response_oof.parquet")
    r = o["y_logit"] - o["pred_transition"]
    z = r / o["sd_transition"]
    print("uncertainty coverage (|resid| <= k*sd): ",
          {k: round(float((z.abs() <= k).mean()), 3) for k in (1.0, 1.645, 1.96)},
          " nominal {1.0: 0.683, 1.645: 0.9, 1.96: 0.95}")
    inr = o["pre_price"].between(0.03, 0.97)
    print("coverage in range:", {k: round(float((z[inr].abs() <= k).mean()), 3) for k in (1.0, 1.645, 1.96)})

    o["support_decile"] = pd.qcut(o["support_transition"].rank(method="first"), 10, labels=False)
    t = o.assign(ae=r.abs()).groupby("support_decile").agg(
        support_min=("support_transition", "min"), mae_logit=("ae", "mean"), n=("ae", "size"))
    print("\nerror by support decile\n", t.round(3).to_string())
    bands = pd.cut(o["support_transition"], [-0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 1.0])
    print("\nerror by support band (percentile)\n",
          o.assign(ae=r.abs()).groupby(bands, observed=True)["ae"].agg(["mean", "size"]).round(3).to_string())

    p = expit(o["pred_transition"])
    bins = pd.cut(p, [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1])
    cal = pd.DataFrame({"pred": p, "actual": o["post_price"]}).groupby(bins, observed=True).agg(
        pred=("pred", "mean"), actual=("actual", "mean"), n=("pred", "size"))
    print("\nprice calibration (transition, out-of-fold)\n", cal.round(3).to_string())

    segs = {
        "OT": o["qtr"] == 5,
        "last 2 min, within 8": (o["game_seconds_remaining"] <= 120) & (o["score_differential"].abs() <= 8),
        "pre price < 0.03 or > 0.97": ~inr,
        "coarse timestamp": o["flag_coarse_time"],
        "replay": o["flag_replay"],
    }
    print("\nedge segments: MAE logit (transition / wp_baseline / persistence)")
    for k, m in segs.items():
        sub = o[m]
        f = lambda c: round(float((sub["y_logit"] - sub[c]).abs().mean()), 3) if sub[c].notna().any() else None
        print(f"  {k:28s} n={len(sub):5d}  {f('pred_transition')} / {f('pred_wp_baseline')} / {f('pred_persistence')}")


if __name__ == "__main__":
    main()

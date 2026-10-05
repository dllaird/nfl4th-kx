"""Profile the plays the engine refuses (out-of-sample backtest setup).

Usage: python scripts/refusal_analysis.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as B
from backtest_inputs import engine_inputs
from nfl4th_py import decision, prep
from nfl4th_py.market.response import TransitionModel


def main() -> None:
    bt = pd.read_parquet(B.REPORTS / "backtest_plays.parquet")
    refused = bt[bt["engine_rec"].isna()]
    d = pd.read_parquet(B.CACHE / "dataset" / "plays_2025.parquet")
    f = d.merge(refused[["game_id", "play_id"]], on=["game_id", "play_id"])
    f.index = pd.RangeIndex(len(f))
    games = prep.build_games(pd.read_parquet(B.CACHE / "schedule_2025.parquet"))
    folds = pd.read_parquet(B.CACHE / "response_folds" / "folds.parquet").set_index("game_id")["fold"]
    plays, snap = engine_inputs(f)
    rows, outcomes = [], []
    for k in range(5):
        m = (f["game_id"].map(folds).fillna(0).astype(int) == k).to_numpy()
        if not m.any():
            continue
        model = TransitionModel.load(B.CACHE / "response_folds" / str(k))
        res, oc = decision.add_4th_probs(plays[m].reset_index(drop=True), snap[m].reset_index(drop=True), games,
                                         model=model, fg_model=B.FG_MODEL, return_outcomes=True)
        idx = np.flatnonzero(m)
        res.index = idx
        rows.append(res)
        for o, v in oc.items():
            outcomes.append(v.assign(option=o, play=idx[v["play"].to_numpy()]))
    res = pd.concat(rows).sort_index()
    oc = pd.concat(outcomes)
    f = f.join(res[["no_recommendation_reason", "go_support", "fg_support", "punt_support",
                    "go_min_support", "fg_min_support", "punt_min_support", "fg_offered"]])
    pd.set_option("display.width", 220)
    print(f"refused plays: {len(f)}")
    n_opts = res[["go_ev", "fg_ev", "punt_ev"]].notna().sum(axis=1)
    print("options available:", n_opts.value_counts().sort_index().to_dict())
    r = f["no_recommendation_reason"].fillna("")
    f["blocked"] = r.str.findall(r"(go|fg|punt): outcome").map(lambda l: "+".join(sorted(l)) or "market")
    print(f["blocked"].value_counts().to_string())
    f["price_band"] = pd.cut(f["pre_price"], [0, .02, .1, .5, .9, .98, 1], labels=["<2%", "2-10%", "10-50%", "50-90%", "90-98%", ">98%"])
    f["margin"] = pd.cut(f["score_differential"].abs(), [-1, 3, 8, 16, 99], labels=["0-3", "4-8", "9-16", "17+"])
    f["time"] = np.select([f["qtr"] <= 2, (f["qtr"] == 4) & (f["quarter_seconds_remaining"] <= 120)], ["1st half", "Q4 last 2 min"], "2nd half")
    print(pd.crosstab(f["price_band"], f["margin"]))
    print(f["time"].value_counts().to_dict())
    # which outcomes are unsupported (prob >= 5% and support < 0.002, or the option's weighted support < 0.01)
    oc = oc.merge(f[["blocked"]], left_on="play", right_index=True)
    weak = oc[(oc["prob"] >= 0.05) & (oc["support"] < 0.002)]
    print("\nlow-support outcomes (prob >= 5%):")
    print(weak.groupby(["option", "outcome", "post_kind"]).size().sort_values(ascending=False).head(12).to_string())
    print("\nexamples:")
    cols = ["game_id", "qtr", "quarter_seconds_remaining", "ydstogo", "yardline_100", "score_differential", "pre_price", "blocked"]
    print(f[cols].sample(12, random_state=1).to_string(index=False))
    f.to_parquet(B.CACHE / "refusals.parquet")


if __name__ == "__main__":
    main()

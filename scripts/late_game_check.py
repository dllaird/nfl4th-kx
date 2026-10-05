"""Inspect engine recommendations in the final 2 minutes of each half (out-of-sample).

Prints coverage, agreement with nfl4th and coaches, and the plays where the engine and
nfl4th disagree. Usage: python scripts/late_game_check.py [--q2]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import backtest as B
from backtest_inputs import engine_inputs
from nfl4th_py import decision, prep
from nfl4th_py.market.response import TransitionModel
from nfl4th_py.models import wp_baseline


def main() -> None:
    q2 = "--q2" in sys.argv
    d = pd.read_parquet(B.CACHE / "dataset" / "plays_2025.parquet")
    f = d[(d.down == 4) & d.play_type.isin(["run", "pass", "punt", "field_goal"]) & (d.qtr == (2 if q2 else 4))
          & (d.quarter_seconds_remaining <= 120)].copy()
    f.index = pd.RangeIndex(len(f))
    f["actual"] = B.actual_choice(f)
    games = prep.build_games(pd.read_parquet(B.CACHE / "schedule_2025.parquet"))
    folds = pd.read_parquet(B.CACHE / "response_folds" / "folds.parquet").set_index("game_id")["fold"]
    models = {k: TransitionModel.load(B.CACHE / "response_folds" / str(k)) for k in range(5)}
    eng = B.run_engine(f, games, folds, models)
    plays, _ = engine_inputs(f)
    orig = wp_baseline.add_4th_probs_baseline(plays.drop(columns=["kicker_player_id", "wind", "temp"]), games)
    f["engine"] = eng["recommendation"]
    f["nfl4th"] = B.rec_of(orig)
    for o in ("go", "fg", "punt"):
        f[f"ev_{o}"] = (100 * eng[f"{o}_ev"]).round(1)
    f["gap"] = (100 * eng["ev_gap"]).round(1)
    f["reason"] = eng["no_recommendation_reason"]
    rec = f["engine"].notna()
    print(f"{'Q2' if q2 else 'Q4'} final 2 min 4th downs: {len(f)}; engine recommends {rec.sum()}")
    print(f["reason"].dropna().str.split("; ").explode().value_counts().to_string())
    r = f[rec]
    print(f"agreement with nfl4th {np.mean(r.engine == r.nfl4th):.3f}; coach followed engine "
          f"{np.mean(r.actual == r.engine):.3f}, nfl4th {np.mean(r.actual == r.nfl4th):.3f}")
    print(pd.crosstab(r.engine, r.nfl4th))
    pd.set_option("display.width", 250)
    cols = ["game_id", "quarter_seconds_remaining", "ydstogo", "yardline_100", "score_differential",
            "posteam_timeouts_remaining", "pre_price", "ev_go", "ev_fg", "ev_punt", "engine", "gap", "nfl4th", "actual"]
    print(r[r.engine != r.nfl4th][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Sanity properties of the trained market-response models (nfl4th_py/data/response).

All else equal, for the pre-play possession team: better field position, a bigger lead
and having the ball must never lower the predicted post-play price.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl4th_py.market import response as R

MODEL_DIR = Path(R.__file__).resolve().parents[1] / "data" / "response"
pytestmark = pytest.mark.skipif(not (MODEL_DIR / "transition.json").exists(), reason="no trained model")


@pytest.fixture(scope="module")
def models():
    tm = R.TransitionModel.load(MODEL_DIR)
    vs = R.ValueSurface()
    import xgboost as xgb
    import json

    vs.booster = xgb.Booster()
    vs.booster.load_model(MODEL_DIR / "value_surface.json")
    vs.booster.set_attr(best_iteration=str(json.loads((MODEL_DIR / "value_surface_meta.json").read_text())["best_iteration"]))
    return tm, vs


@pytest.fixture(scope="module")
def states():
    rng = np.random.default_rng(0)
    n = 2000
    gsr = rng.integers(30, 3600, n).astype(float)
    qtr = np.where(gsr > 2700, 1, np.where(gsr > 1800, 2, np.where(gsr > 900, 3, 4)))
    hsr = np.where(gsr > 1800, gsr - 1800, gsr)
    t = pd.DataFrame(
        {
            "pre_logit": rng.normal(0, 1.2, n), "qtr": qtr, "game_seconds_remaining": gsr,
            "half_seconds_remaining": hsr, "down": 4.0, "ydstogo": rng.integers(1, 11, n).astype(float),
            "yardline_100": rng.integers(5, 95, n).astype(float),
            "score_differential": rng.integers(-17, 18, n).astype(float),
            "posteam_timeouts_remaining": rng.integers(0, 4, n).astype(float),
            "defteam_timeouts_remaining": rng.integers(0, 4, n).astype(float),
            "receive_2h_ko": 0, "kind": "scrimmage",
            "kalshi_posteam_spread": rng.normal(0, 6, n), "kalshi_total_line": rng.normal(45, 4, n),
            "post_has_ball": rng.integers(0, 2, n), "post_kind": "scrimmage",
            "post_fp": rng.integers(5, 95, n).astype(float),
            "post_down": 1.0, "post_ydstogo": 10.0,
        }
    )
    t["post_score_diff"] = t["score_differential"]
    t["post_qtr"] = t["qtr"]
    t["post_game_seconds_remaining"] = np.maximum(t["game_seconds_remaining"] - 6, 0)
    t["post_half_seconds_remaining"] = np.maximum(t["half_seconds_remaining"] - 6, 0)
    t["post_team_timeouts"] = t["posteam_timeouts_remaining"]
    t["post_opp_timeouts"] = t["defteam_timeouts_remaining"]
    return t


def _check(model, base, better, name):
    a, b = model(base), model(better)
    worse = b < a - 1e-6
    assert not worse.any(), f"{name}: {worse.sum()} of {len(a)} states got worse"


@pytest.mark.parametrize("which", ["transition", "value"])
def test_field_position(models, states, which):
    m = models[0].predict_logit if which == "transition" else models[1].predict_logit
    better = states.assign(post_fp=np.maximum(states["post_fp"] - 10, 1))
    _check(m, states, better, "field position")


@pytest.mark.parametrize("which", ["transition", "value"])
def test_bigger_lead(models, states, which):
    m = models[0].predict_logit if which == "transition" else models[1].predict_logit
    _check(m, states, states.assign(post_score_diff=states["post_score_diff"] + 3), "lead")


def test_possession(models, states):
    m = models[0].predict_logit  # production model: monotone in possession by construction
    _check(m, states.assign(post_has_ball=0), states.assign(post_has_ball=1), "possession")


def test_possession_value_surface_mostly_holds(models, states):
    # the value surface values the opponent's side by a sign flip, so possession can't be
    # constrained; it is not used in production. Allow at most 1% of states to violate.
    m = models[1].predict_logit
    worse = m(states.assign(post_has_ball=1)) < m(states.assign(post_has_ball=0)) - 1e-6
    assert worse.mean() <= 0.01


def test_prediction_outputs(models, states):
    p = models[0].predict(states)
    assert p["price"].between(0, 1).all()
    assert (p["sd_logit"] > 0).all()
    assert p["support"].between(0, 1).all()

"""Decision engine: valid recommendations, refusals, and no model-WP fallback."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl4th_py import decision
from tests.conftest import FIXTURES

pytestmark = pytest.mark.skipif(
    not (decision.MODEL_DIR / "transition.json").exists(), reason="no trained response model"
)


def _play(**kw):
    base = dict(
        home_team="KC", away_team="LAC", posteam="KC", type="reg", season=2025, qtr=4,
        quarter_seconds_remaining=600, ydstogo=2, yardline_100=40, score_differential=-3,
        home_opening_kickoff=1, posteam_timeouts_remaining=3, defteam_timeouts_remaining=3,
    )
    base.update(kw)
    return base


def _snap(**kw):
    base = dict(
        price=0.40, bid_ask=0.01, quote_age_s=2.0, n_trades=10, halted=False,
        kalshi_spread_line=3.0, kalshi_total_line=45.5, kalshi_spread_quality=0.8,
        kalshi_total_quality=0.8,
    )
    base.update(kw)
    return base


@pytest.fixture(scope="module")
def games():
    g = pd.read_parquet(FIXTURES / "games.parquet")
    assert ((g.home_team == "KC") & (g.away_team == "LAC") & (g.season == 2025)).any()
    return g


def run(plays, snaps, games, **cfg):
    return decision.add_4th_probs(
        pd.DataFrame(plays), pd.DataFrame(snaps), games, config=decision.DecisionConfig(**cfg)
    )


def test_recommends_with_ev_and_uncertainty(games):
    out = run([_play()], [_snap()], games)
    r = out.iloc[0]
    assert r["recommendation"] in decision.OPTIONS
    for o in decision.OPTIONS:
        assert 0 <= r[f"{o}_ev"] <= 1 and r[f"{o}_sd"] > 0
    best = max(decision.OPTIONS, key=lambda o: r[f"{o}_ev"])
    assert r["recommendation"] == best
    assert r["ev_gap"] >= 0 and 0.5 <= r["prob_best_option_is_best"] <= 1
    assert 0 < r["first_down_prob"] < 1 and r["wp_succeed"] > r["wp_fail"]
    assert r["make_fg_wp"] > r["miss_fg_wp"]


@pytest.mark.parametrize(
    "snap_kw, reason",
    [
        ({"price": 0.99}, "outside"),
        ({"price": np.nan}, "no winner-market price"),
        ({"quote_age_s": 300}, "stale"),
        ({"bid_ask": 0.2}, "wider"),
        ({"halted": True}, "halted"),
        ({"kalshi_spread_line": np.nan}, "spread ladder"),
        ({"kalshi_total_quality": 0.0}, "total ladder"),
    ],
)
def test_refuses_unusable_markets(games, snap_kw, reason):
    out = run([_play()], [_snap(**snap_kw)], games)
    assert out.loc[0, "no_recommendation"]
    assert reason in out.loc[0, "no_recommendation_reason"]
    assert pd.isna(out.loc[0, "go_ev"])  # nothing is valued: no fallback


def test_support_threshold_refuses(games):
    out = run([_play()], [_snap()], games, min_support=1.01)
    assert out.loc[0, "no_recommendation"]
    assert "support" in out.loc[0, "no_recommendation_reason"]


def test_no_punt_inside_31(games):
    out = run([_play(yardline_100=20)], [_snap()], games)
    assert pd.isna(out.loc[0, "punt_ev"])


def test_ot_refused(games):
    out = run([_play(qtr=5)], [_snap()], games)
    assert "overtime" in out.loc[0, "no_recommendation_reason"]


def test_runoff_is_applied(games):
    a = run([_play(score_differential=3, quarter_seconds_remaining=200)], [_snap(price=0.8)], games)
    b = run([_play(score_differential=3, quarter_seconds_remaining=200, runoff=40)], [_snap(price=0.8)], games)
    ga, gb = (x.attrs["outcomes"]["go"] for x in (a, b))
    conv_a = ga[ga.outcome == "convert"]["post_game_seconds_remaining"].iloc[0]
    conv_b = gb[gb.outcome == "convert"]["post_game_seconds_remaining"].iloc[0]
    assert conv_a - conv_b == pytest.approx(40)


def test_decision_log(tmp_path, games):
    out = run([_play()], [_snap()], games)
    path = decision.log_decisions(out, [{"ticker": "X", "bid": 0.39}], tmp_path)
    back = pd.read_parquet(path)
    assert len(back) == 1 and "raw_quotes_json" in back and back.loc[0, "recommendation"] == out.loc[0, "recommendation"]

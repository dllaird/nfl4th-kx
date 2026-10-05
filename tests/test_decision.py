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


def run(plays, snaps, games, return_outcomes=False, **cfg):
    return decision.add_4th_probs(
        pd.DataFrame(plays), pd.DataFrame(snaps), games, config=decision.DecisionConfig(**cfg),
        return_outcomes=return_outcomes,
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
        ({"price": np.nan}, "no winner-market price"),
        ({"quote_age_s": 300}, "stale"),
        ({"bid_ask": 0.2}, "wider"),
        ({"kalshi_spread_line": np.nan}, "no Kalshi pregame spread"),
        ({"kalshi_total_line": np.nan}, "no Kalshi pregame total"),
    ],
)
def test_refuses_unusable_markets(games, snap_kw, reason):
    out = run([_play()], [_snap(**snap_kw)], games)
    assert out.loc[0, "no_recommendation"]
    assert reason in out.loc[0, "no_recommendation_reason"]
    assert pd.isna(out.loc[0, "go_ev"])  # nothing is valued: no fallback


@pytest.mark.parametrize(
    "play_kw, snap_kw, caution",
    [
        ({"score_differential": 21}, {"price": 0.99}, "tick limit"),
        ({"score_differential": -21}, {"price": 0.015}, "tick limit"),
        ({}, {"quote_age_s": 60}, "s old"),
        ({}, {"bid_ask": 0.06}, "wider than 0.04"),
        ({}, {"halted": True}, "no recent trade"),
        ({}, {"kalshi_total_quality": 0.05}, "fit is weak"),
        ({}, {"kalshi_spread_source": "winner_price"}, "fallback"),
    ],
)
def test_marginal_markets_recommend_with_caution(games, play_kw, snap_kw, caution):
    out = run([_play(**play_kw)], [_snap(**snap_kw)], games)
    assert out.loc[0, "recommendation"] in decision.OPTIONS
    assert out.loc[0, "confidence"] == "caution"
    assert caution in out.loc[0, "cautions"]


def test_clean_market_has_normal_confidence(games):
    out = run([_play()], [_snap()], games)
    assert out.loc[0, "confidence"] == "normal" and out.loc[0, "cautions"] is None


def test_support_threshold_refuses(games):
    out = run([_play()], [_snap()], games, min_support=1.01)
    assert out.loc[0, "no_recommendation"]
    assert "support" in out.loc[0, "no_recommendation_reason"]


def test_fg_uses_kicker_and_drops_hopeless_attempts(games):
    from nfl4th_py.models.kicker import KickerFGModel

    km = KickerFGModel.load()
    k = km.kickers[(km.kickers["fg_attempts"] >= 100) & (km.kickers["last_season"] == 2025)]
    strong = k.index[np.argmax(km.predict(np.full(len(k), 60.0), k.index.to_numpy(), 0, 2025))]
    weak = k.index[np.argmin(km.predict(np.full(len(k), 60.0), k.index.to_numpy(), 0, 2025))]
    out = run([_play(yardline_100=43, kicker_player_id=strong), _play(yardline_100=43, kicker_player_id=weak),
               _play(yardline_100=85)], [_snap()] * 3, games)
    assert out.loc[0, "fg_make_prob"] > out.loc[1, "fg_make_prob"]
    assert out.loc[0, "fg_offered"] and out.loc[1, "fg_offered"]
    # a 102-yard attempt is not offered: no FG value and no support veto from it
    assert not out.loc[2, "fg_offered"] and pd.isna(out.loc[2, "fg_ev"])
    assert 0 < out.loc[2, "fg_make_prob"] < 0.01
    assert "fg:" not in (out.loc[2, "no_recommendation_reason"] or "")


def test_no_punt_inside_31(games):
    out = run([_play(yardline_100=20)], [_snap()], games)
    assert pd.isna(out.loc[0, "punt_ev"])


def test_end_of_regulation_is_recommended(games):
    out = run([_play(quarter_seconds_remaining=60)], [_snap()], games)
    assert out.loc[0, "recommendation"] in decision.OPTIONS
    off = run([_play(quarter_seconds_remaining=60)], [_snap()], games, q4_refuse_last_seconds=120)
    assert "regulation" in off.loc[0, "no_recommendation_reason"]


def test_trailing_team_keeps_time_after_late_conversion(games):
    # down 4 with 13 s left: a conversion must not run the clock out on the trailing team
    _, a = run([_play(quarter_seconds_remaining=13, score_differential=-4, posteam_timeouts_remaining=0)], [_snap(price=0.1)], games, True)
    conv = a["go"][a["go"].outcome == "convert"]
    assert (conv["post_game_seconds_remaining"] >= 3).all()


def test_kneel_out_states_use_market_level(games):
    # a FG that leaves the kicking team down 1 with ~10 s and no timeouts: the receiver kneels
    _, a = run([_play(quarter_seconds_remaining=14, score_differential=-4, posteam_timeouts_remaining=0, yardline_100=20)],
               [_snap(price=0.1)], games, True)
    make = a["fg"][a["fg"].outcome == "make"].iloc[0]
    assert make["kneel_out"] and make["price"] <= 0.01 + 1e-9


def test_overtime_supported_with_caution(games):
    out = run([_play(qtr=5, quarter_seconds_remaining=400, score_differential=0)], [_snap(price=0.5)], games)
    assert out.loc[0, "recommendation"] in decision.OPTIONS
    assert "overtime" in out.loc[0, "cautions"]
    off = run([_play(qtr=5)], [_snap()], games, max_qtr=4)
    assert "overtime" in off.loc[0, "no_recommendation_reason"]


def _convert_clock(out):
    go = out["go"]
    return go[go.outcome == "convert"]["post_game_seconds_remaining"].iloc[0]


def test_conversion_clock_and_runoff(games):
    # Q4, 10:00 left: a conversion uses the measured 38 s; runoff 40 -> 6 + 40 = 46 s
    _, a = run([_play(quarter_seconds_remaining=600)], [_snap()], games, True)
    _, b = run([_play(quarter_seconds_remaining=600, runoff=40)], [_snap()], games, True)
    assert _convert_clock(a) == pytest.approx(600 - 38)
    assert _convert_clock(b) == pytest.approx(600 - 46)
    fail = a["go"][a["go"].outcome == "fail"]["post_game_seconds_remaining"].iloc[0]
    assert fail == pytest.approx(600 - 5)


def test_two_minute_warning_stops_clock(games):
    _, a = run([_play(qtr=2, quarter_seconds_remaining=140)], [_snap()], games, True)
    assert _convert_clock(a) == pytest.approx(1800 + 120)


def test_decision_log(tmp_path, games):
    out = run([_play()], [_snap()], games)
    path = decision.log_decisions(out, [{"ticker": "X", "bid": 0.39}], tmp_path)
    back = pd.read_parquet(path)
    assert len(back) == 1 and "raw_quotes_json" in back and back.loc[0, "recommendation"] == out.loc[0, "recommendation"]

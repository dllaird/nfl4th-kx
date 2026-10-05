"""Kicker-specific FG model: shape, pooling, persistence, and engine use."""

import numpy as np
import pandas as pd
import pytest

from nfl4th_py.models import kicker

pytestmark = pytest.mark.skipif(not kicker.MODEL_PATH.exists(), reason="no fitted kicker model")


@pytest.fixture(scope="module")
def model():
    return kicker.KickerFGModel.load()


def test_league_curve_declines_smoothly_with_no_cutoff(model):
    d = np.arange(18, 111)
    for outdoors in (0, 1):
        p = model.predict(d, None, outdoors, 2025)
        assert np.all(np.diff(p) < 0), "make probability must fall with distance"
        assert p[d == 70][0] > 0.05 and 0 < p[-1] < 0.01  # long kicks are unlikely, never exactly 0


def test_every_kicker_curve_declines(model):
    d = np.arange(18, 111, dtype=float)
    for kid in model.kickers.index:
        p = model.predict(d, np.full(len(d), kid, dtype=object), 0, 2025)
        assert np.all(np.diff(p) < 0), kid


def test_unknown_kicker_gets_league_curve(model):
    a = model.predict([50.0], ["not-a-kicker"], 0, 2025)
    b = model.predict([50.0], None, 0, 2025)
    assert a == pytest.approx(b)


def test_kickers_differ(model):
    k = model.kickers[(model.kickers["fg_attempts"] >= 100) & (model.kickers["last_season"] == 2025)]
    p60 = model.predict(np.full(len(k), 60.0), k.index.to_numpy(), 0, 2025)
    assert p60.max() - p60.min() > 0.15


def test_save_load_roundtrip(tmp_path, model):
    path = tmp_path / "k.json"
    model.save(path)
    back = kicker.KickerFGModel.load(path)
    kid = model.kickers.index[0]
    assert back.predict([45.0], [kid], 1, 2025, 10.0, 30.0) == pytest.approx(model.predict([45.0], [kid], 1, 2025, 10.0, 30.0))


def test_kicker_for_falls_back_to_previous_game():
    kicks = pd.DataFrame({
        "game_id": ["g1", "g1", "g2"], "season": [2025] * 3, "week": [1, 1, 2],
        "posteam": ["KC", "KC", "KC"], "kicker_player_id": ["A", "A", "B"],
    })
    lk = kicker.team_kickers(kicks)
    assert kicker.kicker_for(lk, "g2", "KC", 2025, 2) == "B"
    assert kicker.kicker_for(lk, "g3", "KC", 2025, 3) == "B"  # no kick yet: last kicker
    assert kicker.kicker_for(lk, "g0", "KC", 2025, 1) is None

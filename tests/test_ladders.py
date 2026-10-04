import numpy as np
import pandas as pd
from scipy import stats

from nfl4th_py.market import ladders


def _synthetic(mu=3.0, sd=13.0, strikes=np.arange(-20.5, 21, 1.0), noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    p = stats.norm.sf(strikes, mu, sd) + rng.normal(0, noise, len(strikes))
    p = np.clip(p, 0.01, 0.99)
    return pd.DataFrame(
        {"x": strikes, "side": 1, "bid": p - 0.01, "ask": p + 0.01, "age_s": 30.0, "volume": 1000.0}
    )


def test_isotonic_enforces_monotone():
    y = np.array([0.9, 0.7, 0.75, 0.5, 0.55, 0.2])
    out = ladders.isotonic_decreasing(y, np.ones_like(y))
    assert np.all(np.diff(out) <= 1e-12)
    assert out[1] == out[2] == 0.725


def test_normal_ladder_recovers_parameters():
    fit = ladders.fit_ladder(_synthetic(), "margin")
    assert abs(fit.mean - 3.0) < 0.2 and abs(fit.sd - 13.0) < 0.5
    assert fit.quality > 0.5


def test_noisy_ladder_still_close():
    fit = ladders.fit_ladder(_synthetic(noise=0.02, seed=3), "margin")
    assert abs(fit.median - 3.0) < 1.5


def test_wide_and_stale_strikes_are_dropped():
    q = _synthetic()
    q.loc[0:5, "ask"] = q.loc[0:5, "bid"] + 0.3  # wide
    q.loc[6:8, "age_s"] = 7200  # stale
    pts = ladders.strike_points(q)
    assert not pts.loc[pts["x"].isin(q.loc[0:8, "x"]), "used"].any()


def test_too_few_strikes_gives_no_fit():
    fit = ladders.fit_ladder(_synthetic(strikes=np.array([2.5])), "margin")
    assert fit.mean is None and fit.quality == 0


def test_away_ladder_maps_to_home_margin():
    m = pd.DataFrame({"team": ["HOME", "AWAY"], "floor_strike": [3.5, 6.5]})
    q = ladders.margin_quotes(m, "HOME")
    assert list(q["x"]) == [3.5, -6.5] and list(q["side"]) == [1, -1]

import numpy as np
import pandas as pd
import pytest

from nfl4th_py import dataset
from nfl4th_py.market import alignment


def _trades():
    # home market trades at 0.60, away market trades at 0.38 (-> home 0.62)
    return pd.DataFrame(
        {
            "ticker": ["H", "A", "H", "H"],
            "created_time": [
                "2025-09-07T17:00:00.000Z",
                "2025-09-07T17:00:02.000Z",
                "2025-09-07T17:00:04.000Z",
                "2025-09-07T17:01:00.000Z",
            ],
            "yes_price": [0.60, 0.38, 0.60, 0.70],
            "count": [10.0, 30.0, 10.0, 5.0],
            "taker_side": ["yes", "no", "yes", "yes"],
        }
    )


def test_home_series_inverts_away_market():
    s = alignment.home_price_series(_trades(), "H", "A")
    assert list(s["p_home"].round(2)) == [0.60, 0.62, 0.60, 0.70]


def test_price_at_vwap_window_and_age():
    s = alignment.home_price_series(_trades(), "H", "A")
    t0 = s["t"].iat[0]
    w = alignment.price_at(s, t0 + 5, window_s=10)
    assert w.n_trades == 3
    assert w.price == pytest.approx((0.6 * 10 + 0.62 * 30 + 0.6 * 10) / 50)
    assert w.age_s == pytest.approx(1.0)


def test_price_at_respects_floor_and_empty_window():
    s = alignment.home_price_series(_trades(), "H", "A")
    t0 = s["t"].iat[0]
    assert alignment.price_at(s, t0 + 5, 10, t_start_min=t0 + 3).n_trades == 1
    assert alignment.price_at(s, t0 + 30, 5).price is None


def test_finalize_converts_to_posteam_view():
    df = pd.DataFrame(
        {
            "is_home_pos": [True, False],
            "pre_home_vwap": [0.7, 0.7], "pre_home_last": [0.7, 0.7], "pre_last_age_s": [1.0, 1.0],
            "post_home_vwap": [0.8, np.nan], "post_home_last": [0.8, 0.75], "post_last_age_s": [1.0, 2.0],
            "next_snap_ts": [100.0, 100.0], "end_ts": [50.0, 90.0], "snap_ts": [45.0, 85.0],
            "flag_nonmonotonic_time": [False, False],
            "posteam": ["H", "A"], "post_posteam": ["H", "H"], "post_score_differential": [3, 7],
        }
    )
    d = dataset.finalize(df)
    assert d["pre_price"].tolist() == pytest.approx([0.7, 0.3])
    assert d["post_price"].tolist() == pytest.approx([0.8, 0.25])
    assert d["post_price_source"].tolist() == ["trades_vwap", "last_trade"]
    assert d["post_score_differential_pre_view"].tolist() == [3, -7]
    # second play: next snap only 10 s after the play ended -> not settled
    assert d["usable"].tolist() == [True, False]

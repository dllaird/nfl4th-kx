"""Python port vs. the R package on tests/fixtures (built by scripts/make_fixtures.R)."""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import FIXTURES, NON_INPUT_COLS

TOL = 1e-3
COMPONENTS = ["first_down_prob", "fg_make_prob"]
WP_COLS = ["wp_fail", "wp_succeed", "go_wp", "miss_fg_wp", "make_fg_wp", "fg_wp", "punt_wp"]


def _supported(fixtures_4th):
    return fixtures_4th["r_supported"].to_numpy()


def test_fixture_size_and_strata(fixtures_4th):
    assert len(fixtures_4th) >= 2000
    for s in ["end_q2", "end_q4", "goal_to_go", "backed_up", "long_fg", "ot", "runoff"]:
        assert (fixtures_4th["stratum"] == s).sum() > 0


@pytest.mark.parametrize("col", COMPONENTS + WP_COLS + ["go_boost"])
def test_column_parity(col, fixtures_4th, r_outputs, py_outputs):
    sup = _supported(fixtures_4th)
    py = py_outputs[col].to_numpy()[sup]
    r = r_outputs[col].to_numpy()[sup]
    np.testing.assert_array_equal(np.isnan(py), np.isnan(r))
    # go_boost is in percentage points, so its tolerance scales by 100
    tol = TOL * 100 if col == "go_boost" else TOL
    assert np.nanmax(np.abs(py - r)) < tol


def _choice(df: pd.DataFrame) -> np.ndarray:
    vals = np.column_stack([df["go_wp"], df["fg_wp"], df["punt_wp"].fillna(-np.inf)])
    return vals.argmax(axis=1)


def _margin(df: pd.DataFrame) -> np.ndarray:
    vals = np.sort(np.column_stack([df["go_wp"], df["fg_wp"], df["punt_wp"].fillna(-np.inf)]), axis=1)
    return vals[:, -1] - vals[:, -2]


def test_recommendation_agreement(fixtures_4th, r_outputs, py_outputs):
    sup = _supported(fixtures_4th)
    py, r = py_outputs.loc[sup], r_outputs.loc[sup]
    agree = _choice(py) == _choice(r)
    assert agree.mean() >= 0.995
    # every disagreement must be a near-tie in R (top two options within 0.1 WP points)
    assert np.all(_margin(r)[~agree] < TOL)


def test_unsupported_ot_rows_still_compute(fixtures_4th, py_outputs):
    ot = ~_supported(fixtures_4th)
    assert ot.sum() > 0
    assert py_outputs.loc[ot, "first_down_prob"].notna().all()


def test_2pt_parity(fixtures_4th, games):
    from nfl4th_py.models.wp_baseline import add_2pt_probs_baseline

    r = pd.read_parquet(FIXTURES / "r_outputs_2pt.parquet")
    inp = fixtures_4th.set_index("fixture_id").loc[r["fixture_id"]].reset_index()
    inp = inp.drop(columns=NON_INPUT_COLS + ["ydstogo", "yardline_100"])
    py = add_2pt_probs_baseline(inp, games)
    for col in ["wp_0", "wp_1", "wp_2", "conv_1pt", "conv_2pt", "wp_go1", "wp_go2"]:
        assert np.max(np.abs(py[col].to_numpy() - r[col].to_numpy())) < TOL, col


def test_wp_state_parity():
    from nfl4th_py.models.wp_baseline import calculate_win_probability

    ws = pd.read_parquet(FIXTURES / "wp_states.parquet")
    # prepare_df's stale home-perspective spread_time, as R passes it through
    ws["spread_time"] = ws["spread_line"] * np.exp(-4 * ws["elapsed_share"])
    assert np.max(np.abs(calculate_win_probability(ws) - ws["vegas_wp"].to_numpy())) < TOL

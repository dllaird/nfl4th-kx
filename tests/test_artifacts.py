import numpy as np

from nfl4th_py import artifacts


def test_feature_order_matches_boosters():
    for name, spec in artifacts.manifest()["models"].items():
        b = artifacts.booster(name)
        assert b.num_features() == spec["num_feature"] == len(spec["features"])
        assert b.feature_names == spec["features"]


def test_feature_order_pinned():
    # positional orders copied from the R select() calls; changing them silently
    # would corrupt every prediction
    assert artifacts.model_features("fd_model") == [
        "down", "ydstogo", "yardline_100", "era3", "era4", "outdoors", "retractable",
        "dome", "posteam_spread", "total_line", "posteam_total",
    ]
    assert artifacts.model_features("home_wp_model")[-1] == "home_timeouts_remaining"
    assert len(artifacts.model_features("home_wp_model")) == 11


def test_fg_lookup_covers_grid():
    lut = artifacts.fg_lookup()
    assert len(lut) == 99 * 4
    assert set(lut["fg_model_roof"]) == {"00", "01", "10", "11"}
    assert lut["fg_make_prob"].between(0, 1).all()


def test_punt_table_sums_to_one():
    pt = artifacts.punt_table()
    sums = pt.groupby("yardline_100")["pct"].sum()
    assert sums.index.min() == 31 and sums.index.max() == 99
    np.testing.assert_allclose(sums, 1.0, atol=1e-9)

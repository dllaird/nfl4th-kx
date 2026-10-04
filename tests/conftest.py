from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
NON_INPUT_COLS = ["stratum", "fixture_id", "r_supported", "game_id", "play_id"]


@pytest.fixture(scope="session")
def fixtures_4th() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "fixtures_4th.parquet")


@pytest.fixture(scope="session")
def games() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "games.parquet")


@pytest.fixture(scope="session")
def r_outputs() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "r_outputs.parquet")


@pytest.fixture(scope="session")
def py_outputs(fixtures_4th: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    from nfl4th_py.models.wp_baseline import add_4th_probs_baseline

    return add_4th_probs_baseline(fixtures_4th.drop(columns=NON_INPUT_COLS), games)

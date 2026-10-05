"""ESPN fetchers: get_4th_plays() parity with R on saved summaries; live situation parsing."""

import json

import pandas as pd
import pytest

from nfl4th_py.scrape import espn
from tests.conftest import FIXTURES

GAMES = ["2025_20_BUF_DEN", "2025_14_CHI_GB"]
COLS = [
    "qtr", "quarter_seconds_remaining", "posteam", "yardline_100", "ydstogo",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining", "home_opening_kickoff",
    "score_differential", "runoff",
]


@pytest.mark.parametrize("gid", GAMES)
def test_get_4th_plays_matches_r(gid, games):
    r = pd.read_parquet(FIXTURES / f"r_get4th_{gid}.parquet")
    summary = json.loads((FIXTURES / f"espn_summary_{gid}.json").read_text())
    row = games.set_index("game_id").loc[gid].to_dict() | {"game_id": gid}
    py = espn.get_4th_plays(summary, row)
    assert len(py) == len(r)
    m = r.merge(py, on="play_id", suffixes=("_r", "_py"))
    assert len(m) == len(r)
    for c in COLS:
        a, b = m[f"{c}_r"], m[f"{c}_py"]
        assert ((a == b) | (a.isna() & b.isna())).all(), c


def _scoreboard(state="in", down=4):
    return {
        "events": [
            {
                "id": "1",
                "competitions": [
                    {
                        "status": {"period": 4, "displayClock": "2:05", "type": {"state": state}},
                        "competitors": [
                            {"team": {"id": "10", "abbreviation": "NO"}, "homeAway": "home", "score": "17"},
                            {"team": {"id": "20", "abbreviation": "ATL"}, "homeAway": "away", "score": "20"},
                        ],
                        "situation": {
                            "down": down, "distance": 3, "possession": "10", "possessionText": "ATL 35",
                            "homeTimeouts": 1, "awayTimeouts": 2, "downDistanceText": "4th & 3 at ATL 35",
                        },
                    }
                ],
            }
        ]
    }


GAME = {"game_id": "2026_04_ATL_NO", "home_team": "NO", "away_team": "ATL", "type": "reg", "season": 2026}


def test_current_situation():
    s = espn.current_situation(_scoreboard(), "1", GAME, home_opening_kickoff=1)
    assert s["posteam"] == "NO" and s["down"] == 4 and s["ydstogo"] == 3
    assert s["yardline_100"] == 35  # at the opponent's 35
    assert s["quarter_seconds_remaining"] == 125 and s["qtr"] == 4
    assert s["score_differential"] == -3
    assert s["posteam_timeouts_remaining"] == 1 and s["defteam_timeouts_remaining"] == 2


def test_no_situation_when_not_live():
    assert espn.current_situation(_scoreboard(state="post"), "1", GAME, 1) is None


def test_cli_parses():
    from nfl4th_py import cli

    with pytest.raises(SystemExit):
        cli.main(["--help"])

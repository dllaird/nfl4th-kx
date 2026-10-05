"""ESPN live-game fetchers.

`get_4th_plays()` ports nfl4th's R/get_game_data.R: completed 4th-down plays of a game
in add_4th_probs() input format. `current_situation()` reads the live scoreboard's
`situation` block so a decision can be made *before* the snap.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import requests

SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

_ABBR_FIX = {"WSH": "WAS", "LAR": "LA", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "LA", "JAC": "JAX"}
_CHALLENGE_TEAMS = {
    "Arizona": "ARI", "Atlanta": "ATL", "Baltimore": "BAL", "Buffalo": "BUF", "Carolina": "CAR",
    "Chicago": "CHI", "Cincinnati": "CIN", "Cleveland": "CLE", "Dallas": "DAL", "Denver": "DEN",
    "Detroit": "DET", "Green Bay": "GB", "Houston": "HOU", "Indianapolis": "IND",
    "Jacksonville": "JAX", "Kansas City": "KC", "Los Angeles Rams": "LA",
    "Los Angeles Chargers": "LAC", "Las Vegas": "LV", "Miami": "MIA", "Minnesota": "MIN",
    "New England": "NE", "New Orleans": "NO", "New York Giants": "NYG", "New York Jets": "NYJ",
    "Philadelphia": "PHI", "Pittsburgh": "PIT", "Seattle": "SEA", "San Francisco": "SF",
    "Tampa Bay": "TB", "Tennessee": "TEN", "Washington": "WAS",
}


def fix_abbr(x):
    return _ABBR_FIX.get(x, x) if isinstance(x, str) else x


def fetch_summary(espn_id: str | int, session: requests.Session | None = None) -> dict:
    s = session or requests
    r = s.get(SUMMARY_URL, params={"event": espn_id}, timeout=20)
    r.raise_for_status()
    return r.json()


def _drive_plays(summary: dict) -> list[dict]:
    drives = summary.get("drives") or {}
    out = []
    for d in (drives.get("previous") or []) + ([drives["current"]] if drives.get("current") else []):
        abbr = (d.get("team") or {}).get("abbreviation")
        for p in d.get("plays") or []:
            out.append({**p, "_team": abbr})
    return out


def plays_frame(summary: dict) -> pd.DataFrame:
    """All plays (deduplicated by id) with the fields get_4th_plays() uses."""
    rows = []
    seen = set()
    for p in _drive_plays(summary):
        if p.get("id") in seen:
            continue
        seen.add(p.get("id"))
        start = p.get("start") or {}
        rows.append(
            {
                "id": p.get("id"),
                "posteam": fix_abbr(p.get("_team")),
                "qtr": (p.get("period") or {}).get("number"),
                "yardline_100": start.get("yardsToEndzone"),
                "yardline": start.get("possessionText"),
                "down": start.get("down"),
                "ydstogo": start.get("distance"),
                "desc": p.get("text") or "",
                "time": (p.get("clock") or {}).get("displayValue"),
                "type_text": (p.get("type") or {}).get("text"),
                "home_score": p.get("homeScore"),
                "away_score": p.get("awayScore"),
                "wallclock": p.get("wallclock"),
            }
        )
    return pd.DataFrame(rows)


def get_4th_plays(summary: dict, game: dict) -> pd.DataFrame:
    """Port of nfl4th::get_4th_plays(). `game` has game_id, espn, home_team, away_team,
    season, type (nflverse schedule row from prep.build_games)."""
    p = plays_frame(summary)
    if p.empty:
        return p
    p = p[p["qtr"] <= 4].copy()
    t = p["time"].fillna("")
    p["mins"] = pd.to_numeric(t.str.split(":").str[0], errors="coerce")
    p["secs"] = pd.to_numeric(t.str.split(":").str[1], errors="coerce")
    p = p.sort_values(["qtr", "mins", "secs", "id"], ascending=[True, False, False, True], kind="stable")
    p["home_team"], p["away_team"] = game["home_team"], game["away_team"]
    p["defteam"] = np.where(p["posteam"] == p["home_team"], p["away_team"], p["home_team"])
    p["half"] = np.where(p["qtr"] <= 2, 1, 2)

    challenge = p["desc"].str.extract(r"([A-Za-z]*\s*[A-Za-z]*\s*[A-Za-z]*[A-Za-z]+)(?=\schallenged)")[0]
    challenge = challenge.str.replace(r"[\r\n]", "", regex=True).str.strip()
    desc_timeout = p["desc"].str.contains(r"Timeout #\d", regex=True)
    to_team = p["desc"].str.extract(r"(?<=Timeout #\d by )([A-Z]{2,3})")[0].map(fix_abbr)
    to_team = to_team.where(to_team.notna(), challenge.map(_CHALLENGE_TEAMS).where(desc_timeout))
    p["timeout_team"] = to_team
    for side in ("home", "away"):
        used = (p["timeout_team"] == p[f"{side}_team"]).astype(int)
        p[f"{side}_timeouts_remaining"] = 3 - used.groupby(p["half"]).cumsum().clip(upper=3)
    is_home = p["posteam"] == p["home_team"]
    p["posteam_timeouts_remaining"] = np.where(is_home, p["home_timeouts_remaining"], p["away_timeouts_remaining"])
    p["defteam_timeouts_remaining"] = np.where(is_home, p["away_timeouts_remaining"], p["home_timeouts_remaining"])
    p["quarter_seconds_remaining"] = 60 * p["mins"] + p["secs"]
    # score before the play
    p["home_score"] = p["home_score"].shift(1)
    p["away_score"] = p["away_score"].shift(1)
    p["score_differential"] = np.where(is_home, p["home_score"] - p["away_score"], p["away_score"] - p["home_score"])
    first_pos = p["posteam"].dropna().iloc[0]
    p["home_opening_kickoff"] = int(first_pos == game["home_team"])
    p["season"], p["type"], p["game_id"], p["espn_id"] = game["season"], game["type"], game["game_id"], game["espn"]

    p = p[
        (p["down"] == 4)
        & ~((p["quarter_seconds_remaining"] < 15) & (p["qtr"] == 4))
        & p["timeout_team"].isna()
        & (p["type_text"] != "Two-minute warning")
        & (p["type_text"] != "End Period")
    ]
    p = p.drop_duplicates(["qtr", "quarter_seconds_remaining", "ydstogo"]).sort_values(
        ["qtr", "quarter_seconds_remaining", "ydstogo"], ascending=[True, False, True]
    )
    # yardline from the "TEAM 35" text when available (more reliable than yardsToEndzone)
    side = p["yardline"].fillna("").str.split(" ").str[0].map(fix_abbr)
    num = pd.to_numeric(p["yardline"].fillna("").str.split(" ").str[1], errors="coerce")
    temp = np.where((side == p["posteam"]) | (p["yardline_100"] == 50), 100 - num, num)
    p["yardline_100"] = np.where(~np.isnan(temp), temp, p["yardline_100"])

    # end-of-game run-off heuristics (nfl4th's; nfl4th itself never applies them, see
    # PORTING_NOTES 5b, but the market engine does)
    q4, lead, qsr = p["qtr"] == 4, p["score_differential"] > 0, p["quarter_seconds_remaining"]
    p["runoff"] = 0.0
    p.loc[q4 & lead & qsr.between(167, 300), "runoff"] = 40
    p.loc[q4 & lead & qsr.between(127, 166), "runoff"] = qsr - 120 - 6
    p.loc[q4 & lead & (qsr <= 120), "runoff"] = 40
    cols = [
        "game_id", "espn_id", "id", "desc", "type", "qtr", "quarter_seconds_remaining", "posteam",
        "away_team", "home_team", "yardline", "yardline_100", "ydstogo", "posteam_timeouts_remaining",
        "defteam_timeouts_remaining", "home_opening_kickoff", "score_differential", "runoff",
        "home_score", "away_score", "type_text", "season", "wallclock",
    ]
    return p[cols].rename(columns={"id": "play_id"}).reset_index(drop=True)


# -- live situation ----------------------------------------------------------------------


def fetch_scoreboard(dates: str | None = None) -> dict:
    r = requests.get(SCOREBOARD_URL, params={"dates": dates} if dates else None, timeout=20)
    r.raise_for_status()
    return r.json()


_CLOCK = re.compile(r"^(\d+):(\d{2})$")


def current_situation(scoreboard: dict, espn_id: str, game: dict, home_opening_kickoff: int | None) -> dict | None:
    """The live pre-snap situation of one game as an add_4th_probs() input row, or None
    if the game isn't live or has no situation. Not verified against a live game yet:
    field names follow ESPN's scoreboard `situation` block."""
    ev = next((e for e in scoreboard.get("events", []) if str(e.get("id")) == str(espn_id)), None)
    if ev is None:
        return None
    comp = ev["competitions"][0]
    status = comp.get("status") or ev.get("status") or {}
    sit = comp.get("situation")
    if not sit or (status.get("type") or {}).get("state") != "in":
        return None
    teams = {c["team"]["id"]: (fix_abbr(c["team"]["abbreviation"]), c["homeAway"], float(c.get("score") or 0))
             for c in comp["competitors"]}
    pos = teams.get(str(sit.get("possession")))
    if pos is None:
        return None
    posteam, side, _ = pos
    home = next(v for v in teams.values() if v[1] == "home")
    away = next(v for v in teams.values() if v[1] == "away")
    m = _CLOCK.match(status.get("displayClock", ""))
    qsr = 60 * int(m[1]) + int(m[2]) if m else None
    text = sit.get("possessionText") or ""  # e.g. "NO 35"
    parts = text.split(" ")
    yl = None
    if len(parts) == 2 and parts[1].isdigit():
        yl = 100 - int(parts[1]) if fix_abbr(parts[0]) == posteam else int(parts[1])
    elif text.strip() == "50" or sit.get("yardLine") == 50:
        yl = 50
    home_to, away_to = sit.get("homeTimeouts"), sit.get("awayTimeouts")
    is_home = side == "home"
    return {
        "game_id": game["game_id"],
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "posteam": posteam,
        "type": game["type"],
        "season": game["season"],
        "qtr": status.get("period"),
        "quarter_seconds_remaining": qsr,
        "down": sit.get("down"),
        "ydstogo": sit.get("distance"),
        "yardline_100": yl,
        "score_differential": (home[2] - away[2]) * (1 if is_home else -1),
        "home_opening_kickoff": home_opening_kickoff,
        "posteam_timeouts_remaining": home_to if is_home else away_to,
        "defteam_timeouts_remaining": away_to if is_home else home_to,
        "situation_text": sit.get("downDistanceText"),
    }

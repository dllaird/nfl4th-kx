"""Map Kalshi NFL events/markets to nflverse games, without hardcoded tickers.

Tickers look like `KXNFLGAME-25SEP28GBDAL-GB`, `KXNFLSPREAD-25SEP28GBDAL-GB7`,
`KXNFLTOTAL-25SEP28GBDAL-47`. The event code concatenates team codes of variable
length, so teams are read from market suffixes (or `yes_sub_title`) and games are
matched to the schedule on {home, away} plus a date within +-1 day (ticker dates can
differ across series for the same game, e.g. flexed games).
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

import pandas as pd

from nfl4th_py.market.client import KalshiClient

WINNER = "KXNFLGAME"
SPREAD = "KXNFLSPREAD"
TOTAL = "KXNFLTOTAL"
# winner markets for games missing from KXNFLGAME (conference championships, Super Bowl)
CHAMPION_SERIES = ("KXSB", "KXNFLAFCCHAMP", "KXNFLNFCCHAMP")

KALSHI_TO_NFLVERSE = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX", "LVR": "LV", "ARZ": "ARI"}

_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
_EVENT_RE = re.compile(r"^(?P<series>[A-Z0-9]+)-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?P<teams>[A-Z]+)$")


def team_code(kalshi_code: str) -> str:
    return KALSHI_TO_NFLVERSE.get(kalshi_code, kalshi_code)


def event_date(event_ticker: str) -> date | None:
    m = _EVENT_RE.match(event_ticker)
    if not m:
        return None
    return date(2000 + int(m["yy"]), _MONTHS[m["mon"]], int(m["dd"]))


def market_suffix(ticker: str) -> str:
    return ticker.rsplit("-", 1)[1]


def fetch_markets(
    client: KalshiClient, series: list[str], historical: bool, status: str | None = None
) -> pd.DataFrame:
    rows = []
    for s in series:
        for m in client.iter_markets(series_ticker=s, historical=historical, status=status):
            m["series_ticker"] = s
            rows.append(m)
    df = pd.DataFrame(rows).drop(columns=["custom_strike"], errors="ignore")
    if df.empty:
        return df
    df["event_date"] = df["event_ticker"].map(event_date)
    suffix = df["ticker"].map(market_suffix)
    is_total = df["series_ticker"] == TOTAL
    letters = suffix.str.extract(r"^([A-Z]+)", expand=False).where(~is_total)
    # a few tickers use a city name ("BUFFALO") instead of a code; map those through
    # the winner markets' yes_sub_title ("Buffalo") -> code
    names = {}
    win = df[(df["series_ticker"] == WINNER) & (letters.str.len() <= 3)]
    for title, code in zip(win["yes_sub_title"], letters[win.index]):
        if isinstance(title, str):
            names[title.upper().replace(" ", "")] = code
    df["team"] = letters.map(
        lambda c: team_code(names.get(c, c)) if isinstance(c, str) else None
    )
    return df


def match_games(markets: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Attach nflverse `game_id` to KXNFLGAME/SPREAD/TOTAL markets.

    `schedule` needs game_id, gameday (YYYY-MM-DD), home_team, away_team.
    Events whose teams + date match no game (e.g. preseason) get game_id = NaN.
    """
    sched = schedule.assign(gameday=pd.to_datetime(schedule["gameday"]).dt.date)
    by_pair: dict[frozenset, list[tuple[date, str]]] = {}
    for r in sched.itertuples():
        by_pair.setdefault(frozenset((r.home_team, r.away_team)), []).append((r.gameday, r.game_id))

    # an event's teams = all team codes among its markets; totals inherit from the
    # winner/spread event with the same event code
    ev = markets.dropna(subset=["team"]).groupby("event_ticker")["team"].agg(lambda s: frozenset(s))
    code = markets["event_ticker"].str.split("-", n=1).str[1]
    teams_by_code = ev.rename(lambda e: e.split("-", 1)[1]).groupby(level=0).agg(
        lambda s: frozenset().union(*s)
    )
    markets = markets.assign(event_code=code)
    markets["teams"] = markets["event_code"].map(teams_by_code)

    def find(row) -> str | None:
        teams, d = row["teams"], row["event_date"]
        if not isinstance(teams, frozenset) or len(teams) != 2 or d is None:
            return None
        for gd, gid in by_pair.get(teams, []):
            if abs((gd - d).days) <= 1:
                return gid
        return None

    keys = markets[["event_code", "teams", "event_date"]].drop_duplicates("event_code")
    keys["game_id"] = keys.apply(find, axis=1)
    out = markets.merge(keys[["event_code", "game_id"]], on="event_code", how="left")
    out["teams"] = out["teams"].map(lambda t: "|".join(sorted(t)) if isinstance(t, frozenset) else None)
    return out


def champion_winner_markets(
    client: KalshiClient, schedule: pd.DataFrame, game_ids: list[str]
) -> pd.DataFrame:
    """For games with no KXNFLGAME event (conference championships, Super Bowl), use the
    two participants' markets in the champion series. Those settle on the same game, so
    during the game they are winner markets."""
    sched = schedule.set_index("game_id")
    champ = fetch_markets(client, list(CHAMPION_SERIES), historical=True)
    champ["team"] = champ["ticker"].map(market_suffix).map(team_code)
    out = []
    for gid in game_ids:
        g = sched.loc[gid]
        gd = pd.to_datetime(g["gameday"]).date()
        teams = {g["home_team"], g["away_team"]}
        close = pd.to_datetime(champ["close_time"], utc=True, format="ISO8601").dt.date
        cand = champ[champ["team"].isin(teams) & (abs(close - gd) <= timedelta(days=2))]
        # both participants' markets close at the end of this game
        if set(cand["team"]) == teams:
            for series, grp in cand.groupby("series_ticker"):
                if set(grp["team"]) == teams:
                    out.append(grp.assign(game_id=gid, winner_source=series))
                    break
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def utc(ts: str | None) -> datetime | None:
    return None if ts is None else pd.Timestamp(ts).tz_convert("UTC").to_pydatetime()


def dedupe_events(markets: pd.DataFrame) -> pd.DataFrame:
    """Keep one event per (game, series): the one with the most volume. Kalshi sometimes
    lists a second, untraded event for a game whose date changed."""
    m = markets.dropna(subset=["game_id"])
    vol = m.groupby(["game_id", "series_ticker", "event_ticker"])["volume"].sum().reset_index()
    best = vol.sort_values("volume").groupby(["game_id", "series_ticker"]).tail(1)["event_ticker"]
    return m[m["event_ticker"].isin(set(best))].reset_index(drop=True)


def games_needing_champion_fallback(markets: pd.DataFrame, schedule: pd.DataFrame) -> list[str]:
    """Postseason games whose KXNFLGAME winner markets are missing or never traded."""
    win = markets[markets["series_ticker"] == WINNER].groupby("game_id")["volume"].sum()
    post = schedule.loc[schedule["game_type"].isin(["CON", "SB"]), "game_id"]
    return [g for g in post if win.get(g, 0) <= 0]

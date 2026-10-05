"""Live 4th-down decisions: ESPN game state + Kalshi markets -> decision table.

Read-only: ESPN public JSON and Kalshi public REST market data (no API key, no orders).
"""

from __future__ import annotations

import logging
import time
from functools import cache
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nfl4th_py import decision, prep
from nfl4th_py.market import discovery, pregame
from nfl4th_py.market.book import posteam_win_price, quote_from_orderbook
from nfl4th_py.market.client import KalshiClient
from nfl4th_py.models import kicker
from nfl4th_py.scrape import espn

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"


def season_kickers(season: int) -> pd.DataFrame:
    """Each team's kickers by game this season (and last season, for week 1)."""
    frames = []
    for s in (season - 1, season):
        try:
            pbp = pd.read_parquet(PBP_URL.format(season=s),
                                  columns=["game_id", "season", "week", "posteam", "play_type", "kicker_player_id"])
        except Exception:  # the current season's file may not exist yet
            log.warning("no nflverse play-by-play for %s", s)
            continue
        frames.append(pbp[pbp["play_type"].isin(["field_goal", "extra_point"]) & pbp["kicker_player_id"].notna()])
    return kicker.team_kickers(pd.concat(frames)) if frames else pd.DataFrame(
        columns=["season", "week", "game_id", "posteam", "kicker_player_id"])

log = logging.getLogger(__name__)


@dataclass
class GameMarkets:
    winner: dict[str, str]  # nflverse team -> winner-market ticker
    markets: pd.DataFrame  # all of this game's Kalshi markets


def find_game_markets(client: KalshiClient, game: pd.Series, schedule: pd.DataFrame) -> GameMarkets:
    """Open (or recently closed) winner/spread/total markets for one game."""
    frames = [discovery.fetch_markets(client, [s], historical=False, status=st)
              for s in (discovery.WINNER, discovery.SPREAD, discovery.TOTAL) for st in ("open", "closed")]
    mk = pd.concat([f for f in frames if not f.empty], ignore_index=True).drop_duplicates("ticker")
    mk = discovery.match_games(mk, schedule)
    mk = discovery.dedupe_events(mk[mk["game_id"] == game["game_id"]])
    win = mk[mk["series_ticker"] == discovery.WINNER]
    if win["volume"].sum() <= 0:
        champ = discovery.champion_winner_markets(client, schedule, [game["game_id"]])
        if not champ.empty:
            mk = pd.concat([mk[mk["series_ticker"] != discovery.WINNER],
                            champ.assign(champion_series=champ["series_ticker"], series_ticker=discovery.WINNER)])
            win = mk[mk["series_ticker"] == discovery.WINNER]
    return GameMarkets(dict(zip(win["team"], win["ticker"])), mk.reset_index(drop=True))


def pregame_strength(client: KalshiClient, game: pd.Series, gm: GameMarkets, kickoff_ts: int) -> dict:
    """Kalshi pregame implied spread/total from 1-minute candles up to kickoff."""
    candles = {}
    for series, grp in gm.markets.groupby("series_ticker"):
        opened = pd.to_datetime(grp["open_time"], utc=True, format="ISO8601")
        grp = grp[opened.map(lambda x: int(x.timestamp())) < kickoff_ts]
        rows = []
        for m in grp.itertuples():
            real_series = getattr(m, "champion_series", None) or series
            if not isinstance(real_series, str):
                real_series = series
            for c in client.get_candlesticks(real_series, m.ticker, kickoff_ts - 3 * 3600, kickoff_ts, 1):
                rows.append({**c, "ticker": m.ticker})
        candles[series] = pd.DataFrame(rows)
    return pregame.game_strength(game, gm.markets, candles, kickoff_ts)


def winner_snapshot(client: KalshiClient, gm: GameMarkets, posteam: str, opp: str) -> dict:
    tickers = [gm.winner[t] for t in (posteam, opp) if t in gm.winner]
    books = {b.get("ticker") or b.get("market_ticker"): b for b in client.get_orderbooks(tickers)}
    if len(books) < len(tickers):  # fall back to one call per market
        books = {t: client.get_orderbook(t) for t in tickers}
    q_team = quote_from_orderbook(books[gm.winner[posteam]]) if posteam in gm.winner else None
    q_opp = quote_from_orderbook(books[gm.winner[opp]]) if opp in gm.winner else None
    price, source = posteam_win_price(q_team, q_opp)
    status = {m["ticker"]: m.get("status") for m in client.iter_markets(tickers=tickers)}
    widths = [q.spread for q in (q_team, q_opp) if q is not None and q.spread is not None]
    return {
        "price": price,
        "price_source": source,
        "bid_ask": float(np.mean(widths)) if widths else np.nan,
        "quote_age_s": 0.0 if price is not None else np.nan,
        "n_trades": sum(1 for q in (q_team, q_opp) if q is not None and q.two_sided),
        "halted": any(s not in ("active", "open", None) for s in status.values()),
        "raw": {"books": books, "status": status, "fetched_at": datetime.now(timezone.utc).isoformat()},
    }


def decision_table(row: pd.Series) -> pd.DataFrame:
    """Rows: Go / FG / Punt with EV, sd, success prob and success/fail values (percent)."""
    t = pd.DataFrame(
        {
            "choice": ["Go for it", "Field goal attempt", "Punt"],
            "ev": [row.get("go_ev"), row.get("fg_ev"), row.get("punt_ev")],
            "sd": [row.get("go_sd"), row.get("fg_sd"), row.get("punt_sd")],
            "success_prob": [row.get("first_down_prob"), row.get("fg_make_prob"), np.nan],
            "fail_value": [row.get("wp_fail"), row.get("miss_fg_wp"), np.nan],
            "success_value": [row.get("wp_succeed"), row.get("make_fg_wp"), np.nan],
            "support": [row.get("go_support"), row.get("fg_support"), row.get("punt_support")],
        }
    )
    for c in ("ev", "sd", "success_prob", "fail_value", "success_value"):
        t[c] = (100 * t[c].astype(float)).round(1)
    t["support"] = t["support"].astype(float).round(3)
    return t


def _spot(yl: float) -> str:
    yl = int(yl)
    return f"own {100 - yl}" if yl > 50 else ("midfield" if yl == 50 else f"opp {yl}")


@cache
def _kicker_names() -> pd.Series:
    return kicker.KickerFGModel.load().kickers["name"]


def format_decision(row: pd.Series, situation: dict) -> str:
    kid = row.get("kicker_player_id")
    kname = _kicker_names().get(kid, "league-average kicker") if isinstance(kid, str) else "league-average kicker"
    head = (
        f"{situation['posteam']} 4th & {situation['ydstogo']} at {_spot(situation['yardline_100'])} | "
        f"Q{situation['qtr']} {situation['quarter_seconds_remaining']}s | score diff "
        f"{situation['score_differential']:+.0f} | Kalshi win price {row['winner_price']:.3f} "
        f"(bid/ask {row['winner_bid_ask']:.3f}) | pregame spread {row['kalshi_posteam_spread']:+.1f}, "
        f"total {row['kalshi_total_line']:.1f} | kicker {kname}"
    )
    if row["no_recommendation"]:
        return head + f"\n  NO RECOMMENDATION: {row['no_recommendation_reason']}"
    table = decision_table(row).to_string(index=False)
    rec = (
        f"  RECOMMEND: {row['recommendation'].upper()} by {100 * row['ev_gap']:.1f} pts "
        f"(+-{100 * row['ev_gap_sd']:.1f}; P(best) {row['prob_best_option_is_best']:.2f})"
    )
    return f"{head}\n{table}\n{rec}"


@dataclass
class LiveSession:
    game_id: str
    log_dir: Path | None = None
    config: decision.DecisionConfig = field(default_factory=decision.DecisionConfig)
    client: KalshiClient = field(default_factory=lambda: KalshiClient(rate_per_sec=3))

    def __post_init__(self) -> None:
        self.games = prep.load_games()
        self.schedule = pd.read_csv(prep.SCHEDULES_URL, low_memory=False)
        g = self.games.loc[self.games["game_id"] == self.game_id]
        if g.empty:
            raise ValueError(f"unknown game_id {self.game_id}")
        self.game = g.iloc[0]
        self.markets = find_game_markets(self.client, self.game, self.schedule[self.schedule["season"] == self.game["season"]])
        if len(self.markets.winner) < 2:
            log.warning("winner markets found: %s", self.markets.winner)
        try:
            self.kicker_lookup = season_kickers(int(self.game["season"]))
        except Exception:
            log.exception("kicker lookup failed; using the league kicker curve")
            self.kicker_lookup = None
        self.strength: dict | None = None
        self.strength_frozen = False  # frozen once computed at (past) kickoff
        self.strength_at = 0.0
        self.home_opening_kickoff: int | None = None
        self.seen: set = set()

    def _kickoff_ts(self, summary: dict) -> int:
        plays = espn.plays_frame(summary)
        wall = pd.to_datetime(plays["wallclock"], utc=True, errors="coerce").dropna() if not plays.empty else []
        if len(wall):
            return int(wall.min().timestamp())
        date = summary["header"]["competitions"][0]["date"]
        return int(pd.Timestamp(date).timestamp())

    def evaluate(self, situation: dict) -> pd.DataFrame:
        posteam = situation["posteam"]
        opp = self.game["away_team"] if posteam == self.game["home_team"] else self.game["home_team"]
        w = winner_snapshot(self.client, self.markets, posteam, opp)
        s = self.strength or {}
        snap = pd.DataFrame([{
            "price": w["price"], "bid_ask": w["bid_ask"], "quote_age_s": w["quote_age_s"],
            "n_trades": w["n_trades"], "halted": w["halted"],
            "kalshi_spread_line": s.get("kalshi_spread_line"), "kalshi_total_line": s.get("kalshi_total_line"),
            "kalshi_spread_quality": s.get("kalshi_spread_quality"), "kalshi_total_quality": s.get("kalshi_total_quality"),
        }])
        play = pd.DataFrame([{k: situation[k] for k in prep.REQUIRED_4TH_COLUMNS}])
        # the team's kicker: whoever kicked for it today, else its most recent kicker
        play["kicker_player_id"] = (
            kicker.kicker_for(self.kicker_lookup, self.game_id, posteam, int(self.game["season"]), int(self.game["week"]))
            if self.kicker_lookup is not None else None)
        out = decision.add_4th_probs(play, snap, self.games, config=self.config)
        if self.log_dir:
            decision.log_decisions(out, [w["raw"]], self.log_dir)
        return out

    def poll_once(self) -> str | None:
        summary = espn.fetch_summary(self.game["espn"])
        if self.home_opening_kickoff is None:
            plays = espn.plays_frame(summary)
            if not plays.empty:
                self.home_opening_kickoff = int(plays["posteam"].dropna().iloc[0] == self.game["home_team"])
        now = time.time()
        if not self.strength_frozen and (self.strength is None or now - self.strength_at > 300):
            kickoff = self._kickoff_ts(summary)
            # before kickoff the "pregame" line is provisional: the latest ladder, refreshed
            # every 5 minutes; from kickoff on it is the ladder at kickoff, fixed for the game
            self.strength = pregame_strength(self.client, self.game, self.markets, int(min(kickoff, now)))
            self.strength_frozen = kickoff <= now
            self.strength_at = now
            log.info("Kalshi pregame strength: %s", {k: self.strength[k] for k in
                     ("kalshi_spread_line", "kalshi_total_line", "kalshi_spread_quality", "kalshi_total_quality")})
        sit = espn.current_situation(espn.fetch_scoreboard(), self.game["espn"], self.game, self.home_opening_kickoff)
        if not sit or sit.get("down") != 4 or None in (sit["yardline_100"], sit["quarter_seconds_remaining"]):
            return None
        key = (sit["qtr"], sit["quarter_seconds_remaining"], sit["yardline_100"], sit["ydstogo"], sit["posteam"])
        if key in self.seen:
            return None
        self.seen.add(key)
        out = self.evaluate(sit)
        return format_decision(out.iloc[0], sit)

    def run(self, interval_s: float = 10.0) -> None:
        while True:
            try:
                msg = self.poll_once()
                if msg:
                    print(msg, flush=True)
            except Exception:  # keep polling through transient feed errors
                log.exception("poll failed")
            time.sleep(interval_s)

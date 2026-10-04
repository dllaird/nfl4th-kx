"""Backfill Kalshi trades and 1-minute candles for the 2025 NFL season (read-only).

Per game (kickoff = first nflfastR snap/kickoff `time_of_day`, end = last play + 15 min):
  winner markets  : trades and candles from kickoff - 3h to end
  spread / total  : candles (and with --ladder-trades, trades) from kickoff - 3h to
                    kickoff + 10 min, for strikes open before kickoff that ever traded.
                    Only the pregame ladder is used (team strength is pregame-only).

Writes data_cache/kalshi/{trades,candles}/<series>/<game_id>.parquet. Re-running skips
files that already exist, so the script is resumable.

Usage: python scripts/backfill_kalshi.py [--games N] [--workers 3] [--rate 3]
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from nfl4th_py.market import discovery
from nfl4th_py.market.client import KalshiClient

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data_cache"
OUT = CACHE / "kalshi"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.parquet"

log = logging.getLogger("backfill")


def game_windows() -> pd.DataFrame:
    path = CACHE / "pbp_2025.parquet"
    if not path.exists():
        pd.read_parquet(PBP_URL).to_parquet(path)
    pbp = pd.read_parquet(path, columns=["game_id", "time_of_day"])
    t = pd.to_datetime(pbp["time_of_day"], utc=True, format="ISO8601")
    w = t.groupby(pbp["game_id"]).agg(["min", "max"])
    return pd.DataFrame(
        {
            "kickoff": w["min"].map(lambda x: int(x.timestamp())),
            "end": w["max"].map(lambda x: int(x.timestamp())) + 15 * 60,
        }
    )


def fetch_one(client: KalshiClient, kind: str, series: str, game_id: str, mk: pd.DataFrame, lo: int, hi: int) -> str:
    path = OUT / kind / series / f"{game_id}.parquet"
    if path.exists():
        return "skip"
    rows = []
    for m in mk.itertuples():
        if kind == "trades":
            got = list(client.iter_trades(m.ticker, min_ts=lo, max_ts=hi, historical=True))
        else:
            got = client.get_candlesticks(m.series_ticker, m.ticker, lo, hi, 1, historical=True)
        for r in got:
            r["ticker"] = m.ticker
        rows.extend(got)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame({"ticker": pd.Series(dtype=str)})
    df["game_id"] = game_id
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)
    return f"{len(df)} rows"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=None)
    ap.add_argument("--ladder-trades", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    # unauthenticated limit is undocumented; 8 req/s drew 429s, 3 req/s did not
    ap.add_argument("--rate", type=float, default=3.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    markets = pd.read_parquet(CACHE / "markets_2025.parquet")
    windows = game_windows()
    games = sorted(set(markets["game_id"]) & set(windows.index))[: args.games]
    client = KalshiClient(rate_per_sec=args.rate)

    jobs = []
    for gid in games:
        k, end = int(windows.at[gid, "kickoff"]), int(windows.at[gid, "end"])
        for series, grp in markets[markets["game_id"] == gid].groupby("series_ticker"):
            is_ladder = series != discovery.WINNER
            if is_ladder:
                opened = pd.to_datetime(grp["open_time"], utc=True, format="ISO8601")
                grp = grp[(opened.map(lambda x: int(x.timestamp())) < k) & (grp["volume"] > 0)]
            # champion-series winner markets keep their real series for the candles path
            if "champion_series" in grp and grp["champion_series"].notna().any():
                grp = grp.assign(series_ticker=grp["champion_series"])
            lo = k - 3 * 3600
            hi = end if series == discovery.WINNER else k + 600
            kinds = ("candles", "trades") if (not is_ladder or args.ladder_trades) else ("candles",)
            for kind in kinds:
                jobs.append((kind, series, gid, grp, lo, hi))

    log.info("%d games, %d jobs", len(games), len(jobs))
    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(fetch_one, client, *j): j[:3] for j in jobs}
        for f in as_completed(futs):
            done += 1
            kind, series, gid = futs[f]
            try:
                res = f.result()
            except Exception:
                log.exception("failed %s %s %s", kind, series, gid)
                continue
            if done % 50 == 0 or res != "skip":
                log.info("[%d/%d] %s %s %s: %s", done, len(jobs), kind, series, gid, res)


if __name__ == "__main__":
    main()

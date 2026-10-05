"""Command line: `nfl4th-py live --game <nflverse game_id>`.

There is no `record` command: the live orderbook recorder was dropped from scope
(2026-10-04); history comes from the Kalshi trade tape via scripts/backfill_kalshi.py.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nfl4th_py.decision import DecisionConfig


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="nfl4th-py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    live = sub.add_parser("live", help="poll a live game and print a decision table for each 4th down")
    live.add_argument("--game", required=True, help="nflverse game_id, e.g. 2026_05_ATL_NO")
    live.add_argument("--interval", type=float, default=10.0, help="seconds between polls")
    live.add_argument("--log-dir", type=Path, default=Path("decisions"), help="parquet decision log")
    live.add_argument("--once", action="store_true", help="evaluate the current situation once and exit")
    live.add_argument("--max-quote-age", type=float, default=None)
    live.add_argument("--max-bid-ask", type=float, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "live":
        from nfl4th_py.live import LiveSession

        overrides = {k: v for k, v in {"max_quote_age_s": args.max_quote_age, "max_bid_ask": args.max_bid_ask}.items() if v is not None}
        session = LiveSession(args.game, log_dir=args.log_dir, config=DecisionConfig(**overrides))
        if args.once:
            print(session.poll_once() or "no live 4th-down situation right now")
        else:
            session.run(args.interval)


if __name__ == "__main__":
    main()

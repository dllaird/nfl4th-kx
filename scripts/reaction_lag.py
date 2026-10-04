"""Measure how fast the Kalshi winner market reacts to plays (Phase 4 timing).

For plays whose market move (pre-snap -> just before the next snap) is at least
MIN_MOVE, compute the median fraction of that move completed at each offset from the
play end (or from the snap, for the coarse-timestamp check). Prints tables used in
DATASET.md.

Usage: python scripts/reaction_lag.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nfl4th_py.market import alignment

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data_cache"
MIN_MOVE = 0.03
OFFSETS = np.array([-10, -5, -3, -1, 0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60], float)


def frac_at(series, ref, pre_t, post_t, floor, offsets, window=3.0):
    pre = alignment.price_at(series, pre_t, 5.0).price
    post = alignment.price_at(series, post_t, 5.0, t_start_min=floor).price
    if pre is None or post is None or abs(post - pre) < MIN_MOVE:
        return None
    out = {}
    for off in offsets:
        t = ref + off
        if t >= post_t:
            break
        p = alignment.price_at(series, t, window).price
        if p is not None:
            out[off] = (p - pre) / (post - pre)
    return out


def main() -> None:
    d = pd.read_parquet(CACHE / "dataset" / "plays_2025.parquet")
    d = d[d["next_snap_ts"].notna() & ~d["flag_nonmonotonic_time"]]
    rows = []
    for gid, g in d.groupby("game_id"):
        trades = pd.read_parquet(CACHE / "kalshi" / "trades" / "KXNFLGAME" / f"{gid}.parquet")
        s = alignment.home_price_series(trades, g["home_ticker"].iat[0], g["away_ticker"].iat[0])
        for r in g.itertuples():
            post_t = r.next_snap_ts - 1.0
            if not np.isnan(r.end_ts):
                f = frac_at(s, r.end_ts, r.snap_ts, post_t, r.end_ts, OFFSETS)
                if f:
                    rows += [("end", "precise" if not r.flag_coarse_time else "coarse", r.play_type, k, v) for k, v in f.items()]
            f = frac_at(s, r.snap_ts, r.snap_ts, post_t, r.snap_ts, OFFSETS)
            if f:
                rows += [("snap", "coarse" if r.flag_coarse_time else "precise", r.play_type, k, v) for k, v in f.items()]
    res = pd.DataFrame(rows, columns=["ref", "time", "play_type", "offset", "frac"])
    res.to_parquet(CACHE / "reaction_lag.parquet")
    pd.set_option("display.width", 200)
    for ref in ("end", "snap"):
        sub = res[res["ref"] == ref]
        tab = sub.groupby(["offset", "time"])["frac"].median().unstack().round(2)
        n = sub[sub["offset"] == 0].groupby("time").size()
        print(f"\nmedian fraction of the move completed vs seconds from play {ref} (n plays: {n.to_dict()})")
        print(tab.to_string())


if __name__ == "__main__":
    main()

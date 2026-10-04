"""Align plays with winner-market prices built from the Kalshi trade tape.

A game's two winner markets are merged into one home-win price series: a YES trade
at x on the home market is a home price of x, and on the away market 1 - x. Trades
execute at the bid or the ask, so a single trade is a noisy price. `price_at()`
therefore uses a short volume-weighted window and reports how many trades it used,
their dispersion and the age of the newest one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def home_price_series(trades: pd.DataFrame, home_ticker: str, away_ticker: str) -> pd.DataFrame:
    """Columns: t (float Unix seconds), p_home, count, ticker, taker_side; sorted by t."""
    t = trades[trades["ticker"].isin([home_ticker, away_ticker])].copy()
    t["t"] = pd.to_datetime(t["created_time"], utc=True, format="ISO8601").map(lambda x: x.timestamp())
    t["p_home"] = np.where(t["ticker"] == home_ticker, t["yes_price"], 1 - t["yes_price"])
    return t[["t", "p_home", "count", "ticker", "taker_side"]].sort_values("t", kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class WindowPrice:
    price: float | None
    n_trades: int
    volume: float
    age_s: float | None  # seconds from the newest trade used to the reference time
    dispersion: float | None  # volume-weighted sd of trade prices in the window


def price_at(series: pd.DataFrame, t_end: float, window_s: float, t_start_min: float | None = None) -> WindowPrice:
    """VWAP of home-price trades in (t_end - window_s, t_end]; never earlier than
    `t_start_min` (e.g. the previous play's end, so a post-play window can't reach
    back before the play). Falls back to nothing (None) if the window is empty."""
    t = series["t"].to_numpy()
    lo = t_end - window_s if t_start_min is None else max(t_end - window_s, t_start_min)
    i0, i1 = np.searchsorted(t, lo, side="right"), np.searchsorted(t, t_end, side="right")
    if i1 <= i0:
        return WindowPrice(None, 0, 0.0, None, None)
    p = series["p_home"].to_numpy()[i0:i1]
    v = series["count"].to_numpy()[i0:i1]
    vw = float(np.average(p, weights=v)) if v.sum() > 0 else float(p.mean())
    disp = float(np.sqrt(np.average((p - vw) ** 2, weights=v))) if v.sum() > 0 else None
    return WindowPrice(vw, int(i1 - i0), float(v.sum()), float(t_end - t[i1 - 1]), disp)


def last_trade_price(series: pd.DataFrame, t_end: float) -> tuple[float | None, float | None]:
    t = series["t"].to_numpy()
    i = np.searchsorted(t, t_end, side="right") - 1
    if i < 0:
        return None, None
    return float(series["p_home"].iat[i]), float(t_end - t[i])


def reaction_profile(
    series: pd.DataFrame,
    snaps: np.ndarray,
    play_ends: np.ndarray,
    next_snaps: np.ndarray,
    offsets: np.ndarray,
    window_s: float = 5.0,
) -> pd.DataFrame:
    """Event study: home price at `play_end + offset` for each play, relative to the
    pre-snap price, normalized by the total move measured just before the next snap.
    Returns one row per (play, offset) with `frac` = move so far / total move."""
    rows = []
    for k, (s, e, n) in enumerate(zip(snaps, play_ends, next_snaps)):
        pre = price_at(series, s, window_s).price
        post = price_at(series, n - 1.0, window_s, t_start_min=e).price
        if pre is None or post is None:
            continue
        total = post - pre
        for off in offsets:
            if e + off >= n:
                break
            p = price_at(series, e + off, window_s, t_start_min=s).price
            if p is not None:
                rows.append((k, off, p - pre, total))
    df = pd.DataFrame(rows, columns=["play", "offset", "move", "total"])
    df["frac"] = df["move"] / df["total"].where(df["total"].abs() > 0)
    return df

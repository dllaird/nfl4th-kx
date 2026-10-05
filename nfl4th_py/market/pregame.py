"""Kalshi pregame team strength: implied spread and total at kickoff.

Team strength for the conversion model comes only from Kalshi pregame prices: the
spread and total ladders (and the winner price, as a consistency check), snapshotted
at kickoff and held fixed for the whole game. This mirrors how nfl4th uses the
pregame betting line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from nfl4th_py.market import ladders

MARGIN_SD = 13.5  # typical sd of an NFL final margin, used to turn a win price into a spread
# below this ladder fit quality the spread is the average of the ladder and the
# winner-price-implied spread: on the 10 such 2025 games the average misses the betting
# line by 0.76 points, the ladder alone by 1.22, the winner price alone by 0.95
WEAK_LADDER_QUALITY = 0.2

SNAPSHOT_COLUMNS = ["ticker", "bid", "ask", "age_s", "volume", "vwap", "vwap_age_s"]


def snapshot_from_candles(candles: pd.DataFrame, at_ts: int, volume_window_s: int = 3600) -> pd.DataFrame:
    """Per market: the last 1-minute candle ending at or before `at_ts` with a quote,
    its age, and over the preceding window the traded volume, the volume-weighted
    average trade price (from candle mean price x volume) and its volume-weighted age."""
    if candles.empty:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    c = candles[candles["end_period_ts"] <= at_ts]
    win = c[(c["end_period_ts"] > at_ts - volume_window_s) & c["volume"].gt(0) & c["price_mean"].notna()]
    win = win.assign(pv=win["price_mean"] * win["volume"], av=(at_ts - win["end_period_ts"]) * win["volume"])
    agg = win.groupby("ticker")[["volume", "pv", "av"]].sum()
    vol = agg["volume"]
    vwap = agg["pv"] / agg["volume"]
    vwap_age = agg["av"] / agg["volume"]
    quoted = c.dropna(subset=["yes_bid_close", "yes_ask_close"])
    last = quoted.sort_values("end_period_ts").groupby("ticker").tail(1)
    out = pd.DataFrame(
        {
            "ticker": last["ticker"].to_numpy(),
            "bid": last["yes_bid_close"].to_numpy(float),
            "ask": last["yes_ask_close"].to_numpy(float),
            "age_s": (at_ts - last["end_period_ts"]).to_numpy(float),
        }
    )
    # markets that traded but never showed a quote still get a VWAP row
    extra = sorted(set(vol.index) - set(out["ticker"]))
    if extra:
        out = pd.concat(
            [out, pd.DataFrame({"ticker": extra, "bid": np.nan, "ask": np.nan, "age_s": np.nan})],
            ignore_index=True,
        )
    out["volume"] = out["ticker"].map(vol).fillna(0.0)
    out["vwap"] = out["ticker"].map(vwap)
    out["vwap_age_s"] = out["ticker"].map(vwap_age)
    return out


def game_strength(
    game: pd.Series,
    markets: pd.DataFrame,
    candles: dict[str, pd.DataFrame],
    kickoff_ts: int,
    cfg: ladders.LadderConfig = ladders.LadderConfig(),
) -> dict:
    """Implied pregame home margin and total for one game.

    `markets`: this game's markets (series_ticker, ticker, team, floor_strike).
    `candles`: series_ticker -> 1-minute candles for that series' markets.
    """
    snap = {
        s: snapshot_from_candles(candles.get(s, pd.DataFrame()), kickoff_ts)
        for s in ("KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL")
    }
    meta = markets.set_index("ticker")[["series_ticker", "team", "floor_strike"]]
    spread_q = snap["KXNFLSPREAD"].join(meta, on="ticker")
    total_q = snap["KXNFLTOTAL"].join(meta, on="ticker")
    m_fit = ladders.fit_ladder(ladders.margin_quotes(spread_q, game["home_team"]), "margin", cfg)
    t_fit = ladders.fit_ladder(ladders.total_quotes(total_q), "total", cfg)

    win = snap["KXNFLGAME"].join(meta, on="ticker")
    home_win = _home_win_price(win, game["home_team"], game["away_team"])
    # The pregame winner price implies a mean margin under a normal margin distribution.
    # It replaces a spread ladder that can't be fit, and is averaged with a weak one.
    implied = None if home_win is None else float(MARGIN_SD * norm.ppf(np.clip(home_win, 0.01, 0.99)))
    spread_source = "ladder" if m_fit.mean is not None else ("winner_price" if implied is not None else "none")
    if m_fit.mean is None and implied is not None:
        m_fit.mean = m_fit.median = implied
        m_fit.sd, m_fit.model = MARGIN_SD, "winner_price"
    elif m_fit.mean is not None and implied is not None and m_fit.quality < WEAK_LADDER_QUALITY:
        m_fit.mean = m_fit.median = (m_fit.mean + implied) / 2
        spread_source = "ladder+winner_price"
    return {
        "kalshi_spread_source": spread_source,
        "kalshi_total_source": "ladder" if t_fit.mean is not None else "none",
        "game_id": game["game_id"],
        "kickoff_ts": kickoff_ts,
        "kalshi_spread_line": m_fit.mean,
        "kalshi_spread_median": m_fit.median,
        "kalshi_margin_sd": m_fit.sd,
        "kalshi_spread_model": m_fit.model,
        "kalshi_spread_rmse": min(filter(None, [m_fit.rmse_normal, m_fit.rmse_skew]), default=None),
        "kalshi_spread_n_used": m_fit.n_used,
        "kalshi_spread_quality": m_fit.quality,
        "kalshi_spread_quote_age_s": m_fit.quote_age_s,
        "kalshi_total_line": t_fit.mean,
        "kalshi_total_median": t_fit.median,
        "kalshi_total_sd": t_fit.sd,
        "kalshi_total_model": t_fit.model,
        "kalshi_total_rmse": min(filter(None, [t_fit.rmse_normal, t_fit.rmse_skew]), default=None),
        "kalshi_total_n_used": t_fit.n_used,
        "kalshi_total_quality": t_fit.quality,
        "kalshi_total_quote_age_s": t_fit.quote_age_s,
        "kalshi_home_win_pregame": home_win,
        "winner_vs_ladder_gap": ladders.winner_consistency(m_fit, home_win),
    }


def _home_win_price(win: pd.DataFrame, home: str, away: str) -> float | None:
    mids = {}
    for r in win.itertuples():
        if r.bid > 0 and r.ask < 1 and r.ask > r.bid:
            mids[r.team] = (r.bid + r.ask) / 2
    vals = []
    if home in mids:
        vals.append(mids[home])
    if away in mids:
        vals.append(1 - mids[away])
    return float(np.mean(vals)) if vals else None


def with_kalshi_lines(games: pd.DataFrame, strength: pd.DataFrame) -> pd.DataFrame:
    """Replace the pregame betting lines in a games table (prep.build_games output)
    with the Kalshi pregame implied spread and total. Same meaning: expected final
    home margin and expected final total, from 0-0 over a full game. Games without a
    usable Kalshi value get NaN, never the betting line."""
    s = strength.set_index("game_id")
    out = games.copy()
    out["spread_line"] = out["game_id"].map(s["kalshi_spread_line"])
    out["total_line"] = out["game_id"].map(s["kalshi_total_line"])
    out["home_total"] = (out["total_line"] + out["spread_line"]) / 2
    out["away_total"] = (out["total_line"] - out["spread_line"]) / 2
    return out


def fill_missing_totals(strength: pd.DataFrame, week: pd.Series) -> pd.DataFrame:
    """Kalshi-only fallback for games whose total ladder can't be fit: the median Kalshi
    pregame total of that week's other games (source 'week_median')."""
    out = strength.copy()
    wk = out["game_id"].map(week)
    med = out.groupby(wk)["kalshi_total_line"].transform("median")
    miss = out["kalshi_total_line"].isna() & med.notna()
    out.loc[miss, "kalshi_total_line"] = med[miss]
    out.loc[miss, "kalshi_total_source"] = "week_median"
    return out

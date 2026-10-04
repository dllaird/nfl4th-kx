"""Spread / total ladders -> implied margin and total distributions.

Each ladder strike is a binary contract "X wins by over k" or "Over k points". A
snapshot gives, per strike, P(margin > k) (or P(total > k)). This module:
  1. turns quotes into survival probabilities on one axis (home margin, or total),
     dropping empty, wide and stale strikes and weighting the rest by liquidity;
  2. enforces monotonicity across strikes with weighted isotonic regression;
  3. fits a normal and a skew-normal survival curve by weighted least squares;
  4. reports implied mean/median, fit quality and quote age.

Margins are always home - away, matching the sign of nflverse `spread_line`
(positive = home favored).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats


@dataclass(frozen=True)
class LadderConfig:
    max_spread: float = 0.10  # drop strikes whose bid/ask is wider than this
    max_age_s: float = 1800.0  # drop strikes whose last quote is older than this
    min_points: int = 2  # need at least this many usable strikes to fit
    # weak prior on the spread of the distribution: thin ladders cluster strikes near
    # the line, where sd is barely identified. Typical NFL final-margin / total sd.
    sd_prior: dict = field(default_factory=lambda: {"margin": 13.5, "total": 13.0})
    sd_prior_strength: float = 1e-3
    p_clip: float = 0.02  # ignore strikes pinned near 0/1: they carry no shape info
    min_vwap_volume: float = 100.0  # contracts traded in the window to trust a VWAP


@dataclass
class LadderFit:
    kind: str  # "margin" or "total"
    n_strikes: int
    n_used: int
    mean: float | None = None
    median: float | None = None
    sd: float | None = None
    skew_alpha: float | None = None
    skew_loc: float | None = None
    skew_scale: float | None = None
    rmse_normal: float | None = None
    rmse_skew: float | None = None
    quote_age_s: float | None = None
    quality: float = 0.0
    model: str = "none"
    points: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


def isotonic_decreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators for a non-increasing fit (x already sorted)."""
    vals, wts, sizes = [], [], []
    for yi, wi in zip(-np.asarray(y, float), np.asarray(w, float)):
        vals.append(yi)
        wts.append(wi)
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, s2 = vals.pop(), wts.pop(), sizes.pop()
            v1, w1, s1 = vals.pop(), wts.pop(), sizes.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
            sizes.append(s1 + s2)
    return -np.repeat(vals, sizes)


def strike_points(quotes: pd.DataFrame, cfg: LadderConfig = LadderConfig()) -> pd.DataFrame:
    """Usable ladder points from a snapshot.

    `quotes` columns: x (strike on the common axis), side (+1 if the contract pays when
    the value exceeds x, -1 if it pays when the value is below x), bid, ask (YES, 0-1),
    age_s (seconds since the quote), volume (contracts traded in the window) and,
    optionally, vwap / vwap_age_s (volume-weighted trade price and its age).

    Price per strike: the quote mid when two-sided, fresh and tighter than
    `max_spread`; otherwise the VWAP if at least `min_vwap_volume` traded. Returns x,
    p (P(value > x)), source, weight, spread, age_s, used, p_mono.
    """
    q = quotes.copy()
    for col in ("vwap", "vwap_age_s"):
        if col not in q:
            q[col] = np.nan
    two_sided = q["bid"].gt(0) & q["ask"].lt(1) & q["ask"].gt(q["bid"])
    q["spread"] = (q["ask"] - q["bid"]).where(two_sided)
    quote_ok = two_sided & q["spread"].le(cfg.max_spread) & q["age_s"].le(cfg.max_age_s)
    vwap_ok = ~quote_ok & q["vwap"].notna() & q["volume"].fillna(0).ge(cfg.min_vwap_volume)
    yes_p = np.where(quote_ok, (q["bid"] + q["ask"]) / 2, np.where(vwap_ok, q["vwap"], np.nan))
    q["source"] = np.where(quote_ok, "quote", np.where(vwap_ok, "vwap", "none"))
    q["age_s"] = np.where(vwap_ok, q["vwap_age_s"], q["age_s"])
    q["p"] = np.where(q["side"] > 0, yes_p, 1 - yes_p)
    q["used"] = (quote_ok | vwap_ok) & q["p"].between(cfg.p_clip, 1 - cfg.p_clip)
    # liquidity weight: tighter and more traded strikes count more; a VWAP counts like
    # a quote with a 5-cent spread
    eff_spread = np.where(quote_ok, q["spread"], 0.05)
    q["weight"] = np.where(
        q["used"], np.sqrt(1 + q["volume"].fillna(0)) / np.maximum(eff_spread, 0.01), 0.0
    )
    q = q.sort_values("x").reset_index(drop=True)
    used = q["used"].to_numpy()
    q["p_mono"] = np.nan
    if used.sum() >= 2:
        q.loc[used, "p_mono"] = isotonic_decreasing(q.loc[used, "p"], q.loc[used, "weight"])
    return q


def _fit_normal(x, p, w, sd0, lam):
    def loss(theta):
        mu, log_sd = theta
        fit = np.sum(w * (stats.norm.sf(x, mu, np.exp(log_sd)) - p) ** 2)
        return fit + lam * (log_sd - np.log(sd0)) ** 2

    mu0 = float(np.interp(0.5, p[::-1], x[::-1]))
    res = optimize.minimize(loss, [mu0, np.log(sd0)], method="Nelder-Mead")
    return res.x[0], float(np.exp(res.x[1]))


def _fit_skew(x, p, w, mu0, sd0, lam=0.0):
    def loss(theta):
        loc, log_scale, a = theta
        fit = np.sum(w * (stats.skewnorm.sf(x, a, loc, np.exp(log_scale)) - p) ** 2)
        return fit + lam * (log_scale - np.log(sd0)) ** 2

    res = optimize.minimize(loss, [mu0, np.log(sd0), 0.0], method="Nelder-Mead")
    loc, scale, a = res.x[0], float(np.exp(res.x[1])), res.x[2]
    return loc, scale, a


def fit_ladder(quotes: pd.DataFrame, kind: str, cfg: LadderConfig = LadderConfig()) -> LadderFit:
    pts = strike_points(quotes, cfg)
    used = pts[pts["used"]]
    fit = LadderFit(kind=kind, n_strikes=len(pts), n_used=len(used), points=pts)
    if len(used) < cfg.min_points:
        return fit
    x = used["x"].to_numpy(float)
    p = used["p_mono"].to_numpy(float)
    w = used["weight"].to_numpy(float)
    w = w / w.sum()

    sd0, lam = cfg.sd_prior[kind], cfg.sd_prior_strength
    mu, sd = _fit_normal(x, p, w, sd0, lam)
    rmse_n = float(np.sqrt(np.sum(w * (stats.norm.sf(x, mu, sd) - p) ** 2)))
    loc, scale, a = _fit_skew(x, p, w, mu, sd, lam)
    sk = stats.skewnorm(a, loc, scale)
    rmse_s = float(np.sqrt(np.sum(w * (sk.sf(x) - p) ** 2)))

    fit.sd, fit.rmse_normal, fit.rmse_skew = sd, rmse_n, rmse_s
    fit.skew_alpha, fit.skew_loc, fit.skew_scale = float(a), float(loc), scale
    # prefer the 2-parameter normal unless skew clearly helps (guards against overfit
    # on short ladders)
    if len(used) >= 6 and rmse_s < 0.8 * rmse_n:
        fit.model, fit.mean, fit.median = "skewnorm", float(sk.mean()), float(sk.median())
    else:
        fit.model, fit.mean, fit.median = "normal", float(mu), float(mu)
    fit.quote_age_s = float(np.average(used["age_s"], weights=used["weight"]))
    fit.quality = ladder_quality(fit, cfg)
    return fit


def ladder_quality(fit: LadderFit, cfg: LadderConfig) -> float:
    """0-1 score: number of usable strikes, fit error and quote freshness."""
    if fit.mean is None:
        return 0.0
    rmse = min(fit.rmse_normal or 1, fit.rmse_skew or 1)
    s_n = min(1.0, fit.n_used / 8)
    s_fit = max(0.0, 1 - rmse / 0.05)
    s_age = max(0.0, 1 - (fit.quote_age_s or 0) / cfg.max_age_s)
    return float(s_n * s_fit * (0.5 + 0.5 * s_age))


def margin_quotes(markets: pd.DataFrame, home_team: str) -> pd.DataFrame:
    """Spread-ladder rows -> common home-margin axis.

    "HOME wins by over k" pays when margin > k  -> x = k,  side +1
    "AWAY wins by over k" pays when margin < -k -> x = -k, side -1
    """
    is_home = markets["team"] == home_team
    return markets.assign(
        x=np.where(is_home, markets["floor_strike"], -markets["floor_strike"]),
        side=np.where(is_home, 1, -1),
    )


def total_quotes(markets: pd.DataFrame) -> pd.DataFrame:
    return markets.assign(x=markets["floor_strike"], side=1)


def winner_consistency(margin_fit: LadderFit, home_win_price: float | None) -> float | None:
    """Winner price minus the ladder-implied P(home wins) (+ half a tie, which the
    continuous fit splits evenly around 0). Large gaps flag stale or thin markets."""
    if margin_fit.mean is None or home_win_price is None:
        return None
    if margin_fit.model == "skewnorm":
        p = stats.skewnorm.sf(0, margin_fit.skew_alpha, margin_fit.skew_loc, margin_fit.skew_scale)
    else:
        p = stats.norm.sf(0, margin_fit.mean, margin_fit.sd)
    return float(home_win_price - p)


"""Kicker-specific field goal make probability.

A logistic model with partial pooling across kickers:

    logit P(make) = f(distance) + roof/weather/era terms + is_xp
                    + u[kicker] + v[kicker] * (distance - 40) / 10

* f is a restricted (natural) cubic spline in kick distance. It is linear in the logit
  beyond the last knot, so make probability keeps falling smoothly for distances longer
  than anything attempted; there is no hard cutoff.
* u (accuracy) and v (how fast a kicker's accuracy falls with distance, i.e. leg
  strength) are per-kicker effects with Gaussian priors, fit by maximum a posteriori.
  Kickers with few attempts are shrunk toward the league curve; an unknown kicker gets
  the league curve.
* Extra points enter as attempts at their distance with their own intercept shift, which
  adds information about each kicker's accuracy.

This replaces nfl4th's roof/era GAM in the market decision engine only. The nfl4th port
(`field_goal.py`) and the benchmark keep nfl4th's model for parity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

DEFAULT_KNOTS = (22.0, 33.0, 42.0, 49.0, 55.0, 62.0)
MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "kicker_model.json"


def rcs_basis(x: np.ndarray, knots: tuple[float, ...]) -> np.ndarray:
    """Restricted cubic spline basis (Harrell): x plus k-2 nonlinear terms; linear beyond
    the outer knots."""
    x = np.asarray(x, float)
    t = np.asarray(knots, float)
    k = len(t)
    scale = (t[-1] - t[0]) ** 2
    cols = [x]
    pos = lambda z: np.maximum(z, 0.0) ** 3  # noqa: E731
    for j in range(k - 2):
        term = (pos(x - t[j])
                - pos(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                + pos(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]))
        cols.append(term / scale)
    return np.column_stack(cols)


FIXED_NAMES = ["intercept", "outdoors", "wind_mph_outdoors", "cold_lt40f", "season_trend", "is_xp"]


def design(df: pd.DataFrame, knots: tuple[float, ...]) -> tuple[np.ndarray, list[str]]:
    d = df["kick_distance"].to_numpy(float)
    outdoors = df["outdoors"].to_numpy(float)
    wind = np.nan_to_num(df.get("wind", pd.Series(0.0, index=df.index)).to_numpy(float), nan=0.0)
    temp = df.get("temp", pd.Series(np.nan, index=df.index)).to_numpy(float)
    cold = np.where(np.isnan(temp), 0.0, (temp < 40).astype(float)) * outdoors
    trend = (df["season"].to_numpy(float) - 2020.0) / 5.0
    xp = df.get("is_xp", pd.Series(0, index=df.index)).to_numpy(float)
    spline = rcs_basis(d / 10.0, tuple(k / 10.0 for k in knots))
    x = np.column_stack([np.ones(len(df)), outdoors, wind * outdoors / 10.0, cold, trend, xp, spline])
    names = FIXED_NAMES + [f"dist_rcs{i}" for i in range(spline.shape[1])]
    return x, names


@dataclass
class KickerFGModel:
    knots: tuple[float, ...] = DEFAULT_KNOTS
    sd_accuracy: float = 0.35  # prior sd of the kicker intercept (logit)
    sd_range: float = 0.15  # prior sd of the kicker slope per 10 yards (logit)
    beta: np.ndarray | None = None
    names: list[str] = field(default_factory=list)
    kickers: pd.DataFrame | None = None  # kicker_player_id -> u, v, name, attempts

    def fit(self, kicks: pd.DataFrame) -> KickerFGModel:
        x, self.names = design(kicks, self.knots)
        y = kicks["made"].to_numpy(float)
        ids, uniq = pd.factorize(kicks["kicker_player_id"])
        k, p = len(uniq), x.shape[1]
        z = (kicks["kick_distance"].to_numpy(float) - 40.0) / 10.0

        def nlp(theta):
            b, u, v = theta[:p], theta[p:p + k], theta[p + k:]
            eta = x @ b + u[ids] + v[ids] * z
            mu = 1 / (1 + np.exp(-eta))
            ll = np.sum(y * eta - np.logaddexp(0, eta))
            pen = 0.5 * np.sum(u**2) / self.sd_accuracy**2 + 0.5 * np.sum(v**2) / self.sd_range**2
            pen += 0.5 * np.sum(b[1:] ** 2) / 10.0**2  # weak prior on fixed effects
            r = mu - y
            gb = x.T @ r
            gb[1:] += b[1:] / 10.0**2
            gu = np.bincount(ids, r, k) + u / self.sd_accuracy**2
            gv = np.bincount(ids, r * z, k) + v / self.sd_range**2
            return -ll + pen, np.concatenate([gb, gu, gv])

        theta0 = np.zeros(p + 2 * k)
        theta0[0] = np.log(y.mean() / (1 - y.mean()))
        res = optimize.minimize(nlp, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 5000})
        if not res.success:
            raise RuntimeError(f"kicker model did not converge: {res.message}")
        self.beta = res.x[:p]
        names = kicks.groupby("kicker_player_id")["kicker_player_name"].agg(lambda s: s.mode().iat[0])
        fg = kicks[kicks["is_xp"] == 0].groupby("kicker_player_id").agg(
            fg_attempts=("made", "size"), last_season=("season", "max"))
        self.kickers = pd.DataFrame({"u": res.x[p:p + k], "v": res.x[p + k:]}, index=pd.Index(uniq, name="kicker_player_id"))
        self.kickers = self.kickers.join(names.rename("name")).join(fg).fillna({"fg_attempts": 0})
        return self

    def predict(self, kick_distance, kicker_player_id=None, outdoors=0, season=2025, wind=0.0, temp=np.nan, is_xp=0) -> np.ndarray:
        """Make probability. Scalars broadcast. Unknown or missing kickers get the league
        curve (u = v = 0)."""
        d = np.atleast_1d(np.asarray(kick_distance, float))
        n = len(d)
        bc = lambda v: np.broadcast_to(np.asarray(v, dtype=object if isinstance(v, str) else None), (n,))  # noqa: E731
        frame = pd.DataFrame({
            "kick_distance": d, "outdoors": bc(outdoors).astype(float), "season": bc(season).astype(float),
            "wind": bc(wind).astype(float), "temp": bc(temp).astype(float), "is_xp": bc(is_xp).astype(float)})
        x, _ = design(frame, self.knots)
        eta = x @ self.beta
        if kicker_player_id is not None and self.kickers is not None:
            kid = pd.Series(np.broadcast_to(np.asarray(kicker_player_id, dtype=object), (n,)))
            u = kid.map(self.kickers["u"]).fillna(0.0).to_numpy()
            v = kid.map(self.kickers["v"]).fillna(0.0).to_numpy()
            eta = eta + u + v * (d - 40.0) / 10.0
        return 1 / (1 + np.exp(-eta))

    def save(self, path: Path = MODEL_PATH) -> None:
        payload = {
            "knots": list(self.knots), "sd_accuracy": self.sd_accuracy, "sd_range": self.sd_range,
            "names": self.names, "beta": self.beta.tolist(),
            "kickers": self.kickers.reset_index().to_dict(orient="list"),
        }
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> KickerFGModel:
        p = json.loads(Path(path).read_text())
        m = cls(knots=tuple(p["knots"]), sd_accuracy=p["sd_accuracy"], sd_range=p["sd_range"])
        m.beta, m.names = np.asarray(p["beta"]), p["names"]
        m.kickers = pd.DataFrame(p["kickers"]).set_index("kicker_player_id")
        return m


def prepare_kicks(kicks: pd.DataFrame) -> pd.DataFrame:
    """nflfastR kick rows -> model frame (outdoors flag from roof)."""
    k = kicks.copy()
    k["outdoors"] = (k["roof"] == "outdoors").astype(float)
    return k


def team_kickers(kicks: pd.DataFrame) -> pd.DataFrame:
    """(game_id, posteam) -> the kicker who took most of the team's kicks in that game,
    with each team's previous-game kicker as a fallback for games without a kick.
    `kicks` needs game_id, season, week, posteam, kicker_player_id."""
    per_game = (kicks.groupby(["season", "week", "game_id", "posteam"])["kicker_player_id"]
                .agg(lambda s: s.mode().iat[0]).reset_index())
    return per_game.sort_values(["posteam", "season", "week"]).reset_index(drop=True)


def kicker_for(lookup: pd.DataFrame, game_id: str, posteam: str, season: int, week: int) -> str | None:
    """The team's kicker in this game, else its most recent kicker before it."""
    t = lookup[lookup["posteam"] == posteam]
    same = t[t["game_id"] == game_id]
    if len(same):
        return same["kicker_player_id"].iat[0]
    before = t[(t["season"] < season) | ((t["season"] == season) & (t["week"] < week))]
    return before["kicker_player_id"].iat[-1] if len(before) else None

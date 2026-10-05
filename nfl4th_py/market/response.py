"""Market-response models: value post-play states with Kalshi winner prices.

All quantities are from the **pre-play possession team's** view ("team"). A transition
frame has pre-play columns and `post_*` columns describing the resulting state:

  pre : pre_logit, qtr, game_seconds_remaining, half_seconds_remaining, down, ydstogo,
        yardline_100, score_differential, posteam_timeouts_remaining,
        defteam_timeouts_remaining, receive_2h_ko, kind, kalshi_posteam_spread,
        kalshi_total_line
  post: post_has_ball, post_kind ("scrimmage" | "pat" | "kickoff"), post_fp (yards from
        the team's scoring end zone: yardline_100 if the team has the ball, else 100 -
        yardline_100), post_score_diff (team - opponent), post_down, post_ydstogo,
        post_qtr, post_game_seconds_remaining, post_half_seconds_remaining,
        post_team_timeouts, post_opp_timeouts

Two formulations:
  TransitionModel  (a) logit(post price) = pre_logit + f(pre, post, deltas)
  ValueSurface     (b) V(state, strength) fit to observed prices; counterfactual value
                   = V(post) + (pre_logit - V(pre))  (re-anchored to the live price)

Monotone constraints guarantee, all else equal: better field position (lower post_fp),
a bigger lead, and having the ball never lower the predicted price.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial import cKDTree

EPS = 1e-3


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


# --------------------------------------------------------------------------------------
# building transition frames from the Phase 4 dataset


def transitions_from_dataset(d: pd.DataFrame) -> pd.DataFrame:
    """Phase 4 rows -> transition frame (team = pre-play posteam)."""
    has = d["post_pre_pos_has_ball"].astype(bool)
    t = pd.DataFrame(index=d.index)
    t["pre_logit"] = d["pre_logit"]
    for c in [
        "qtr", "game_seconds_remaining", "half_seconds_remaining", "down", "ydstogo",
        "yardline_100", "score_differential", "posteam_timeouts_remaining",
        "defteam_timeouts_remaining", "receive_2h_ko", "kind", "kalshi_posteam_spread",
        "kalshi_total_line",
    ]:
        t[c] = d[c]
    t["post_has_ball"] = has.astype(int)
    t["post_kind"] = d["post_kind"]
    t["post_fp"] = np.where(has, d["post_yardline_100"], 100 - d["post_yardline_100"])
    t["post_score_diff"] = d["post_score_differential_pre_view"]
    t["post_down"] = d["post_down"]
    t["post_ydstogo"] = d["post_ydstogo"]
    t["post_qtr"] = d["post_qtr"]
    t["post_game_seconds_remaining"] = d["post_game_seconds_remaining"]
    t["post_half_seconds_remaining"] = d["post_half_seconds_remaining"]
    t["post_team_timeouts"] = np.where(has, d["post_posteam_timeouts_remaining"], d["post_defteam_timeouts_remaining"])
    t["post_opp_timeouts"] = np.where(has, d["post_defteam_timeouts_remaining"], d["post_posteam_timeouts_remaining"])
    return t


def states_from_dataset(d: pd.DataFrame) -> pd.DataFrame:
    """Phase 4 rows -> observed (state, price) pairs for the value surface, from the
    posteam's view at the snap."""
    s = pd.DataFrame(index=d.index)
    s["logit"] = d["pre_logit"]
    s["has_ball"] = 1
    s["kind"] = d["kind"]
    s["fp"] = d["yardline_100"]
    s["score_diff"] = d["score_differential"]
    for a, b in [
        ("down", "down"), ("ydstogo", "ydstogo"), ("qtr", "qtr"),
        ("game_seconds_remaining", "game_seconds_remaining"),
        ("half_seconds_remaining", "half_seconds_remaining"),
        ("team_timeouts", "posteam_timeouts_remaining"), ("opp_timeouts", "defteam_timeouts_remaining"),
        ("team_spread", "kalshi_posteam_spread"), ("total", "kalshi_total_line"),
    ]:
        s[a] = d[b]
    return s


# --------------------------------------------------------------------------------------
# (a) transition model

PRE_FEATURES = [
    "pre_logit", "qtr", "game_seconds_remaining", "half_seconds_remaining", "down", "ydstogo",
    "yardline_100", "score_differential", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "receive_2h_ko", "pre_kind_pat", "pre_kind_kickoff",
    "kalshi_posteam_spread", "kalshi_total_line",
]
POST_FEATURES = [
    "post_has_ball", "post_kind_pat", "post_kind_kickoff", "post_fp", "post_score_diff",
    "post_down", "post_ydstogo", "post_qtr", "post_game_seconds_remaining",
    "post_half_seconds_remaining", "post_team_timeouts", "post_opp_timeouts",
]
DELTA_FEATURES = ["d_score", "d_fp", "d_seconds"]
TRANSITION_FEATURES = PRE_FEATURES + POST_FEATURES + DELTA_FEATURES
TRANSITION_MONOTONE = {"post_has_ball": 1, "post_fp": -1, "post_score_diff": 1, "d_score": 1, "d_fp": -1}


def transition_matrix(t: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=t.index)
    for c in PRE_FEATURES:
        if c == "pre_kind_pat":
            x[c] = (t["kind"] == "pat").astype(float)
        elif c == "pre_kind_kickoff":
            x[c] = (t["kind"] == "kickoff").astype(float)
        else:
            x[c] = t[c].astype(float)
    for c in POST_FEATURES:
        if c == "post_kind_pat":
            x[c] = (t["post_kind"] == "pat").astype(float)
        elif c == "post_kind_kickoff":
            x[c] = (t["post_kind"] == "kickoff").astype(float)
        else:
            x[c] = t[c].astype(float)
    x["d_score"] = x["post_score_diff"] - x["score_differential"]
    # pre-play field position from the team's view is yardline_100 (team has the ball)
    x["d_fp"] = x["post_fp"] - x["yardline_100"]
    x["d_seconds"] = x["post_game_seconds_remaining"] - x["game_seconds_remaining"]
    return x[TRANSITION_FEATURES]


def _monotone_tuple(features, spec):
    return "(" + ",".join(str(spec.get(f, 0)) for f in features) + ")"


DEFAULT_PARAMS = {
    "objective": "reg:squarederror",
    "eta": 0.03,
    "max_depth": 6,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 5.0,
    "tree_method": "hist",
    "seed": 2025,
}


@dataclass
class Support:
    """Training-data support: the mean distance to the k nearest training transitions in
    a standardized key-feature space, expressed as a percentile of the same distance
    for training transitions themselves. score = share of training transitions that are
    *less* typical (farther from their neighbors) than this point: 0.5 = a typical
    training point, 0.02 = more isolated than 98% of the training data."""

    columns: list[str]
    mean: np.ndarray
    scale: np.ndarray
    tree: cKDTree
    k: int = 25
    ref_quantiles: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))

    @classmethod
    def fit(cls, x: pd.DataFrame, columns: list[str], k: int = 25) -> Support:
        z = x[columns].to_numpy(float)
        mean, scale = np.nanmean(z, 0), np.nanstd(z, 0) + 1e-9
        zz = np.nan_to_num((z - mean) / scale)
        sup = cls(columns, mean, scale, cKDTree(zz), k)
        sup.ref_quantiles = np.quantile(sup._dist(x, exclude_self=True), np.linspace(0, 1, 1001))
        return sup

    def _dist(self, x: pd.DataFrame, exclude_self: bool = False) -> np.ndarray:
        z = np.nan_to_num((x[self.columns].to_numpy(float) - self.mean) / self.scale)
        k = self.k + (1 if exclude_self else 0)
        d, _ = self.tree.query(z, k=k)
        return d[:, 1:].mean(1) if exclude_self else d.mean(1)

    def score(self, x: pd.DataFrame) -> np.ndarray:
        q = np.searchsorted(self.ref_quantiles, self._dist(x), side="left") / (len(self.ref_quantiles) - 1)
        return 1 - np.clip(q, 0, 1)


SUPPORT_COLUMNS = [
    "pre_logit", "game_seconds_remaining", "score_differential", "post_has_ball", "post_fp",
    "post_score_diff", "post_kind_pat", "post_kind_kickoff", "post_game_seconds_remaining",
]


@dataclass
class TransitionModel:
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    booster: xgb.Booster | None = None
    sd_booster: xgb.Booster | None = None
    support: Support | None = None
    # multiplies the sd model so that +-1.645 sd covers 90% of out-of-fold residuals
    sd_scale: float = 1.0

    def fit(self, t: pd.DataFrame, y_logit: np.ndarray, t_val: pd.DataFrame, y_val: np.ndarray, rounds: int = 3000) -> TransitionModel:
        x, xv = transition_matrix(t), transition_matrix(t_val)
        params = dict(self.params, monotone_constraints=_monotone_tuple(TRANSITION_FEATURES, TRANSITION_MONOTONE))
        dtr = xgb.DMatrix(x, label=y_logit, base_margin=x["pre_logit"].to_numpy())
        dva = xgb.DMatrix(xv, label=y_val, base_margin=xv["pre_logit"].to_numpy())
        self.booster = xgb.train(params, dtr, rounds, evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
        # heteroscedastic uncertainty: model |residual| on validation rows (out of sample)
        resid = np.abs(y_val - self.predict_logit(t_val))
        sd_params = dict(DEFAULT_PARAMS, objective="reg:absoluteerror", max_depth=4, min_child_weight=50)
        self.sd_booster = xgb.train(sd_params, xgb.DMatrix(xv, label=resid), 300)
        self.support = Support.fit(x, SUPPORT_COLUMNS)
        return self

    def predict_logit(self, t: pd.DataFrame) -> np.ndarray:
        x = transition_matrix(t)
        d = xgb.DMatrix(x, base_margin=x["pre_logit"].to_numpy())
        return self.booster.predict(d, iteration_range=(0, self.booster.best_iteration + 1))

    def predict(self, t: pd.DataFrame) -> pd.DataFrame:
        """Predicted post-play price, logit-scale sd (MAE-based, x1.25 ~ normal sd) and
        support score in [0, 1]."""
        x = transition_matrix(t)
        mu = self.predict_logit(t)
        sd = self.sd_scale * 1.25 * np.maximum(self.sd_booster.predict(xgb.DMatrix(x)), 1e-3)
        return pd.DataFrame(
            {"logit": mu, "price": expit(mu), "sd_logit": sd, "support": self.support.score(x)},
            index=t.index,
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(directory / "transition.json")
        self.sd_booster.save_model(directory / "transition_sd.json")
        np.savez(
            directory / "transition_support.npz",
            data=self.support.tree.data, mean=self.support.mean, scale=self.support.scale,
            k=self.support.k, ref_quantiles=self.support.ref_quantiles,
        )
        (directory / "transition_meta.json").write_text(
            json.dumps({"features": TRANSITION_FEATURES, "support_columns": SUPPORT_COLUMNS,
                        "best_iteration": self.booster.best_iteration, "params": self.params,
                        "sd_scale": self.sd_scale}, indent=1)
        )

    @classmethod
    def load(cls, directory: Path) -> TransitionModel:
        m = cls()
        meta = json.loads((directory / "transition_meta.json").read_text())
        if meta["features"] != TRANSITION_FEATURES:
            raise ValueError("saved transition model has a different feature list")
        m.booster = xgb.Booster()
        m.booster.load_model(directory / "transition.json")
        m.booster.set_attr(best_iteration=str(meta["best_iteration"]))
        m.sd_booster = xgb.Booster()
        m.sd_booster.load_model(directory / "transition_sd.json")
        s = np.load(directory / "transition_support.npz")
        m.sd_scale = float(meta.get("sd_scale", 1.0))
        m.support = Support(meta["support_columns"], s["mean"], s["scale"], cKDTree(s["data"]), int(s["k"]), s["ref_quantiles"])
        return m


# --------------------------------------------------------------------------------------
# (b) value surface

VALUE_FEATURES = [
    "kind_pat", "kind_kickoff", "fp", "score_diff", "down", "ydstogo", "qtr",
    "game_seconds_remaining", "half_seconds_remaining", "team_timeouts", "opp_timeouts",
    "team_spread", "total",
]
VALUE_MONOTONE = {"fp": -1, "score_diff": 1, "team_spread": 1}


def value_matrix(s: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=s.index)
    x["kind_pat"] = (s["kind"] == "pat").astype(float)
    x["kind_kickoff"] = (s["kind"] == "kickoff").astype(float)
    for c in VALUE_FEATURES[2:]:
        x[c] = s[c].astype(float)
    return x[VALUE_FEATURES]


def post_states(t: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Post-play states from the side of whoever has the ball, plus a sign: +1 if the
    team has it (V is the team's value), -1 if the opponent has it (team = 1 - V)."""
    has = t["post_has_ball"].astype(bool).to_numpy()
    s = pd.DataFrame(index=t.index)
    s["kind"] = t["post_kind"]
    s["fp"] = np.where(has, t["post_fp"], 100 - t["post_fp"])
    s["score_diff"] = np.where(has, t["post_score_diff"], -t["post_score_diff"])
    s["down"], s["ydstogo"], s["qtr"] = t["post_down"], t["post_ydstogo"], t["post_qtr"]
    s["game_seconds_remaining"] = t["post_game_seconds_remaining"]
    s["half_seconds_remaining"] = t["post_half_seconds_remaining"]
    s["team_timeouts"] = np.where(has, t["post_team_timeouts"], t["post_opp_timeouts"])
    s["opp_timeouts"] = np.where(has, t["post_opp_timeouts"], t["post_team_timeouts"])
    s["team_spread"] = np.where(has, t["kalshi_posteam_spread"], -t["kalshi_posteam_spread"])
    s["total"] = t["kalshi_total_line"]
    return s, np.where(has, 1.0, -1.0)


def pre_states(t: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame(index=t.index)
    s["kind"] = t["kind"]
    s["fp"] = t["yardline_100"]
    s["score_diff"] = t["score_differential"]
    s["down"], s["ydstogo"], s["qtr"] = t["down"], t["ydstogo"], t["qtr"]
    s["game_seconds_remaining"] = t["game_seconds_remaining"]
    s["half_seconds_remaining"] = t["half_seconds_remaining"]
    s["team_timeouts"] = t["posteam_timeouts_remaining"]
    s["opp_timeouts"] = t["defteam_timeouts_remaining"]
    s["team_spread"] = t["kalshi_posteam_spread"]
    s["total"] = t["kalshi_total_line"]
    return s


@dataclass
class ValueSurface:
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    booster: xgb.Booster | None = None

    def fit(self, s: pd.DataFrame, s_val: pd.DataFrame, rounds: int = 4000) -> ValueSurface:
        params = dict(self.params, monotone_constraints=_monotone_tuple(VALUE_FEATURES, VALUE_MONOTONE))
        dtr = xgb.DMatrix(value_matrix(s), label=s["logit"].to_numpy())
        dva = xgb.DMatrix(value_matrix(s_val), label=s_val["logit"].to_numpy())
        self.booster = xgb.train(params, dtr, rounds, evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
        return self

    def value(self, s: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(
            xgb.DMatrix(value_matrix(s)), iteration_range=(0, self.booster.best_iteration + 1)
        )

    def predict_logit(self, t: pd.DataFrame) -> np.ndarray:
        """Counterfactual post logit re-anchored to the live pre-play price."""
        post, sign = post_states(t)
        offset = t["pre_logit"].to_numpy() - self.value(pre_states(t))
        # the offset is the market's game-specific adjustment for the team; apply it
        # with the same sign as the team's value
        return sign * self.value(post) + offset

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(directory / "value_surface.json")
        (directory / "value_surface_meta.json").write_text(
            json.dumps({"features": VALUE_FEATURES, "best_iteration": self.booster.best_iteration}, indent=1)
        )

"""Loading of model artifacts exported by scripts/export_from_r.R."""

from __future__ import annotations

import gzip
import hashlib
import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path(__file__).parent / "data"


@cache
def manifest() -> dict:
    return json.loads((DATA_DIR / "MANIFEST.json").read_text())


def model_features(name: str) -> list[str]:
    return list(manifest()["models"][name]["features"])


@cache
def booster(name: str) -> xgb.Booster:
    """Load an exported xgboost model. The R boosters carry no feature names, so
    the manifest's positional feature order is attached here and enforced by
    `predict()`."""
    spec = manifest()["models"][name]
    path = DATA_DIR / spec["file"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["hash"]:
        raise ValueError(f"{path.name} does not match the hash in MANIFEST.json")
    b = xgb.Booster()
    b.load_model(bytearray(gzip.decompress(raw)))
    if int(b.num_features()) != spec["num_feature"]:
        raise ValueError(f"{name}: booster has {b.num_features()} features, manifest {spec['num_feature']}")
    b.feature_names = spec["features"]
    return b


def predict(name: str, df: pd.DataFrame) -> np.ndarray:
    """Predict with model `name`, selecting the model's features from `df` in the
    exact order R used. Multi-class models return shape (n, num_class)."""
    features = model_features(name)
    x = df[features].to_numpy(dtype=np.float64)
    dm = xgb.DMatrix(x, feature_names=features, missing=np.nan)
    return booster(name).predict(dm)


@cache
def fg_lookup() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "fg_lookup.parquet")


@cache
def punt_table() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "punt_df.parquet")

# analysis/thesis/graph_codes/util.py
from __future__ import annotations
import re, numpy as np, pandas as pd
from pathlib import Path
from .config import MODEL_NAME

_EID = re.compile(r"^(base(?:_\d+)?)_(image|text|text_image)$")


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def rmse(a) -> float:
    a = np.asarray(a, float)
    return float(np.sqrt(np.mean(a * a)))


def sqrt_bias_squared(errors: np.ndarray) -> float:
    return np.abs(np.mean(errors))


def sqrt_variance(errors: np.ndarray) -> float:
    return np.sqrt(np.var(errors))


def pretty_model(series: pd.Series) -> pd.Series:
    return (
        series.map(MODEL_NAME)
        .fillna(series)
        .astype(str)
        .str.replace("_", " ")
        .str.replace("-", " ")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def add_ablation_modality(df: pd.DataFrame) -> pd.DataFrame:
    if {"ablation", "modality"}.issubset(df.columns):
        return df
    out = df.copy()
    out[["ablation", "modality"]] = out["exp_id"].astype(str).str.extract(_EID)
    return out


def duplicate_text_base(tidy: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """For modality=='text', display Base values in columns 3 & 5."""
    t = tidy.copy()
    t = t[
        ~((t["modality"] == "text") & t["ablation"].isin(["base_3", "base_5"]))
    ].copy()
    base = t[(t["modality"] == "text") & (t["ablation"] == "base")]
    if not base.empty:
        for new_ab in ["base_3", "base_5"]:
            dup = base.copy()
            dup["ablation"] = new_ab
            t = pd.concat([t, dup], ignore_index=True)
    return t

# analysis/thesis_graphs/nrmse_summary.py
from __future__ import annotations
import numpy as np
import pandas as pd
from analysis.thesis_graphs.util import (
    rmse,
    pretty_model,
    add_ablation_modality,
    duplicate_text_base,
    ensure_dir,
    sqrt_bias_squared,
    sqrt_variance,
)
from analysis.thesis_graphs.config import ABLATION_LABEL


def build_nrmse_observed(
    exp_df: pd.DataFrame, baseline: str = "experiment_mean"
) -> pd.DataFrame:
    d = add_ablation_modality(exp_df).copy()
    d = d.dropna(subset=["correct", "response", "exp_id", "llm_model"])

    eps = 1e-12

    # Compute per-exp_id mean of correct
    d["mean_correct_exp"] = d.groupby("exp_id")["correct"].transform("mean")
    # Baseline RMSE per exp_id
    d["baseline_rmse_exp"] = (d["mean_correct_exp"] - d["correct"]) ** 2
    baseline_rmse_per_exp = (
        d.groupby("exp_id")["baseline_rmse_exp"].mean().apply(np.sqrt)
    )

    # Compute per-(llm_model, exp_id) RMSE on valid trials only
    d = d.assign(err=d["response"] - d["correct"]).dropna(subset=["err"])

    grp = (
        d.groupby(["llm_model", "exp_id"])["err"]
        .agg(
            rmse=lambda s: rmse(s.to_numpy()),
            sqrt_bias_squared=lambda s: sqrt_bias_squared(s.to_numpy()),
            sqrt_variance=lambda s: sqrt_variance(s.to_numpy()),
        )
        .reset_index()
    )
    # import pdb

    # pdb.set_trace()
    # Attach ablation/modality keys
    grp = grp.merge(
        d[["exp_id", "ablation", "modality"]].drop_duplicates(), on="exp_id", how="left"
    )

    # Normalize: use per-exp_id baseline RMSE
    grp["baseline_rmse"] = grp["exp_id"].map(baseline_rmse_per_exp)
    grp["nrmse"] = grp["rmse"] / (grp["baseline_rmse"] + eps)
    grp["n_sqrt_bias_squared"] = grp["sqrt_bias_squared"] / (grp["baseline_rmse"] + eps)
    grp["n_sqrt_variance"] = grp["sqrt_variance"] / (grp["baseline_rmse"] + eps)

    # Collapse to (llm_model, ablation, modality)
    out = grp.groupby(["llm_model", "ablation", "modality"], as_index=False)[
        [
            "nrmse",
            "sqrt_bias_squared",
            "sqrt_variance",
            "n_sqrt_bias_squared",
            "n_sqrt_variance",
        ]
    ].mean()

    # Duplicate text Base into Base_3/Base_5 (only for text modality)
    out = duplicate_text_base(
        out,
        [
            "nrmse",
            "sqrt_bias_squared",
            "sqrt_variance",
            "n_sqrt_bias_squared",
            "n_sqrt_variance",
        ],
    )

    # Pretty labels
    out["model_pretty"] = pretty_model(out["llm_model"])
    out["ablation_pretty"] = out["ablation"].map(ABLATION_LABEL).fillna(out["ablation"])

    # Final: drop combos that are still empty (no valid trials)
    out = out.dropna(subset=["nrmse"]).reset_index(drop=True)

    return out[
        [
            "llm_model",
            "model_pretty",
            "ablation",
            "ablation_pretty",
            "modality",
            "nrmse",
            "sqrt_bias_squared",
            "sqrt_variance",
            "n_sqrt_bias_squared",
            "n_sqrt_variance",
        ]
    ]


def save_nrmse_observed_csv(
    exp_path: str, out_dir: str, exp_tag: str, baseline="experiment_mean"
):
    df = pd.read_csv(exp_path)
    ensure_dir(out_dir)
    tidy = build_nrmse_observed(df, baseline=baseline)
    tidy.to_csv(f"{out_dir}/{exp_tag}_summary_nrmse_observed.csv", index=False)

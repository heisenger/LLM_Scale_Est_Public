from __future__ import annotations
from glob import glob
from pathlib import Path
import pandas as pd, numpy as np
from .config import ABLATION_LABEL, COMBO_MODELS
from .util import rmse, ensure_dir, pretty_model


def _read_weights(paths: list[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        df = pd.read_parquet(p)
        ab = "base" if "base" in df.columns else "ablation"
        ll = "LLM_model" if "LLM_model" in df.columns else "llm_model"
        wc = "w_img_bayes" if "w_img_bayes" in df.columns else "w_img"
        df = df.rename(columns={ab: "ablation", ll: "llm_model", wc: "w_img_bayes"})
        dfs.append(df[["llm_model", "ablation", "w_img_bayes"]])
    return (
        pd.concat(dfs, ignore_index=True)
        if dfs
        else pd.DataFrame(columns=["llm_model", "ablation", "w_img_bayes"])
    )


def load_bayes_weights_artifacts(root: str | Path):
    root = Path(root)
    r_files = glob(str(root / "**/weights_bayes_per_range.parquet"), recursive=True)
    s_files = glob(str(root / "**/weights_bayes_per_stimulus.parquet"), recursive=True)
    R = _read_weights(r_files)
    S = _read_weights(s_files)
    w_range = (
        R.groupby(["llm_model", "ablation"], as_index=False)["w_img_bayes"]
        .mean()
        .rename(columns={"w_img_bayes": "w_img_bayes_range"})
    )
    w_stim = (
        S.groupby(["llm_model", "ablation"], as_index=False)["w_img_bayes"]
        .mean()
        .rename(columns={"w_img_bayes": "w_img_bayes_stimulus"})
    )
    return w_range, w_stim


def load_empiricallinear_weights_artifacts(root: str | Path) -> pd.DataFrame:
    """
    Find **weights_global.parquet** files and return a tidy DF with
    columns:
        - ablation
        - llm_model   (if the file includes it)
        - w_img_empirical  (from w_img_norm or similar)

    If the file has no llm_model column, we return only (ablation, w_img_empirical),
    which we will broadcast to all LLMs for that ablation.
    """
    root = Path(root)
    files = glob(str(root / "**/weights_global.parquet"), recursive=True)
    dfs = []
    for p in files:
        df = pd.read_parquet(p)

        # standardize column names
        ab = "base" if "base" in df.columns else "ablation"
        ll = (
            "LLM_model"
            if "LLM_model" in df.columns
            else ("llm_model" if "llm_model" in df.columns else None)
        )
        wcol = next(
            (
                c
                for c in [
                    "w_img_norm",
                    "w_img",
                    "w_image",
                    "weight_image",
                    "w_image_norm",
                ]
                if c in df.columns
            ),
            None,
        )
        if wcol is None:
            continue

        df = df.rename(columns={ab: "ablation"})
        df["w_img_empirical"] = pd.to_numeric(df[wcol], errors="coerce")

        keep_cols = ["ablation", "w_img_empirical"]
        if ll is not None:
            df = df.rename(columns={ll: "llm_model"})
            keep_cols.append("llm_model")

        dfs.append(df[keep_cols].copy())

    if not dfs:
        return pd.DataFrame(columns=["llm_model", "ablation", "w_img_empirical"])

    out = pd.concat(dfs, ignore_index=True)
    # If multiple files per key, average
    if "llm_model" in out.columns:
        out = out.groupby(["llm_model", "ablation"], as_index=False)[
            "w_img_empirical"
        ].mean()
    else:
        out = out.groupby(["ablation"], as_index=False)["w_img_empirical"].mean()
    return out


def build_cue_combo_summary(
    exp_df: pd.DataFrame,
    cue_df: pd.DataFrame,
    artifacts_root: str | Path,
    baseline: str = "experiment_mean",
) -> pd.DataFrame:
    w_range, w_stim = load_bayes_weights_artifacts(artifacts_root)

    cue = cue_df.rename(columns={"base": "ablation", "LLM_model": "llm_model"}).copy()
    cue["modality"] = "text_image"
    cue = cue[
        ["llm_model", "ablation", "modality", "model", "aic_weight", "rmse_vs_truth"]
    ]

    # normalize candidate RMSE
    if baseline == "observed_text_image":
        obs = (
            exp_df.assign(err=lambda d: d["response"] - d["correct"])
            .query("exp_id.str.endswith('_text_image')", engine="python")
            .groupby(["llm_model", "exp_id"], as_index=False)["err"]
            .apply(lambda s: rmse(s.to_numpy()))
            .rename(columns={"err": "rmse_obs_text_image"})
        )
        obs["ablation"] = obs["exp_id"].str.extract(r"^(base(?:_\d+)?)_text_image$")
        cue = cue.merge(
            obs[["llm_model", "ablation", "rmse_obs_text_image"]],
            on=["llm_model", "ablation"],
            how="left",
        )
        cue["nrmse"] = cue["rmse_vs_truth"] / (cue["rmse_obs_text_image"] + 1e-12)
    else:
        mu = exp_df["correct"].mean()
        cue["nrmse"] = cue["rmse_vs_truth"] / (rmse(mu - exp_df["correct"]) + 1e-12)

    # join Bayes weights + optional empirical weight
    cue = cue.merge(w_range, on=["llm_model", "ablation"], how="left").merge(
        w_stim, on=["llm_model", "ablation"], how="left"
    )

    # --- NEW: EmpiricalLinear weights from artifacts (weights_global.parquet)
    emp_art = load_empiricallinear_weights_artifacts(artifacts_root)
    if not emp_art.empty:
        if "llm_model" in emp_art.columns:
            cue = cue.merge(emp_art, on=["llm_model", "ablation"], how="left")
        else:
            # file had no llm_model → broadcast per ablation
            cue = cue.merge(emp_art, on=["ablation"], how="left")

    # (optional) keep the old path in case overall.parquet *also* carries a param:
    emp_col = next(
        (
            c
            for c in [
                "w_img",
                "param_w_img",
                "param_w_image",
                "param_w",
                "weight_image",
            ]
            if c in cue_df.columns
        ),
        None,
    )
    if emp_col:
        Emp = (
            cue_df.rename(columns={"base": "ablation", "LLM_model": "llm_model"})
            .query("model == 'EmpiricalLinear'")[["llm_model", "ablation", emp_col]]
            .rename(columns={emp_col: "w_img_empirical_overall"})
        )
        cue = cue.merge(Emp, on=["llm_model", "ablation"], how="left")
        # prefer explicit overall value if present
        cue["w_img_empirical"] = cue["w_img_empirical_overall"].combine_first(
            cue["w_img_empirical"]
        )
        if "w_img_empirical_overall" in cue.columns:
            cue.drop(columns=["w_img_empirical_overall"], inplace=True)

    rows = []
    for (llm, ab, mod, name), g in cue.groupby(
        ["llm_model", "ablation", "modality", "model"], sort=False
    ):
        if name not in COMBO_MODELS:  # only the four you want
            continue
        prob = float(g["aic_weight"].sum())
        nrmse_cand = float(g["nrmse"].mean())
        if name == "Bayes(range)":
            w = g["w_img_bayes_range"].mean()
        elif name == "Bayes(stimulus)":
            w = g["w_img_bayes_stimulus"].mean()
        elif name == "Equal":
            w = 0.5
        else:  # EmpiricalLinear
            # import pdb

            # pdb.set_trace()
            w = g["empirical_linear_w_img_norm"].mean()

        rows.append(
            {
                "llm_model": llm,
                "model_pretty": pretty_model(pd.Series([llm])).iloc[0],
                "ablation": ab,
                "ablation_pretty": ABLATION_LABEL.get(ab, ab),
                "modality": mod,
                "combo_model": name,
                "prob": prob,
                "nrmse": nrmse_cand,
                "w_img": float(w) if pd.notna(w) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_cue_combo_csv(
    exp_path: str,
    cue_path: str,
    artifacts_root: str,
    out_dir: str,
    exp_tag: str,
    baseline="experiment_mean",
):
    # import pdb

    # pdb.set_trace()
    exp_df = pd.read_csv(exp_path)
    cue_df = (
        pd.read_parquet(cue_path)
        if cue_path.endswith(".parquet")
        else pd.read_csv(cue_path)
    )
    ensure_dir(out_dir)
    cue_long = build_cue_combo_summary(
        exp_df, cue_df, artifacts_root=artifacts_root, baseline=baseline
    )
    cue_long.to_csv(f"{out_dir}/{exp_tag}_summary_cue_combo_long.csv", index=False)

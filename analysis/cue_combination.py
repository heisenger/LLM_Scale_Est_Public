import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from scipy.optimize import minimize

import os, json
import pandas as pd


# ---------- fitting utils ----------
def _aic_from_mse(mse: float, n: int, k: int) -> float:
    return float(n * np.log(mse + 1e-12) + 2 * k)


def _aic_weights(aic_series: pd.Series) -> pd.Series:
    aic = aic_series.copy()
    aic_min = aic.min(skipna=True)
    delta = aic - aic_min
    w_raw = np.exp(-0.5 * delta)
    return w_raw / w_raw.sum(skipna=True)


def _rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan


def _bias(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(a[m] - b[m])) if m.any() else np.nan


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return np.nan
    yt = y_true[m]
    yp = y_pred[m]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


# ---------- alignment / reliabilities ----------


def _sanitize_merged(merged: pd.DataFrame) -> pd.DataFrame:
    # just removes things that are nan or infinity
    cols = ["response_txt", "response_img", "response_combo", "correct"]
    for c in cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    m = np.isfinite(merged[cols]).all(axis=1)
    return merged.loc[m].reset_index(drop=True)


def _merge_trials(
    df_text: pd.DataFrame, df_image: pd.DataFrame, df_combo: pd.DataFrame
) -> pd.DataFrame:
    """
    Strict alignment on (range_category, trial, stimulus_id, correct).
    Assumes trials are synchronized across modalities.
    """
    merged = df_text.merge(
        df_image,
        on=["range_category", "trial", "stimulus_id", "correct"],
        suffixes=("_txt", "_img"),
    ).merge(df_combo, on=["range_category", "trial", "stimulus_id", "correct"])
    merged = merged.rename(columns={"response": "response_combo"})
    return merged


def _per_range_error_variances(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Variance per (range) using the 50 measurements — reliability per stimulus.
    """
    eps = 1e-6  # small delta to prevent zero variance
    g = merged.groupby("range_category", sort=False)
    s2_txt = g.apply(
        lambda d: np.maximum(
            eps, np.var((d["response_txt"] - d["correct"]).to_numpy(), ddof=1)
        )
    ).rename("s2_txt")
    s2_img = g.apply(
        lambda d: np.maximum(
            eps, np.var((d["response_img"] - d["correct"]).to_numpy(), ddof=1)
        )
    ).rename("s2_img")

    return pd.concat([s2_txt, s2_img], axis=1).reset_index()


def _per_stim_error_variances(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Variance per (range, stimulus) using the 5 repeats — reliability per stimulus.
    """
    eps = 1e-6  # small delta to prevent zero variance
    g = merged.groupby(["range_category", "stimulus_id"], sort=False)
    s2_txt = g.apply(
        lambda d: np.maximum(
            eps, np.var((d["response_txt"] - d["correct"]).to_numpy(), ddof=1)
        )
    ).rename("s2_txt")
    s2_img = g.apply(
        lambda d: np.maximum(
            eps, np.var((d["response_img"] - d["correct"]).to_numpy(), ddof=1)
        )
    ).rename("s2_img")
    return pd.concat([s2_txt, s2_img], axis=1).reset_index()


# ---------- combo scheme predictions ----------
def pred_equal(txt, img):
    return 0.5 * txt + 0.5 * img


def pred_bayes_range(merged: pd.DataFrame, s2_range: pd.DataFrame):
    # Variance weighted cue combination prediction and weights (by range)
    tbl = s2_range.copy()
    w_img = (1.0 / tbl["s2_img"]) / ((1.0 / tbl["s2_img"]) + (1.0 / tbl["s2_txt"]))
    tbl["w_img_bayes"] = w_img
    tbl["w_txt_bayes"] = 1.0 - w_img
    w_map = tbl.set_index("range_category")[["w_img_bayes", "w_txt_bayes"]].to_dict(
        "index"
    )
    preds = np.empty(len(merged), dtype=float)
    for i, (rng, t, im) in enumerate(
        zip(merged["range_category"], merged["response_txt"], merged["response_img"])
    ):
        w = w_map[rng]
        preds[i] = w["w_img_bayes"] * im + w["w_txt_bayes"] * t
    return preds, tbl[["range_category", "w_img_bayes", "w_txt_bayes"]]


def pred_bayes_stimulus(merged: pd.DataFrame, s2_stim: pd.DataFrame):
    # Variance weighted cue combination prediction and weights (by stimulus)
    tbl = s2_stim.copy()
    w_img = (1.0 / tbl["s2_img"]) / ((1.0 / tbl["s2_img"]) + (1.0 / tbl["s2_txt"]))
    tbl["w_img_bayes"] = w_img
    tbl["w_txt_bayes"] = 1.0 - w_img
    # merge weights onto trials by (range, stimulus)
    weights = tbl[["range_category", "stimulus_id", "w_img_bayes", "w_txt_bayes"]]
    merged_w = merged.merge(weights, on=["range_category", "stimulus_id"], how="left")
    preds = (
        merged_w["w_img_bayes"].to_numpy() * merged_w["response_img"].to_numpy()
        + merged_w["w_txt_bayes"].to_numpy() * merged_w["response_txt"].to_numpy()
    )
    return preds, weights


def fit_empirical_linear(merged: pd.DataFrame):
    """
    Constrained linear fit:
    response_combo ~ w_txt*response_txt + w_img*response_img
    subject to w_txt >= 0, w_img >= 0, w_txt + w_img = 1.
    Intercept removed (canonical psychophysics cue integration).
    """
    txt = merged["response_txt"].to_numpy(float)
    img = merged["response_img"].to_numpy(float)
    y = merged["response_combo"].to_numpy(float)

    mfit = np.isfinite(txt) & np.isfinite(img) & np.isfinite(y)
    if mfit.sum() == 0:
        return None

    txt, img, y = txt[mfit], img[mfit], y[mfit]

    # Optimize over w_img only (since w_txt = 1 - w_img)
    def mse_loss(w_img):
        w_img = float(w_img)
        if w_img < 0 or w_img > 1:
            return np.inf
        yhat = (1 - w_img) * txt + w_img * img
        return np.mean((y - yhat) ** 2)

    res = minimize(mse_loss, x0=0.5, bounds=[(0, 1)])
    w_img = float(res.x[0])
    w_txt = 1 - w_img

    # Predictions
    yhat_full = w_txt * txt + w_img * img
    mse = float(np.mean((y - yhat_full) ** 2)) if len(y) else np.nan
    aic = _aic_from_mse(mse, len(y), 2) if np.isfinite(mse) else np.nan

    return {
        "params": {"w_txt_norm": w_txt, "w_img_norm": w_img},
        "pred": yhat_full,
        "mse": mse,
        "aic": aic,
        "k": 2,  # free params: w_img + intercept is fixed out
    }


def fit_empirical_linear_unconstrained(merged: pd.DataFrame):
    """
    Linear fit: response_combo ~ a_txt*response_txt + a_img*response_img + intercept
    NaN/Inf-safe with robust fallbacks.
    """
    txt = merged["response_txt"].to_numpy(dtype=float)
    img = merged["response_img"].to_numpy(dtype=float)
    y = merged["response_combo"].to_numpy(dtype=float)

    # finite mask for fitting
    mfit = np.isfinite(txt) & np.isfinite(img) & np.isfinite(y)

    # robust least squares
    X = np.column_stack([txt[mfit], img[mfit], np.ones(mfit.sum())])
    y_fit = y[mfit]
    try:
        beta, *_ = np.linalg.lstsq(X, y_fit, rcond=None)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X) @ y_fit

    a_txt, a_img, a0 = map(float, beta)

    # full predictions; eval on finite pairs
    yhat_full = a_txt * txt + a_img * img + a0
    meval = np.isfinite(yhat_full) & np.isfinite(y)
    mse = float(np.mean((y[meval] - yhat_full[meval]) ** 2)) if meval.any() else np.nan
    aic = _aic_from_mse(mse, int(meval.sum()), 3) if np.isfinite(mse) else np.nan

    # normalized weights with guard
    s = a_txt + a_img
    w_txt = a_txt / s if s != 0 else np.nan
    w_img = a_img / s if s != 0 else np.nan

    return {
        "params": {
            "a_txt": a_txt,
            "a_img": a_img,
            "intercept": a0,
            "w_txt_norm": w_txt,
            "w_img_norm": w_img,
        },
        "pred": yhat_full,
        "mse": mse,
        "aic": aic,
        "k": 3,
    }


def pred_wta_truth(merged: pd.DataFrame):
    """winner takes all algorithm"""
    txt = merged["response_txt"].to_numpy()
    img = merged["response_img"].to_numpy()
    tru = merged["correct"].to_numpy()
    choose_txt = np.abs(txt - tru) <= np.abs(img - tru)
    return np.where(choose_txt, txt, img)


def pred_combo_oracle(
    df: pd.DataFrame,
    y_col: str = "correct",
    text_col: str = "pred_text",
    image_col: str = "pred_image",
    ridge: float = 1e-9,
) -> np.ndarray:
    """
    Oracle (optimistic) linear fusion, per *current df slice*:
      1) Affine-calibrate each unimodal output to y (same rows).
      2) Estimate residual covariance Σ and compute BLUE weights.
      3) Return fused prediction per row (aligned to df.index).

    Assumes df has columns: y_col, text_col, image_col.
    Returns: np.ndarray (len(df)) with fused preds (NaN where fusion not possible).
    """
    n = len(df)
    out = np.full(n, np.nan, dtype=float)

    # pull columns
    y = df[y_col].to_numpy(float)
    rt = df[text_col].to_numpy(float)
    ri = df[image_col].to_numpy(float)

    # valid rows (need y, rt, ri)
    m = ~np.isnan(y) & ~np.isnan(rt) & ~np.isnan(ri)
    if m.sum() < 2:
        return out  # not enough data to calibrate/fuse

    yv, rtv, riv = y[m], rt[m], ri[m]

    # affine calibration: y ≈ a*r + b (least squares with intercept)
    def _affine(y_, r_):
        X = np.column_stack([r_, np.ones_like(r_)])
        (a, b), *_ = np.linalg.lstsq(X, y_, rcond=None)
        return float(a), float(b)

    a_t, b_t = _affine(yv, rtv)
    a_i, b_i = _affine(yv, riv)

    yhat_t = a_t * rtv + b_t
    yhat_i = a_i * riv + b_i

    # residual covariance (population) + small ridge for stability
    et = yhat_t - yv
    ei = yhat_i - yv
    E = np.vstack([et, ei])
    Sigma = np.cov(E, bias=True) + ridge * np.eye(2)

    try:
        inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        return out  # pathological; keep NaNs

    ones = np.ones((2, 1))
    w = (inv @ ones) / float(ones.T @ inv @ ones)
    w_t, w_i = float(w[0, 0]), float(w[1, 0])

    fused_valid = w_t * yhat_t + w_i * yhat_i
    out[m] = fused_valid
    return out, {"w_txt_norm": w_t, "w_img_norm": w_i}


import numpy as np
import pandas as pd
from typing import Tuple, Dict


def pred_combo_non_oracle(
    df: pd.DataFrame,
    text_col: str = "pred_text",
    image_col: str = "pred_image",
    id_cols: tuple = ("range_category", "stimulus_id"),
    eps: float = 1e-9,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Non-oracle (truth-free) fusion using *channel-level* reliability
    estimated from *per-stimulus repeat variances*.

    Steps:
      1) For each stimulus (id_cols), compute repeat variance of text & image.
      2) Average those variances across stimuli -> channel reliabilities.
      3) Use global precision weights (1/var) to fuse every trial.

    Returns
    -------
    fused : np.ndarray
        Fused prediction per trial (len(df)); NaN where a modality is NaN.
    info : dict
        {'w_txt_norm', 'w_img_norm'}
    """
    n = len(df)
    out = np.full(n, np.nan, dtype=float)

    # Extract modality predictions
    t_all = df[text_col].to_numpy(float)
    i_all = df[image_col].to_numpy(float)
    m_all = np.isfinite(t_all) & np.isfinite(i_all)

    # Group by stimulus to isolate repeat variability
    key = df[list(id_cols)].apply(tuple, axis=1)

    vt_list, vi_list = [], []
    for _, idx in key.groupby(key).groups.items():
        idx = np.asarray(idx)
        tm = t_all[idx]
        im = i_all[idx]
        m = np.isfinite(tm) & np.isfinite(im)
        if m.sum() < 2:
            continue  # need ≥2 repeats to estimate variance
        vt_list.append(float(np.var(tm[m], ddof=1)))
        vi_list.append(float(np.var(im[m], ddof=1)))

    # Average variances across stimuli
    vt_bar = np.mean(vt_list) if vt_list else np.nan
    vi_bar = np.mean(vi_list) if vi_list else np.nan

    # Convert to precision weights
    wt = 1.0 / (vt_bar + eps) if np.isfinite(vt_bar) else 0.0
    wi = 1.0 / (vi_bar + eps) if np.isfinite(vi_bar) else 0.0
    Z = wt + wi

    if Z > 0 and np.isfinite(Z):
        wtn, win = wt / Z, wi / Z
    else:
        wtn = win = 0.5  # fallback: equal weights

    # Apply global weights to all valid trials
    out[m_all] = wtn * t_all[m_all] + win * i_all[m_all]

    return out, {"w_txt_norm": wtn, "w_img_norm": win}


def pred_combo_non_oracle_old(
    df: pd.DataFrame,
    text_col: str = "pred_text",
    image_col: str = "pred_image",
    id_cols: tuple = ("range_category", "stimulus_id"),
    eps: float = 1e-9,
) -> tuple[np.ndarray, dict]:
    """
    Non-oracle (truth-free) fusion for the *current df slice*:
      - Rows sharing (range_category, stimulus_id) are repeats of the same stimulus.
      - Compute per-stimulus variances per modality.
      - Fuse per-trial predictions with precision weights 1/var.
    Returns
    -------
    fused : np.ndarray
        Length = len(df), fused prediction per trial.
    info : dict
        Average normalized weights across stimuli.
    """
    n = len(df)
    out = np.full(n, np.nan, dtype=float)
    weights = []

    if any(c not in df.columns for c in id_cols):
        return out, {}

    # group indices for fast assignment
    key = df[list(id_cols)].apply(tuple, axis=1)

    for stim_key, idx in key.groupby(key).groups.items():
        h = df.iloc[idx]
        t = h[text_col].to_numpy(float)
        i = h[image_col].to_numpy(float)

        m = ~np.isnan(t) & ~np.isnan(i)
        if m.sum() == 0:
            continue

        t, i = t[m], i[m]
        vt = float(np.var(t, ddof=1)) if t.size > 1 else 0.0
        vi = float(np.var(i, ddof=1)) if i.size > 1 else 0.0

        wt = 1.0 / (vt + eps)
        wi = 1.0 / (vi + eps)
        Z = wt + wi

        if Z <= 0 or not np.isfinite(Z):
            fused_trials = 0.5 * (t + i)  # per-trial equal weights
        else:
            fused_trials = (wt * t + wi * i) / Z  # per-trial weighted

        out[np.array(idx)[m]] = fused_trials  # assign back to true indices
        weights.append((wt / Z, wi / Z))

    # Average weights across stimuli
    if weights:
        w_txt_norm, w_img_norm = map(np.mean, zip(*weights))
    else:
        w_txt_norm = w_img_norm = np.nan

    return out, {"w_txt_norm": w_txt_norm, "w_img_norm": w_img_norm}


# ---------- top-level ----------
def combo_model_selection(
    df_text: pd.DataFrame, df_image: pd.DataFrame, df_combo: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Trial-aligned combo model selection & benchmarking.
    Returns {
      "overall": metrics per model,
      "per_range": metrics per model within range,
      "weights_bayes_per_range": per-range Bayes weights,
      "weights_bayes_per_stimulus": per-stimulus Bayes weights,
      "weights_global": params of empirical/dynamic models,
      "merged": merged trial-aligned DataFrame
    }
    """
    merged = _merge_trials(df_text, df_image, df_combo)
    merged = _sanitize_merged(merged)

    # predictions for each scheme
    preds: Dict[str, Dict] = {}

    # Bayes optimal for a ground truth aware oracle
    y_pred_oracle, oracle_params = pred_combo_oracle(
        merged, y_col="correct", text_col="response_txt", image_col="response_img"
    )
    mse_pred_oracle = float(
        np.mean((merged["response_combo"].to_numpy() - y_pred_oracle) ** 2)
    )
    preds["Combo_Oracle"] = {"pred": y_pred_oracle, "mse_fit": mse_pred_oracle, "k": 0}

    # Bayes optimal for non ground truth aware oracle
    y_preds_non_oracle, non_oracle_params = pred_combo_non_oracle(
        merged,
        text_col="response_txt",
        image_col="response_img",
        id_cols=("range_category", "stimulus_id"),
    )

    mse_pred_non_oracle = float(
        np.mean((merged["response_combo"].to_numpy() - y_preds_non_oracle) ** 2)
    )
    preds["Combo_NonOracle"] = {
        "pred": y_preds_non_oracle,
        "mse_fit": mse_pred_non_oracle,
        "k": 0,
    }

    # Equal: 0 params
    y_equal = pred_equal(
        merged["response_txt"].to_numpy(), merged["response_img"].to_numpy()
    )
    mse_equal = float(np.mean((merged["response_combo"].to_numpy() - y_equal) ** 2))
    preds["Equal"] = {"pred": y_equal, "mse_fit": mse_equal, "k": 0}

    # EmpiricalLinear: 3 params
    emp = fit_empirical_linear(merged)
    preds["EmpiricalLinear"] = {
        "pred": emp["pred"],
        "mse_fit": emp["mse"],
        "k": emp["k"],
        "params": emp["params"],
    }

    y_llm = merged["response_combo"].to_numpy()
    y_true = merged["correct"].to_numpy()
    overall_rows = []

    for name, d in preds.items():
        n = len(y_llm)
        k = d["k"]
        aic = _aic_from_mse(d["mse_fit"], n, k)
        overall_rows.append(
            {
                "model": name,
                "scope": "overall",
                "k": k,
                "aic": aic,
                "rmse_to_llm": _rmse(d["pred"], y_llm),
                "r2_to_llm": _r2(y_llm, d["pred"]),
                "r2_to_truth": _r2(y_true, d["pred"]),
                "rmse_vs_truth": _rmse(d["pred"], y_true),
                "bias_vs_truth": _bias(d["pred"], y_true),
            }
        )
    # here we add the actual observed combination
    overall_rows.append(
        {
            "model": "LLM(Combined)",
            "scope": "overall",
            "k": np.nan,
            "aic": np.nan,
            "rmse_to_llm": 0.0,
            "r2_to_llm": 1.0,
            "r2_to_truth": _r2(y_true, y_llm),
            "rmse_vs_truth": _rmse(y_llm, y_true),
            "bias_vs_truth": _bias(y_llm, y_true),
        }
    )
    overall_df = (
        pd.DataFrame(overall_rows)
        .sort_values(["aic", "rmse_to_llm"], na_position="last")
        .reset_index(drop=True)
    )

    overall_df["aic_weight"] = np.nan
    mask = overall_df["aic"].notna()
    overall_df.loc[mask, "aic_weight"] = _aic_weights(overall_df.loc[mask, "aic"])

    # PER-RANGE metrics
    per_rows = []
    for rng, g in merged.groupby("range_category", sort=False):
        y_llm_r = g["response_combo"].to_numpy()
        y_true_r = g["correct"].to_numpy()
        mask = (merged["range_category"] == rng).to_numpy()
        for name, d in preds.items():
            yhat = np.asarray(d["pred"])[mask]
            n = len(yhat)
            k = d["k"]
            mse_fit_r = float(np.mean((y_llm_r - yhat) ** 2))
            aic_r = _aic_from_mse(mse_fit_r, n, k)
            per_rows.append(
                {
                    "range": rng,
                    "model": name,
                    "k": k,
                    "aic": aic_r,
                    "rmse_to_llm": _rmse(yhat, y_llm_r),
                    "r2_to_llm": _r2(y_llm_r, yhat),
                    "rmse_vs_truth": _rmse(yhat, y_true_r),
                    "bias_vs_truth": _bias(yhat, y_true_r),
                }
            )
        # now we add the actual observed combination
        per_rows.append(
            {
                "range": rng,
                "model": "LLM(Combined)",
                "k": np.nan,
                "aic": np.nan,
                "rmse_to_llm": 0.0,
                "r2_to_llm": 1.0,
                "rmse_vs_truth": _rmse(y_llm_r, y_true_r),
                "bias_vs_truth": _bias(y_llm_r, y_true_r),
            }
        )
    per_range_df = (
        pd.DataFrame(per_rows)
        .sort_values(["range", "aic", "rmse_to_llm"], na_position="last")
        .reset_index(drop=True)
    )

    # Compute the weights of image
    weights_global = {
        "empirical_linear": emp["params"],
        "combo_oracle": oracle_params,
        "combo_non_oracle": non_oracle_params,
    }

    predictions = {
        "Combo_Oracle": y_pred_oracle,
        "Combo_NonOracle": y_preds_non_oracle,
        "Equal": y_equal,
        "EmpiricalLinear": emp["pred"],
        "LLM(Combined)": merged["response_combo"].to_numpy(),
    }
    return {
        "overall": overall_df,
        "per_range": per_range_df,
        "weights_global": weights_global,
        "merged": merged,
        "predictions": predictions,
    }


def export_combo_results(llm_name: str, out: dict, outdir: str = "results"):
    """
    Export summary output for visualization:
      - overall.csv, per_range.csv
      - weights_bayes_per_range.csv, weights_bayes_per_stimulus.csv
      - weights_global.json
      - predictions.parquet (per-trial yhat for each model)
    """
    # d = os.path.join(outdir, f"{llm_name}")
    d = outdir
    os.makedirs(d, exist_ok=True)

    out["overall"].to_csv(os.path.join(d, "overall.csv"), index=False)
    out["per_range"].to_csv(os.path.join(d, "per_range.csv"), index=False)

    with open(os.path.join(d, "weights_global.json"), "w") as f:
        json.dump(out["weights_global"], f, indent=2)

    # Long-format predictions for flexible plotting later
    preds = out["predictions"]
    merged = out["merged"].copy()
    rows = []
    for model_name, yhat in preds.items():
        rows.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "range_category": merged["range_category"],
                    "trial": merged["trial"],
                    "stimulus_id": merged["stimulus_id"],
                    "correct": merged["correct"],
                    "y_text": merged["response_txt"],
                    "y_img": merged["response_img"],
                    "y_llm": merged["response_combo"],
                    "yhat": yhat,
                }
            )
        )
    long_df = pd.concat(rows, ignore_index=True)
    long_df.to_parquet(os.path.join(d, "predictions.parquet"), index=False)

    print(f"[export_combo_results] Saved to {d}")


def run_and_save(llm_name, df_text, df_image, df_combo, outdir="results"):
    """
    One-shot convenience: fit, compare, and export.
    """
    out = combo_model_selection(df_text, df_image, df_combo)
    export_combo_results(llm_name, out, outdir=outdir)
    return out  # optional


if __name__ == "__main__":
    # folder_path_parent = "experiments/text_image/marker_location"
    folder_path_parents = [
        "experiments/text_image/line_length_ratio",
        "experiments/text_image/marker_location",
        "experiments/text_image/maze_distance",
    ]

    exp_name_list = [
        "base",
        "base_1",
        "base_2",
        "base_3",
        "base_4",
        "base_5",
        "base_6",
        "base_7",
        "base_8",
    ]

    exp_ids_list = [
        ["base_image", "base_text", "base_text_image"],
        ["base_1_image", "base_1_text", "base_1_text_image"],
        ["base_2_image", "base_2_text", "base_2_text_image"],
        ["base_3_image", "base_text", "base_3_text_image"],
        ["base_4_image", "base_4_text", "base_4_text_image"],
        ["base_5_image", "base_text", "base_5_text_image"],
        ["base_6_image", "base_6_text", "base_6_text_image"],
        ["base_7_image", "base_7_text", "base_7_text_image"],
        ["base_8_image", "base_8_text", "base_8_text_image"],
    ]

    llm_names = [
        "google_gemini-2.5-flash-lite",
        "gemma-3-4b-it",
        # "openai_gpt-4.1-mini",
        "microsoft_phi-4-multimodal-instruct",
        "meta-llama_llama-4-maverick",
        "mistralai_mistral-small-3.2-24b-instruct",
        "qwen_qwen2.5-vl-32b-instruct",
        "openai_gpt-4o-2024-08-06",
        "anthropic_claude-3.7-sonnet",
        "GPT-5 Mini",
    ]

    for folder_path_parent in folder_path_parents:
        for exp_name, exp_ids in zip(exp_name_list, exp_ids_list):
            folder_paths = [
                # os.path.join(folder_path_parent, exp_id, mode)
                f"{folder_path_parent}/runs/{exp_id}"
                for exp_id in exp_ids
            ]

            for llm_name in llm_names:
                print(f"Processing {llm_name}...")
                df_image = pd.read_csv(
                    os.path.join(
                        folder_paths[0], llm_name, "derived/experimental_data.csv"
                    )
                )
                df_text = pd.read_csv(
                    os.path.join(
                        folder_paths[1], llm_name, "derived/experimental_data.csv"
                    )
                )
                df_combo = pd.read_csv(
                    os.path.join(
                        folder_paths[2], llm_name, "derived/experimental_data.csv"
                    )
                )

                # Run the combo model selection and export results
                run_and_save(
                    llm_name,
                    df_text,
                    df_image,
                    df_combo,
                    outdir=f"{folder_path_parent}/runsets/{exp_name}/{llm_name}",
                )

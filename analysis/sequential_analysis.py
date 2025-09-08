import numpy as np
import pandas as pd
from typing import Sequence, Optional, Dict


# ---------- helpers ----------
def ewma_prior_within_runs(
    df, lam=0.8, y_col="correct", run_col="run_id", pos_col="stimulus_id"
):
    m_all = np.empty(len(df), dtype=float)
    # df is already sorted by [run_id, stimulus_id] and has a RangeIndex 0..n-1
    for _, g in df.groupby(run_col, sort=False):
        y = g[y_col].to_numpy()
        m = np.empty_like(y, dtype=float)
        m[0] = y[0]
        for t in range(1, len(y)):
            m[t] = lam * m[t - 1] + (1 - lam) * y[t - 1]
        m_all[g.index.to_numpy()] = m  # now safe (positional)
    return m_all


def position_weights(
    df: pd.DataFrame,
    m_col: str = "m_prior",
    y_col: str = "correct",
    r_col: str = "response",
    pos_col: str = "stimulus_id",
    run_col: str = "run_id",
) -> pd.DataFrame:
    """
    For each within-run position (stimulus_id), pool across available runs:
      w_hat = sum_r (y-m)(r-m) / sum_r (y-m)^2
    Returns columns: ['stimulus_id','w_hat','num','den','n_runs'].
    """
    rows = []
    for k, gk in df.groupby(pos_col, sort=True):
        X = (gk[y_col] - gk[m_col]).to_numpy()
        Z = (gk[r_col] - gk[m_col]).to_numpy()
        den = float(np.nansum(X * X))
        num = float(np.nansum(X * Z))
        if den <= 0 or not np.isfinite(den):
            w = np.nan
        else:
            w = num / den
            if np.isfinite(w):
                w = min(1.0, max(0.0, w))  # optional clip to [0,1]
            else:
                w = np.nan
        rows.append(
            {
                "stimulus_id": int(k),
                "w_hat": w,
                "num": num,
                "den": den,
                "n_runs": gk[run_col].nunique(),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("stimulus_id") if not out.empty else out


def position_prior_pull(
    df: pd.DataFrame,
    m_col: str = "m_prior",
    y_col: str = "correct",
    r_col: str = "response",
    pos_col: str = "stimulus_id",
) -> pd.DataFrame:
    """
    For each within-run position, fit through-origin: (r - y) ~ beta * (m - y).
    Returns columns: ['stimulus_id','beta_prior'].
    """
    rows = []
    for k, gk in df.groupby(pos_col, sort=True):
        X = (gk[m_col] - gk[y_col]).to_numpy()
        Y = (gk[r_col] - gk[y_col]).to_numpy()
        den = float(np.nansum(X * X))
        num = float(np.nansum(X * Y))
        beta = np.nan if den <= 0 or not np.isfinite(den) else (num / den)
        rows.append({"stimulus_id": int(k), "beta_prior": beta})
    out = pd.DataFrame(rows)
    return out.sort_values("stimulus_id") if not out.empty else out


# ---------- bootstrap with robust skipping ----------
def block_bootstrap_by_run(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    lam: float = 0.8,
    B: int = 1000,
    random_state: int = 1,
) -> Dict[str, pd.DataFrame]:
    """
    Resample runs with replacement within each analysis cell.
    Skips cells with no runs and resamples that end up empty.
    Returns CI tables keyed by group_cols + ['stimulus_id'].
    """
    rng = np.random.default_rng(random_state)
    w_records, b_records = [], []

    for keys, cell in df.groupby(list(group_cols), sort=False):
        # **drop invalid rows** for this cell
        cell = cell[np.isfinite(cell["correct"]) & np.isfinite(cell["response"])].copy()
        runs = cell["run_id"].unique()
        if runs.size == 0:
            continue  # nothing to do for this cell
        per_run = {rid: g for rid, g in cell.groupby("run_id", sort=False)}

        for b in range(B):
            sampled_runs = rng.choice(runs, size=runs.size, replace=True)
            sampled = [per_run[rid] for rid in sampled_runs]
            if len(sampled) == 0:
                continue  # guard: shouldn't happen, but safe
            boot = pd.concat(sampled, ignore_index=True)
            if boot.empty:
                continue
            # causal prior in within-run order
            boot = boot.sort_values(["run_id", "stimulus_id"]).copy()
            boot["m_prior"] = ewma_prior_within_runs(boot, lam=lam)

            w_pos = position_weights(boot)
            bp_pos = position_prior_pull(boot)

            # w_pos / bp_pos can be empty if all dens are zero after filtering
            if not w_pos.empty:
                for _, row in w_pos.iterrows():
                    rec = {
                        **{
                            c: k
                            for c, k in zip(
                                group_cols,
                                (keys if isinstance(keys, tuple) else (keys,)),
                            )
                        },
                        "stimulus_id": int(row["stimulus_id"]),
                        "w_hat": (
                            float(row["w_hat"]) if np.isfinite(row["w_hat"]) else np.nan
                        ),
                        "boot": b,
                    }
                    w_records.append(rec)
            if not bp_pos.empty:
                for _, row in bp_pos.iterrows():
                    rec = {
                        **{
                            c: k
                            for c, k in zip(
                                group_cols,
                                (keys if isinstance(keys, tuple) else (keys,)),
                            )
                        },
                        "stimulus_id": int(row["stimulus_id"]),
                        "beta_prior": (
                            float(row["beta_prior"])
                            if np.isfinite(row["beta_prior"])
                            else np.nan
                        ),
                        "boot": b,
                    }
                    b_records.append(rec)

    w_df = pd.DataFrame(w_records)
    b_df = pd.DataFrame(b_records)

    def ci(df_in: pd.DataFrame, val_col: str) -> pd.DataFrame:
        if df_in.empty:
            # return an empty frame with expected columns
            return pd.DataFrame(
                columns=list(group_cols) + ["stimulus_id", "q2p5", "q50", "q97p5"]
            )
        return df_in.groupby(list(group_cols) + ["stimulus_id"], as_index=False)[
            val_col
        ].agg(
            q2p5=lambda s: np.nanpercentile(s, 2.5),
            q50=lambda s: np.nanpercentile(s, 50),
            q97p5=lambda s: np.nanpercentile(s, 97.5),
        )

    return {"w_ci": ci(w_df, "w_hat"), "beta_ci": ci(b_df, "beta_prior")}


# ---------- main API ----------
def compute_sequential_weights(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("model", "modality", "ablation", "range_category"),
    lam: float = 0.8,
    bootstrap_B: Optional[int] = 1000,
) -> Dict[str, pd.DataFrame]:
    """
    Computes per-position cue weight (w_hat) and prior-pull (beta) with optional
    block bootstrap CIs, robust to missing runs/positions.
    """
    df = df.copy()

    # **drop invalid rows**
    df = df[np.isfinite(df["correct"]) & np.isfinite(df["response"])].copy()

    # if stimulus_id/run_id missing, try to infer; not needed if you already have them
    if "stimulus_id" not in df.columns and "trial" in df.columns:
        df["stimulus_id"] = (df["trial"] % 10).astype(int)
    if "run_id" not in df.columns and "trial" in df.columns:
        # derive run_id within each analysis cell from trial index blocks of 10
        df["run_id"] = df.groupby(list(group_cols))["trial"].transform(
            lambda s: ((s - s.min()) // 10).astype(int)
        )

    required = {"correct", "response", "run_id", "stimulus_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # df = df.sort_values(["run_id","stimulus_id"]).copy()
    # sort by within-run order and reset index
    df = df.sort_values(["run_id", "stimulus_id"]).reset_index(drop=True)
    df["m_prior"] = ewma_prior_within_runs(df, lam=lam)

    # point estimates per cell
    weight_rows, pull_rows = [], []
    for keys, cell in df.groupby(list(group_cols), sort=False):
        # skip truly empty cells (e.g., all invalid)
        if cell.empty:
            continue
        w_pos = position_weights(cell)
        if not w_pos.empty:
            w_pos[list(group_cols)] = keys if isinstance(keys, tuple) else (keys,)
            weight_rows.append(w_pos)

        bp_pos = position_prior_pull(cell)
        if not bp_pos.empty:
            bp_pos[list(group_cols)] = keys if isinstance(keys, tuple) else (keys,)
            pull_rows.append(bp_pos)

    weights = (
        pd.concat(weight_rows, ignore_index=True)
        if weight_rows
        else pd.DataFrame(
            columns=["stimulus_id", "w_hat", "num", "den", "n_runs", *group_cols]
        )
    )
    prior_pull = (
        pd.concat(pull_rows, ignore_index=True)
        if pull_rows
        else pd.DataFrame(columns=["stimulus_id", "beta_prior", *group_cols])
    )

    out = {"weights": weights, "prior_pull": prior_pull}

    # optional bootstrap CIs
    if bootstrap_B and bootstrap_B > 0:
        cis = block_bootstrap_by_run(
            df, group_cols, lam=lam, B=bootstrap_B, random_state=1
        )
        out["weights_ci"] = cis["w_ci"]
        out["prior_pull_ci"] = cis["beta_ci"]

    return out


def summarize_w(weights_df):
    pos_col = "stimulus_id" if "stimulus_id" in weights_df.columns else "position"
    gcols = [
        c
        for c in ["model", "modality", "ablation", "range_category"]
        if c in weights_df.columns
    ]

    def slope(x, y):
        A = np.c_[x, np.ones_like(x)]
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return m

    # weighted mean by den (more stable)
    w_summary = (
        weights_df.assign(
            den_w=lambda d: d["den"] / d.groupby(gcols)["den"].transform("sum")
        )
        .groupby(gcols, as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "mean_w_hat": np.average(g["w_hat"], weights=g["den_w"]),
                    "slope_w_vs_pos": slope(
                        g[pos_col].to_numpy(), g["w_hat"].to_numpy()
                    ),
                }
            )
        )
        .reset_index(drop=True)
    )
    return w_summary


def filter_tiny_den(weights_df, factor=10):
    gcols = [
        c
        for c in ["model", "modality", "ablation", "range_category"]
        if c in weights_df.columns
    ]
    med_den = weights_df.groupby(gcols)["den"].transform("median")
    keep = weights_df["den"] >= (med_den / factor)
    return weights_df[keep].copy()


# Example usage
# model_name = "openai_gpt-4o-2024-08-06" # "GPT-5 Mini" # "microsoft_phi-4-multimodal-instruct" #'anthropic_claude-3.7-sonnet'
# results_df_list = []
# for exp_name in ['base_image', 'base_3_image', 'base_5_image']:
#     exp_df = pd.read_csv(
#         f"../experiments/text_image/line_length_ratio/runs/{exp_name}/{model_name}/derived/experimental_data.csv"
#     )
#     exp_df.head()
#     exp_df = preprocess_dataframe(exp_df)
#     exp_df["ablation"] = exp_name
#     exp_df["modality"] = "image"
#     exp_df["model"] = model_name
#     results_df = compute_sequential_weights(exp_df)
#     results_df_list.append(results_df)

# w1 = filter_tiny_den(results_df_list[1]['weights'])
# summary_raw = summarize_w(results_df_list[1]['weights'])
# summary_filt = summarize_w(w1)

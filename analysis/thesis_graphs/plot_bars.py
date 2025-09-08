# analysis/thesis_graphs/plot_bars.py
from __future__ import annotations
from importlib import resources
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import ABLATION_LABEL, MODEL_PLOT_ORDER
from .util import pretty_model, rmse


def _order_index_like(idx, order):
    if not order:
        return list(idx)
    present = [m for m in order if m in idx]
    missing = [m for m in idx if m not in present]
    return present + missing


# --- tiny local style loader (same idea as plotters._apply_style) ---
def _apply_style():
    try:
        with resources.as_file(
            resources.files(__package__).joinpath("thesis.mplstyle")
        ) as p:
            plt.style.use(str(p))
    except Exception:
        pass


# ---------- 1) Clustered bars: Bayesian-ness ----------
def plot_bayesness_bars(
    summary_dir: str | Path,
    exp_tag: str,
    save_as: str | Path,
    model_order: list[str] | None = None,
):
    _apply_style()
    model_order = model_order or MODEL_PLOT_ORDER

    summary_dir = Path(summary_dir)
    df = pd.read_csv(summary_dir / f"{exp_tag}_summary_factor_bayesian.csv")

    keep_mod = {"image", "text_image"}
    keep_ab = {"base", "base_3", "base_5"}  # <-- base, constant noise, increasing noise
    d = df[(df["modality"].isin(keep_mod)) & (df["ablation"].isin(keep_ab))].copy()

    key_map = {
        ("image", "base"): "Image — Base",
        ("image", "base_3"): "Image — Constant noise",
        ("image", "base_5"): "Image — Increasing noise",
        ("text_image", "base"): "Text+Image — Base",
        ("text_image", "base_3"): "Text+Image — Constant noise",
        ("text_image", "base_5"): "Text+Image — Increasing noise",
    }
    d["bar_key"] = d[["modality", "ablation"]].apply(tuple, axis=1).map(key_map)

    g = d.pivot_table(
        index="model_pretty", columns="bar_key", values="prob", aggfunc="mean"
    )

    col_order = [v for v in key_map.values() if v in g.columns]
    g = g.reindex(columns=col_order)
    # reorder models using your global order
    g = g.reindex(index=_order_index_like(g.index.tolist(), model_order))

    # --- plot
    fig, ax = plt.subplots(figsize=(10.8, 4.0))
    x = np.arange(len(g.index))
    n = g.shape[1]
    width = min(0.8 / max(n, 1), 0.14)  # 6 bars → a bit narrower

    ymax = float(np.nanmax(g.to_numpy())) if g.size else 1.0
    ax.set_ylim(0, max(1.20, ymax + 0.06))  # more headroom so labels don't clip

    for k, col in enumerate(g.columns):
        xpos = x + (k - (n - 1) / 2) * width
        vals = g[col].values.astype(float)
        ax.bar(xpos, vals, width=width, label=col)
        for xi, yi in zip(xpos, vals):
            if np.isfinite(yi):
                ax.text(
                    xi, yi + 0.02, f"{yi:.2f}", ha="center", va="bottom", fontsize=8
                )

    ax.set_xticks(x, g.index, rotation=20, ha="right")
    ax.set_ylabel("Probability (Bayesian)")
    ax.legend(ncol=3, fontsize=8, loc="upper right", frameon=False)
    ax.set_title("Bayesian-ness by model — Base vs Noise (constant, increasing)")
    fig.tight_layout()
    fig.savefig(save_as, bbox_inches="tight")
    plt.close(fig)


# ---------- 2) Clustered bars: Observed LLM vs Bayes(range) NRMSE ----------
def plot_observed_vs_bayes_bars(
    summary_dir: str | Path,
    exp_tag: str,
    exp_data_path: str | Path,
    save_as: str | Path,
    model_order: list[str] | None = None,
):
    _apply_style()
    model_order = model_order or MODEL_PLOT_ORDER

    summary_dir = Path(summary_dir)

    cue = pd.read_csv(summary_dir / f"{exp_tag}_summary_cue_combo_long.csv")
    bayes_base = (
        cue[(cue["combo_model"] == "Bayes(range)") & (cue["ablation"] == "base")][
            ["llm_model", "model_pretty", "nrmse"]
        ]
        .copy()
        .rename(columns={"nrmse": "NRMSE Bayes(range)"})
    )

    exp = pd.read_csv(exp_data_path)
    mu = exp["correct"].mean()
    baseline_rmse = rmse(mu - exp["correct"])

    obs = (
        exp[exp["exp_id"].str.endswith("_text_image")]
        .assign(
            ablation=lambda d: d["exp_id"].str.extract(r"^(base(?:_\d+)?)_text_image$")
        )
        .query("ablation == 'base'")
        .assign(err=lambda d: d["response"] - d["correct"])
        .groupby("llm_model")["err"]
        .apply(lambda s: rmse(s.to_numpy()) / (baseline_rmse + 1e-12))
        .reset_index()
        .rename(columns={"err": "NRMSE Observed"})
    )
    obs["model_pretty"] = pretty_model(obs["llm_model"])

    M = obs.merge(bayes_base, on=["llm_model", "model_pretty"], how="outer")
    g = M.set_index("model_pretty")[["NRMSE Observed", "NRMSE Bayes(range)"]]
    g = g.reindex(index=_order_index_like(g.index.tolist(), model_order))

    fig, ax = plt.subplots(figsize=(10.8, 4.0))
    x = np.arange(len(g.index))
    width = 0.35

    vals_obs = g["NRMSE Observed"].values.astype(float)
    vals_bays = g["NRMSE Bayes(range)"].values.astype(float)
    ymax = float(np.nanmax(np.vstack([vals_obs, vals_bays]))) if g.size else 1.0
    ax.set_ylim(0, ymax * 1.15)  # headroom

    ax.bar(
        x - width / 2, vals_obs, width=width, label="Observed LLM (text+image — base)"
    )
    ax.bar(x + width / 2, vals_bays, width=width, label="Bayes(range) combo (base)")

    for xi, yi in zip(x - width / 2, vals_obs):
        if np.isfinite(yi):
            ax.text(
                xi, yi + ymax * 0.02, f"{yi:.2f}", ha="center", va="bottom", fontsize=8
            )
    for xi, yi in zip(x + width / 2, vals_bays):
        if np.isfinite(yi):
            ax.text(
                xi, yi + ymax * 0.02, f"{yi:.2f}", ha="center", va="bottom", fontsize=8
            )

    ax.set_xticks(x, g.index, rotation=20, ha="right")
    ax.set_ylabel("NRMSE (↓ better)")
    ax.legend(ncol=2, fontsize=8, loc="upper right", frameon=False)
    ax.set_title("Observed LLM vs Bayes(range) — NRMSE (Base case, Text+Image)")
    fig.tight_layout()
    fig.savefig(save_as, bbox_inches="tight")
    plt.close(fig)


def _order_index_like(idx, order):
    if not order:
        return list(idx)
    present = [m for m in order if m in idx]
    missing = [m for m in idx if m not in present]
    return present + missing


def plot_observed_vs_bayes_bars_two_rows(
    summary_dir: str | Path,
    exp_tag: str,
    exp_data_path: str | Path,
    save_as: str | Path,
    model_order: list[str] | None = None,
):
    """
    Row 1: two bars per model (Observed vs Bayes(range)) using the configured model order.
    Row 2: single bar per model, showing Bayes(range) as % of Observed, sorted (best → worst).
    """
    _apply_style()
    model_order = model_order or MODEL_PLOT_ORDER

    summary_dir = Path(summary_dir)

    # --- Bayes(range) from cue summary
    cue = pd.read_csv(summary_dir / f"{exp_tag}_summary_cue_combo_long.csv")
    bayes_base = (
        cue[(cue["combo_model"] == "Bayes(range)") & (cue["ablation"] == "base")][
            ["llm_model", "model_pretty", "nrmse"]
        ]
        .copy()
        .rename(columns={"nrmse": "NRMSE Bayes(range)"})
    )

    # --- Observed LLM on text+image@base from long-form data
    exp = pd.read_csv(exp_data_path)
    mu = exp["correct"].mean()
    baseline_rmse = rmse(mu - exp["correct"])

    obs = (
        exp[exp["exp_id"].str.endswith("_text_image")]
        .assign(
            ablation=lambda d: d["exp_id"].str.extract(r"^(base(?:_\d+)?)_text_image$")
        )
        .query("ablation == 'base'")
        .dropna(subset=["response", "correct"])
        .assign(err=lambda d: d["response"] - d["correct"])
        .groupby("llm_model")["err"]
        .apply(lambda s: rmse(s.to_numpy()) / (baseline_rmse + 1e-12))
        .reset_index()
        .rename(columns={"err": "NRMSE Observed"})
    )
    obs["model_pretty"] = pretty_model(obs["llm_model"])

    # import pdb

    # pdb.set_trace()  # Debugging breakpoint

    # --- combine
    M = obs.merge(bayes_base, on=["llm_model", "model_pretty"], how="outer")
    g = M.set_index("model_pretty")[["NRMSE Observed", "NRMSE Bayes(range)"]].astype(
        float
    )

    # ---------------- Row 1 (absolute NRMSEs, ordered by complexity) ----------------
    g1 = g.reindex(index=_order_index_like(g.index.tolist(), model_order))
    x1 = np.arange(len(g1.index))
    width = 0.35

    # figure
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.8, 6.8), sharex=False, gridspec_kw=dict(hspace=0.35)
    )

    vals_obs = g1["NRMSE Observed"].to_numpy()
    vals_bays = g1["NRMSE Bayes(range)"].to_numpy()
    vmax1 = np.nanmax(np.vstack([vals_obs, vals_bays])) if g1.size else 1.0
    ax1.set_ylim(0, vmax1 * 1.15)

    ax1.bar(
        x1 - width / 2, vals_obs, width=width, label="Observed LLM (text+image — base)"
    )
    ax1.bar(x1 + width / 2, vals_bays, width=width, label="Bayes(range) combo (base)")

    for xi, yi in zip(x1 - width / 2, vals_obs):
        if np.isfinite(yi):
            ax1.text(
                xi, yi + vmax1 * 0.02, f"{yi:.2f}", ha="center", va="bottom", fontsize=8
            )
    for xi, yi in zip(x1 + width / 2, vals_bays):
        if np.isfinite(yi):
            ax1.text(
                xi, yi + vmax1 * 0.02, f"{yi:.2f}", ha="center", va="bottom", fontsize=8
            )

    ax1.set_xticks(x1, g1.index, rotation=20, ha="right")
    ax1.set_ylabel("NRMSE (↓ better)")
    ax1.legend(ncol=2, fontsize=8, loc="upper right", frameon=False)
    ax1.set_title("Observed vs Bayes(range) — NRMSE (Base, Text+Image)")

    # ---------------- Row 2 (percentage & ranking, sorted by improvement) ----------------
    ratio = 100.0 * (g["NRMSE Bayes(range)"] / g["NRMSE Observed"])
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()

    # sort best → worst (lower % is better)
    order2 = ratio.sort_values().index.tolist()
    r2 = ratio.reindex(order2)
    x2 = np.arange(len(r2))

    vmax2 = float(np.nanmax(r2.to_numpy())) if r2.size else 100.0
    ymax2 = max(110.0, vmax2 * 1.15)
    ax2.set_ylim(0, ymax2)

    bars = ax2.bar(x2, r2.to_numpy(), width=0.6, label="Bayes(range) as % of Observed")
    # add a reference line at 100% (tie)
    ax2.axhline(100.0, linestyle="--", alpha=0.6)

    # annotate with percentage and rank
    for rank, (xi, yi) in enumerate(zip(x2, r2.to_numpy()), start=1):
        if np.isfinite(yi):
            ax2.text(
                xi,
                yi + ymax2 * 0.02,
                f"{yi:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax2.text(
                xi, 0, f"#{rank}", ha="center", va="bottom", fontsize=8, rotation=0
            )

    ax2.set_xticks(x2, order2, rotation=20, ha="right")
    ax2.set_ylabel("Bayes as % of Observed")
    ax2.set_title("Ranking by Bayes(range) / Observed (Base, Text+Image)")

    fig.tight_layout()
    fig.savefig(save_as, bbox_inches="tight")
    plt.close(fig)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Optional
import pandas as pd
import plotly.express as px
from analysis.thesis_graphs.config import MODEL_NAME

LLM_ORDER = [
    "GPT 5 Mini",
    "GPT 4o",
    "Claude 3.7 Sonnet",
    "Llama 4 Maverick",
    "Gemini 2.5 Flash Lite",
    "Mistral 24B",
    "Phi 4 Multimodal",
    "Gemma 3 4B it",
    "Qwen2.5 VL 32B",
]

COLORS = {
    "Bayesian": "#1f77b4",
    "Weber": "#ff7f0e",
    "Sequential": "#2ca02c",
    "NRMSE": "#9467bd",
    "EmpiricalLinear": "#1f77b4",
    "Bayes(range)": "#ff7f0e",
    "Bayes(stimulus)": "#2ca02c",
    "Equal": "#d62728",
    "ΔNRMSE": "#9467bd",
}


def _wrap_labels(labels, max_width=12):
    out = []
    for lab in labels:
        parts = lab.split()
        lines, line = [], ""
        for p in parts:
            if len(line) + (1 if line else 0) + len(p) <= max_width:
                line = (line + " " + p) if line else p
            else:
                lines.append(line)
                line = p
        if line:
            lines.append(line)
        out.append("\n".join(lines))
    return out


def _set_xtick_models(ax, models, max_width=8):
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(
        _wrap_labels(models, max_width=max_width), rotation=0, ha="center", fontsize=8
    )


def _annotate_bars(ax, bars, fmt="{:.1f}", y_offset_frac=0.02):
    ymax = ax.get_ylim()[1]
    y_offset = max(y_offset_frac * ymax, 0.02)
    for rect in bars:
        h = rect.get_height()
        if h is None or np.isnan(h):
            continue
        ax.annotate(
            fmt.format(h),
            (rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _cluster_positions(n_series, n_models, cluster_width=0.82):
    """
    Return centers (0..n_models-1) and per-series offsets so that bars within a model cluster
    are contiguous (no gaps), and there are visible gaps between clusters.
    """
    centers = np.arange(n_models)  # cluster centers at integers
    bar_w = cluster_width / n_series
    # offsets so bars fill the cluster: [-W/2 + (i+0.5)w]
    offsets = (-cluster_width / 2) + (np.arange(n_series) + 0.5) * bar_w
    return centers, offsets, bar_w


# --- Plots with contiguous clusters ---
def plot_behavior_bars_clustered(df, *, title, savepath=None, figsize=(7.4, 2.5)):
    models = df["model_pretty"].tolist()
    centers, offsets, bar_w = _cluster_positions(
        n_series=4, n_models=len(models), cluster_width=0.86
    )
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    # LHS: three probs
    b_b = ax.bar(
        centers + offsets[0],
        df["bayes"],
        bar_w,
        label="Bayesian",
        color=COLORS["Bayesian"],
    )
    b_w = ax.bar(
        centers + offsets[1], df["weber"], bar_w, label="Weber", color=COLORS["Weber"]
    )
    b_s = ax.bar(
        centers + offsets[2],
        df["sequential"],
        bar_w,
        label="Sequential",
        color=COLORS["Sequential"],
    )
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Probability / Weight")
    # RHS: NRMSE aligned as 4th bar in cluster
    ax2 = ax.twinx()
    b_e = ax2.bar(
        centers + offsets[3], df["nrmse"], bar_w, label="NRMSE", color=COLORS["NRMSE"]
    )
    ax2.set_ylim(0, 1.5)
    ax2.set_ylabel("NRMSE")

    _set_xtick_models(ax, models, max_width=8)
    ax.set_title(title)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, ncol=4, frameon=False, fontsize=8, loc="upper left")

    for bars in [b_b, b_w, b_s]:
        _annotate_bars(ax, bars, "{:.1f}")
    _annotate_bars(ax2, b_e, "{:.1f}")

    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


def plot_cue_fit_bars_clustered(df, *, title, savepath=None, figsize=(7.4, 2.5)):
    # import pdb

    # pdb.set_trace()
    df = pd.read_parquet(df)
    models = [MODEL_NAME[model] for model in df["LLM_model"].tolist()]
    centers, offsets, bar_w = _cluster_positions(
        n_series=5, n_models=len(models), cluster_width=0.86
    )
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    b1 = ax.bar(
        centers + offsets[0],
        df["prob_EmpiricalLinear"],
        bar_w,
        label="EmpiricalLinear",
        color=COLORS["EmpiricalLinear"],
    )
    b2 = ax.bar(
        centers + offsets[1],
        df["prob_Bayes_range"],
        bar_w,
        label="Bayes(range)",
        color=COLORS["Bayes(range)"],
    )
    b3 = ax.bar(
        centers + offsets[2],
        df["prob_Bayes_stimulus"],
        bar_w,
        label="Bayes(stimulus)",
        color=COLORS["Bayes(stimulus)"],
    )
    b4 = ax.bar(
        centers + offsets[3], df["prob_Equal"], bar_w, label="Equal", color="#8c564b"
    )
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Fit Probability")
    ax2 = ax.twinx()
    b5 = ax2.bar(
        centers + offsets[4],
        df["nrmse"],
        bar_w,
        label="ΔNRMSE vs Oracle",
        color=COLORS["ΔNRMSE"],
    )
    ax2.set_ylim(0, 1.5)
    ax2.set_ylabel("ΔNRMSE vs Oracle")

    _set_xtick_models(ax, models, max_width=8)
    ax.set_title(title)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, ncol=5, frameon=False, fontsize=8, loc="upper left")

    for bars in [b1, b2, b3, b4]:
        _annotate_bars(ax, bars, "{:.1f}")
    _annotate_bars(ax2, b5, "{:.1f}")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


def plot_cue_weight_bars(df, *, title, savepath=None, figsize=(7.4, 2.5)):
    models = df["model_pretty"].tolist()
    centers = np.arange(len(models))
    bar_w = 0.6
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    bars = ax.bar(
        centers, df["w_img"], bar_w, label="Image weight (empirical)", color="#1f77b4"
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Image Weight (0..1)")
    _set_xtick_models(ax, models, max_width=8)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _annotate_bars(ax, bars, "{:.1f}")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


def _infer_modality_from_exp_id(exp_id: str) -> str:
    eid = str(exp_id).lower()
    if "text_image" in eid or "text+image" in eid or "multi" in eid:
        return "text+image"
    if "image" in eid:
        return "image"
    if "text" in eid:
        return "text"
    return "unknown"


def _range_colors():
    # Use default color cycle indices to keep distinct but consistent across subplots
    # We'll just return indices for short/medium/long to rely on MPL defaults.
    return {"short": 0, "medium": 1, "long": 2}


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


def _infer_modality_from_exp_id(exp_id: str) -> str:
    e = str(exp_id).lower()
    if "text_image" in e or "text+image" in e or "multi" in e:
        return "text_image"
    if "image" in e:
        return "image"
    if "text" in e:
        return "text"
    return "unknown"


def scatter_grid_3x3(
    csv_path: str,
    exp_id: str,
    models: Optional[
        List[str]
    ] = None,  # e.g. ["GPT-5 Mini", "GPT 4o", "Claude 3.7 Sonnet"]
    modalities: Tuple[str, str, str] = ("text", "image", "text_image"),
    savepath: Optional[str] = None,
    figsize: Tuple[float, float] = (9.0, 9.0),
):
    """
    Draw a 3x3 grid of scatter plots:
      rows   = modalities (text, image, text_image)
      cols   = chosen models (3 of them)
    """

    sel_cols = [
        "correct",
        "response",
        "range_category",
        "stimulus_id",
        "llm_model",
        "exp_id",
    ]

    df = pd.read_csv(csv_path, usecols=sel_cols)

    df["modality"] = df["exp_id"].apply(_infer_modality_from_exp_id)

    # Fixed range order
    range_order = ["short", "medium", "long"]

    fig, axes = plt.subplots(3, 3, figsize=figsize, dpi=150, sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(3, 3)

    legend_drawn = False

    for r, mod in enumerate(modalities):
        for c, model in enumerate(models):
            ax = axes[r, c]
            g_all = df[
                (df["modality"] == mod)
                & (df["llm_model"] == model)
                & (df["exp_id"] == exp_id)
            ].copy()

            # panel title on the top row; row label on left
            if r == 0:
                ax.set_title(model, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"Response\n({mod})", fontsize=9)

            # plot per-range
            for rng in range_order:
                g = g_all[g_all["range_category"] == rng]
                if g.empty:
                    continue

                # Choose a base color for this range
                color_idx = range_order.index(rng)
                base_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][
                    color_idx
                ]

                # raw points (light/small)
                ax.scatter(
                    g["correct"], g["response"], s=8, alpha=0.1, label=f"{rng} (raw)"
                )

                # per-stimulus averages
                avg = g.groupby(["range_category", "stimulus_id"], as_index=False).agg(
                    correct=("correct", "mean"), response=("response", "mean")
                )
                ax.scatter(
                    avg["correct"],
                    avg["response"],
                    s=30,
                    alpha=0.9,
                    color=base_color,
                    label=f"{rng} (avg)",
                )

                # # linear fit on averages
                # if len(avg) >= 2:
                #     x = avg["correct"].to_numpy()
                #     y = avg["response"].to_numpy()
                #     m, b = np.polyfit(x, y, 1)
                #     xx = np.linspace(0, 1, 50)
                #     ax.plot(xx, m * xx + b, linewidth=1.5, label=f"{rng} fit")

            # axes + grid
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            if r == 2:
                ax.set_xlabel("Correct (ground truth)", fontsize=9)

            # one legend for the whole figure (from the first populated panel)
            # if not legend_drawn and not g_all.empty:
            #     ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper left")
            #     legend_drawn = True

    fig.suptitle(
        "Psychophysics Scatter — by Modality (rows) × Model (cols)",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


# ---------- Demo with your files (ablation='base') ----------
exp_name = "maze_distance"  # "line_length_ratio"
exp_modality = "text_image"  # "text"
base = f"experiments/{exp_modality}/{exp_name}/artifacts"

nrmse_fp = f"{base}/{exp_name}_summary_nrmse_observed.csv"
bayes_fp = f"{base}/{exp_name}_summary_factor_bayesian.csv"
weber_fp = f"{base}/{exp_name}_summary_factor_weber.csv"
seq_fp = f"{base}/{exp_name}_summary_factor_sequential.csv"
cue_fp = f"{base}/overall_cue_combo.parquet"
scatter_fp = f"{base}/{exp_name}/experimental_data.csv"

os.makedirs(f"temp/{exp_name}", exist_ok=True)
save_path = f"temp/{exp_name}"


fig = plot_cue_fit_bars_clustered(
    cue_fp,
    title="Cue Combination Model Fit",
    savepath=f"{save_path}/cue_fit_quality_base.pdf",
)
plt.close(fig)

fig = plot_cue_weight_bars(
    cue_fp,
    title="Cue Combination — Image Weight (text+image, base)",
    savepath=f"{save_path}/cue_weights_base.pdf",
)
plt.close(fig)

# Scatter graphs
fig = scatter_grid_3x3(
    scatter_fp,
    exp_id="base_text_image",
    models=["GPT-5 Mini", "openai_gpt-4o-2024-08-06", "anthropic_claude-3.7-sonnet"],
    savepath=f"{save_path}/scatter_row_demo.pdf",
)
plt.close(fig)

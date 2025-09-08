# analysis/thesis_graphs/plot_scatter.py
from __future__ import annotations
from importlib import resources
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .util import pretty_model
from .config import MODEL_PLOT_ORDER


def _apply_style():
    try:
        with resources.as_file(
            resources.files(__package__).joinpath("thesis.mplstyle")
        ) as p:
            plt.style.use(str(p))
    except Exception:
        pass


def plot_nrmse_vs_bayes(summary_dir: str | Path, exp_tag: str, save_as: str | Path):
    """
    For each LLM and each modality, compute:
      x = mean Bayesian-ness across ablations
      y = mean observed NRMSE across ablations
    Then scatter: color by modality, 3 dots per model.
    """
    _apply_style()
    summary_dir = Path(summary_dir)

    # load summaries
    bay = pd.read_csv(summary_dir / f"{exp_tag}_summary_factor_bayesian.csv")
    nr = pd.read_csv(summary_dir / f"{exp_tag}_summary_nrmse_observed.csv")

    # average across ablations per (model, modality)
    bx = (
        bay.groupby(["llm_model", "model_pretty", "modality"], as_index=False)["prob"]
        .mean()
        .rename(columns={"prob": "bayesianess"})
    )
    ny = (
        nr.groupby(["llm_model", "model_pretty", "modality"], as_index=False)["nrmse"]
        .mean()
        .rename(columns={"nrmse": "nrmse"})
    )

    df = bx.merge(ny, on=["llm_model", "model_pretty", "modality"], how="inner")

    # consistent model ordering (optional for legend grouping)
    order = MODEL_PLOT_ORDER or []
    df["model_pretty"] = pd.Categorical(
        df["model_pretty"],
        categories=order + [m for m in df["model_pretty"].unique() if m not in order],
        ordered=True,
    )
    df = df.sort_values(["model_pretty", "modality"])

    # colors per modality
    color_map = {"text": "#1B9E77", "image": "#D95F02", "text_image": "#7570B3"}

    fig, ax = plt.subplots(figsize=(7.8, 5.4), constrained_layout=True)

    # limits with headroom
    xvals = df["bayesianess"].to_numpy(dtype=float)
    yvals = df["nrmse"].to_numpy(dtype=float)
    if np.isfinite(xvals).any():
        xmin, xmax = np.nanmin(xvals), np.nanmax(xvals)
    else:
        xmin, xmax = 0.0, 1.0
    if np.isfinite(yvals).any():
        ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
    else:
        ymin, ymax = 0.0, 1.0

    xr = xmax - xmin if np.isfinite(xmax - xmin) else 1.0
    yr = ymax - ymin if np.isfinite(ymax - ymin) else 1.0
    ax.set_xlim(
        max(0.0, xmin - 0.05 * xr), min(1.15, xmax + 0.05 * xr)
    )  # bayesianess usually in [0,1], allow a bit >1
    ax.set_ylim(max(0.0, ymin - 0.05 * yr), ymax + 0.08 * yr)

    # plot per modality
    for mod, sub in df.groupby("modality", sort=False):
        ax.scatter(
            sub["bayesianess"],
            sub["nrmse"],
            s=60,
            alpha=0.85,
            edgecolors="white",
            linewidths=1.0,
            label=mod.replace("_", "+"),
            color=color_map.get(mod, None),
        )
        # light labels near points (optional; comment out if too busy)
        for _, r in sub.iterrows():
            ax.text(
                r["bayesianess"],
                r["nrmse"] + 0.01 * yr,
                str(r["model_pretty"]),
                fontsize=7,
                ha="center",
                va="bottom",
                alpha=0.75,
            )

    ax.set_xlabel("Bayesian-ness (mean probability across ablations)")
    ax.set_ylabel("Observed NRMSE (mean across ablations)  ↓ better")
    ax.legend(title="Modality", frameon=False)
    ax.set_title("NRMSE vs Bayesian-ness (one dot per model × modality)")
    fig.savefig(save_as, bbox_inches="tight")
    plt.close(fig)


def plot_ratio_scatter_nrmse_vs_bayes(
    summary_dir: str | Path,
    exp_tag: str,
    save_as: str | Path,
    include_base: bool = False,
):
    """
    For each (llm_model, modality):
      x = mean_{ablations}  [ Bayes_prob / Bayes_prob(base) ]
      y = mean_{ablations}  [ NRMSE / NRMSE(base) ]
    By default we exclude the base ablation from the mean.
    """
    _apply_style()
    summary_dir = Path(summary_dir)
    eps = 1e-12

    # load summaries
    bay = pd.read_csv(summary_dir / f"{exp_tag}_summary_factor_bayesian.csv")
    nr = pd.read_csv(summary_dir / f"{exp_tag}_summary_nrmse_observed.csv")

    # --- Bayes ratios to base
    b = bay[["llm_model", "model_pretty", "modality", "ablation", "prob"]].copy()
    base_b = b[b["ablation"] == "base"][["llm_model", "modality", "prob"]].rename(
        columns={"prob": "prob_base"}
    )
    b = b.merge(base_b, on=["llm_model", "modality"], how="left")
    b["bayes_ratio"] = b["prob"] / (b["prob_base"] + eps)

    # --- NRMSE ratios to base
    n = nr[["llm_model", "model_pretty", "modality", "ablation", "nrmse"]].copy()
    base_n = n[n["ablation"] == "base"][["llm_model", "modality", "nrmse"]].rename(
        columns={"nrmse": "nrmse_base"}
    )
    n = n.merge(base_n, on=["llm_model", "modality"], how="left")
    n["nrmse_ratio"] = n["nrmse"] / (n["nrmse_base"] + eps)

    # choose ablations to average over
    if include_base:
        bx = b
        nx = n
    else:
        bx = b[b["ablation"] != "base"]
        nx = n[n["ablation"] != "base"]

    # aggregate to one point per (model, modality)
    bx_agg = bx.groupby(["llm_model", "model_pretty", "modality"], as_index=False)[
        "bayes_ratio"
    ].mean()
    nx_agg = nx.groupby(["llm_model", "model_pretty", "modality"], as_index=False)[
        "nrmse_ratio"
    ].mean()

    df = bx_agg.merge(nx_agg, on=["llm_model", "model_pretty", "modality"], how="inner")

    # modality colors
    color_map = {"text": "#1B9E77", "image": "#D95F02", "text_image": "#7570B3"}

    fig, ax = plt.subplots(figsize=(7.8, 5.4), constrained_layout=True)

    # reference lines at 1.0
    ax.axvline(1.0, linestyle="--", alpha=0.5)
    ax.axhline(1.0, linestyle="--", alpha=0.5)

    # limits with headroom
    xv = df["bayes_ratio"].to_numpy(float)
    yv = df["nrmse_ratio"].to_numpy(float)

    def span(vals, pad=0.08, lo_clip=0.0):
        if not np.isfinite(vals).any():
            return (0.9, 1.1)
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        rng = max(1e-6, vmax - vmin)
        lo = max(lo_clip, vmin - pad * rng)
        hi = vmax + pad * rng
        return lo, hi

    ax.set_xlim(*span(xv, pad=0.10, lo_clip=0.0))
    ax.set_ylim(*span(yv, pad=0.10, lo_clip=0.0))

    # plot points per modality
    for mod, sub in df.groupby("modality", sort=False):
        ax.scatter(
            sub["bayes_ratio"],
            sub["nrmse_ratio"],
            s=60,
            alpha=0.9,
            edgecolors="white",
            linewidths=1.0,
            label=mod.replace("_", "+"),
            color=color_map.get(mod, None),
        )
        # optional tiny labels
        for _, r in sub.iterrows():
            ax.text(
                r["bayes_ratio"],
                r["nrmse_ratio"] + 0.01,
                str(r["model_pretty"]),
                fontsize=7,
                ha="center",
                va="bottom",
                alpha=0.75,
            )

    ax.set_xlabel("Bayesian-ness / Base (mean over ablations)")
    ax.set_ylabel("NRMSE / Base (mean over ablations)  ↓ better")
    ax.legend(title="Modality", frameon=False)
    ax.set_title(
        "Relative to Base: NRMSE vs Bayesian-ness (one dot per model × modality)"
    )
    fig.savefig(save_as, bbox_inches="tight")
    plt.close(fig)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re


def _parse_exp(exp_id: str):
    s = str(exp_id)
    for suf in ("_text_image", "_image", "_text"):
        if s.endswith(suf):
            return s[: -len(suf)], suf.lstrip("_")  # (ablation_core, modality)
    # fallback: no suffix found
    return s, None


def plot_response_vs_stimulus_grid_all_ablations(
    exp_df: pd.DataFrame,
    model_pretty_list=(
        "GPT 5 Mini",
        "GPT 4o",
        "Claude 3.7 Sonnet",
        "Llama-4 Maverick",
        "Gemini 2.5 Flash-Lite",
        "Mistral 24B",
        "Phi 4 Multimodal",
        "Gemma 3 4B-it",
        "Qwen2.5 VL 32B",
    ),
    ablation_list=None,  # now interpreted as ablation_core names (no modality)
    save_dir="scatter_plots",
    groupby_list=None,
    range_colors=None,
):
    """
    Rows  = ablations (exp_id *without* modality suffix)
    Cols  = modalities [text, image, text_image]
    Subplot title = full exp_id for that cell: "<ablation_core>_<modality>"
    One figure per model in model_pretty_list.
    """
    color_map = range_colors or {
        "short": "#D95F02",
        "medium": "#1B9E77",
        "long": "#7570B3",
    }

    d = exp_df.copy()
    model_col = "model_pretty"
    if "model_pretty" not in d.columns:
        d["model_pretty"] = pretty_model(d["llm_model"])

    # Parse exp_id -> ablation_core, modality (unless a real 'modality' column exists)
    if "exp_id" not in d.columns:
        raise KeyError("exp_df needs 'exp_id' for ablation rows.")
    parsed = d["exp_id"].astype(str).map(_parse_exp)
    d["ablation_core"] = [a for a, m in parsed]
    if "modality" in d.columns:
        d["__modality__"] = (
            d["modality"]
            .astype(str)
            .str.lower()
            .str.replace("+", "_", regex=False)
            .str.replace(" ", "", regex=False)
        )
    else:
        d["__modality__"] = [m for a, m in parsed]

    # Guard: keep only known modalities
    d = d[d["__modality__"].isin(["text", "image", "text_image"])].copy()

    # Which ablations to show (by core name)
    if ablation_list is None:
        ablation_list = sorted(d["ablation_core"].unique().tolist())

    # Grouping for per-stimulus mean
    if groupby_list is None:
        groupby_list = (
            ["range_category", "stimulus_id"]
            if "stimulus_id" in d.columns
            else ["range_category", "correct"]
        )

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    modalities = [("text", "Text"), ("image", "Image"), ("text_image", "Text+Image")]

    def _limits(dfm):
        if dfm.empty:
            return 0.0, 1.0
        vals = (
            dfm[["correct"]].to_numpy().ravel()
        )  # , "response" (if some responses are too high we ignore them)
        vmax = float(np.nanmax(vals)) if vals.size else 1.0
        return 0.0, max(vmax * 1.05, 1e-3)

    # import pdb

    # pdb.set_trace()  # Debugging

    for mp in model_pretty_list:
        dd = d[d[model_col].astype(str) == str(mp)].copy()
        if dd.empty:
            print(f"[warn] no data for model '{mp}', skipping.")
            continue

        n_rows, n_cols = len(ablation_list), 3
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.6 * n_cols, 2.9 * n_rows),
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        if n_rows == 1:
            axes = np.array([axes])

        xmin, vmax = _limits(dd)

        for i, ab_core in enumerate(ablation_list):
            dd_ab = dd[dd["ablation_core"] == ab_core]
            for j, (mod_key, mod_title) in enumerate(modalities):
                ax = axes[i, j]
                if i == 0:
                    ax.annotate(
                        mod_title,
                        xy=(0.5, 1.10),
                        xycoords="axes fraction",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                    )

                # Title = full exp_id string for that cell
                ax.set_title(f"{ab_core}_{mod_key}", fontsize=9)

                cell = dd_ab[dd_ab["__modality__"] == mod_key]
                if cell.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", alpha=0.7)
                    ax.plot(
                        [xmin, vmax],
                        [xmin, vmax],
                        "--",
                        color="grey",
                        alpha=0.7,
                        lw=1.0,
                    )
                    ax.set_xlim(xmin, vmax)
                    ax.set_ylim(xmin, vmax)
                    ax.set_aspect("equal")
                    if i == n_rows - 1:
                        ax.set_xlabel("Stimulus (truth)")
                    if j == 0:
                        ax.set_ylabel("Response")
                    continue

                for rng, g in cell.groupby("range_category", sort=False):
                    base_color = color_map.get(str(rng), "#999999")
                    ax.scatter(
                        g["correct"],
                        g["response"],
                        s=10,
                        alpha=0.18,
                        color=base_color,
                        linewidths=0,
                    )
                    means = (
                        g.groupby(groupby_list)[["correct", "response"]]
                        .mean()
                        .reset_index(drop=True)
                    )
                    ax.scatter(
                        means["correct"],
                        means["response"],
                        s=58,
                        alpha=1.0,
                        color=base_color,
                        edgecolors="white",
                        linewidths=1.2,
                        zorder=3,
                    )

                ax.plot(
                    [xmin, vmax], [xmin, vmax], "--", color="grey", alpha=0.7, lw=1.0
                )
                ax.set_xlim(xmin, vmax)
                ax.set_ylim(xmin, vmax)
                ax.set_aspect("equal")
                if i == n_rows - 1:
                    ax.set_xlabel("Stimulus (truth)")
                if j == 0:
                    ax.set_ylabel("Response")

        # Legend
        handles = labels = None
        for i in range(n_rows):
            for j in range(n_cols):
                h, l = axes[i, j].get_legend_handles_labels()
                if h:
                    handles, labels = h[:3], l[:3]
                    break
            if handles:
                break
        if handles:
            fig.legend(handles, labels, title="Range", loc="upper right", frameon=False)

        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(mp))
        out = Path(save_dir) / f"{safe}_scatter_grid.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] saved {out}")


def plot_response_vs_stimulus_grid_base_two_models(
    exp_df: pd.DataFrame,
    model_pretty_list=("GPT 4o", "Claude 3.7 Sonnet"),
    save_as=None,
    groupby_list: list[str] | None = None,
    range_colors: dict[str, str] | None = None,
):
    """
    2 × 3 grid:
      rows   = the two models (pretty names)
      cols   = modalities [text, image, text+image]
      ablation = base only
    Each panel: raw scatter (small, transparent) + per-stimulus mean (bigger, white edge).
    """
    _apply_style()

    # color scheme from config (fallback to your default if not present)
    try:
        from .config import RANGE_COLORS as CFG_RANGE_COLORS
    except Exception:
        CFG_RANGE_COLORS = None
    color_map = (
        range_colors
        or CFG_RANGE_COLORS
        or {
            "short": "#D95F02",
            "medium": "#1B9E77",
            "long": "#7570B3",
        }
    )

    d = exp_df.copy()
    # build a mapping llm_model -> pretty
    if "model_pretty" not in d.columns:
        d["model_pretty"] = pretty_model(d["llm_model"])

    # import pdb

    # pdb.set_trace()  # Debugging breakpoint
    wanted = list(model_pretty_list)
    d = d[d["model_pretty"].isin(wanted)]

    # helper to select base rows for a modality
    def sel_mod(df, mod):
        return df[df["exp_id"] == f"base_{mod}"].copy()

    panels = [("text", "Text"), ("image", "Image"), ("text_image", "Text+Image")]
    groups = [
        (mp, sel_mod(d[d["model_pretty"] == mp], mod), mod, mod_title)
        for mp in wanted
        for (mod, mod_title) in panels
    ]

    # global axes (start at 0)
    vals = []
    for _, dfm, _, _ in groups:
        if not dfm.empty:
            vals.append(dfm[["correct", "response"]].to_numpy().ravel())
    vmax = float(np.nanmax(np.concatenate(vals))) if vals else 1.0
    vmax *= 1.05
    xmin = ymin = 0.0

    # default mean grouping
    if groupby_list is None:
        groupby_list = (
            ["range_category", "stimulus_id"]
            if "stimulus_id" in d.columns
            else ["range_category", "correct"]
        )

    fig, axes = plt.subplots(
        2, 3, figsize=(12.8, 7.6), constrained_layout=True, sharex=True, sharey=True
    )

    for ax, (mp, dfm, mod, mod_title) in zip(axes.ravel(), groups):
        ax.set_title(f"{mp} — {mod_title}")

        if dfm.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", alpha=0.7)
            ax.set_xlim(xmin, vmax)
            ax.set_ylim(ymin, vmax)
            ax.plot([0, vmax], [0, vmax], "--", color="grey", alpha=0.7, linewidth=1.0)
            ax.set_aspect("equal", adjustable="box")
            continue

        for rng, g in dfm.groupby("range_category", sort=False):
            base_color = color_map.get(str(rng), "#999999")
            # raw points
            ax.scatter(
                g["correct"],
                g["response"],
                s=10,
                alpha=0.20,
                color=base_color,
                linewidths=0,
                label=None,
            )
            # means
            means = (
                g.groupby(groupby_list)[["correct", "response"]]
                .mean()
                .reset_index(drop=True)
            )
            ax.scatter(
                means["correct"],
                means["response"],
                s=60,
                alpha=1.0,
                color=base_color,
                edgecolors="white",
                linewidths=1.5,
                zorder=3,
                label=str(rng),
            )

        # y=x
        ax.plot(
            [xmin, vmax], [xmin, vmax], "--", color="grey", alpha=0.7, linewidth=1.0
        )
        ax.set_xlim(xmin, vmax)
        ax.set_ylim(ymin, vmax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Stimulus (truth)")

    axes[0, 0].set_ylabel("Response")
    axes[1, 0].set_ylabel("Response")

    # single legend (range)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles[:3], labels[:3], title="Range", loc="upper right", frameon=False
        )

    if save_as:
        fig.savefig(save_as, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

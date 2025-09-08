# analysis/thesis_graphs/plotters.py
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from .config import ABLATION_ORDER, ABLATION_LABEL, MODEL_PLOT_ORDER, MPL_STYLE
from importlib import resources


def _apply_style():
    if not MPL_STYLE:
        return
    try:
        with resources.as_file(resources.files(__package__).joinpath(MPL_STYLE)) as p:
            plt.style.use(str(p))
    except Exception:
        pass


def _order_index_like(idx, order):
    if not order:
        return list(idx)
    present = [m for m in order if m in idx]
    missing = [m for m in idx if m not in present]
    return present + missing


def _grid(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot to model x ablation with robust ordering and fallbacks."""
    if df is None or df.empty:
        return pd.DataFrame()
    t = df.copy()
    if "ablation_pretty" in t.columns:
        t["Ablation"] = t["ablation_pretty"]
    else:
        t["Ablation"] = t["ablation"].map(ABLATION_LABEL).fillna(t["ablation"])
    if "model_pretty" not in t.columns:
        t["model_pretty"] = t.get("llm_model", "model")

    g = t.pivot_table(
        index="model_pretty", columns="Ablation", values=value_col, aggfunc="mean"
    ).apply(pd.to_numeric, errors="coerce")

    # enforce ablation column order (keep only those present)
    desired_cols = [
        ABLATION_LABEL[a] for a in ABLATION_ORDER if ABLATION_LABEL[a] in g.columns
    ]
    if desired_cols:
        g = g.reindex(columns=desired_cols)

    # model row order: LARGE → SMALL (reverse of MODEL_PLOT_ORDER)
    large_to_small = list(reversed(MODEL_PLOT_ORDER or []))
    g = g.reindex(index=_order_index_like(g.index.tolist(), large_to_small))

    # drop all-NaN rows/cols
    g = g.dropna(how="all").dropna(axis=1, how="all")
    return g


def _annotated_heat(ax, grid: pd.DataFrame, norm: Normalize, cmap, fmt="%.2f"):
    M = grid.to_numpy()
    im = ax.imshow(M, norm=norm, cmap=cmap)
    ax.set_xticks(range(grid.shape[1]), labels=grid.columns, rotation=28, ha="right")
    ax.set_yticks(range(grid.shape[0]), labels=grid.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):  # skip NaNs
                continue
            r, g, b, _ = cmap(norm(v))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(
                j,
                i,
                fmt % v,
                ha="center",
                va="center",
                color=("black" if lum > 0.6 else "white"),
                fontsize=9,
                fontweight="semibold",
            )
    return im


def plot_triptych(
    tables,
    *,
    figsize=(10.8, 4.0),
    cmap_name="viridis",
    colorbar_label=None,
    title=None,
    save_as=None
):
    _apply_style()
    fig, axs = plt.subplots(
        1, 3, figsize=figsize, sharey=True, gridspec_kw=dict(wspace=0.07)
    )
    cmap = plt.get_cmap(cmap_name)

    # global vmin/vmax from all finite values across panels
    vals = []
    for df, col, _ in tables:
        if df is not None and not df.empty and col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce").to_numpy()
            x = x[np.isfinite(x)]
            if x.size:
                vals.extend(x.tolist())
    vmin, vmax = (0.0, 1.0) if not vals else (float(np.min(vals)), float(np.max(vals)))
    norm = Normalize(vmin=vmin, vmax=vmax)

    last_im = None
    for k, (df, col, label) in enumerate(tables):
        ax = axs[k]
        grid = _grid(df, col)
        if grid.empty:  # nothing to draw → hide axis
            ax.set_title(label)
            ax.axis("off")
            continue
        im = _annotated_heat(ax, grid, norm, cmap)
        last_im = im
        ax.set_title(label)
        ax.set_xlabel("Ablation")
        if k == 0:
            ax.set_ylabel("Model")

    if last_im is not None:
        cbar = fig.colorbar(
            last_im, ax=axs, location="right", fraction=0.035, pad=0.02, aspect=30
        )
        if colorbar_label:
            cbar.set_label(colorbar_label)

    if title:
        fig.suptitle(title, y=0.995, fontsize=11)
    # tight_layout sometimes warns with colorbars; it's OK
    try:
        fig.tight_layout(rect=[0, 0, 0.985, 0.97])
    except Exception:
        pass
    if save_as:
        fig.savefig(save_as, bbox_inches="tight")
    return fig

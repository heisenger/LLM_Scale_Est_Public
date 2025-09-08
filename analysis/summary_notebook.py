from typing import Iterable
import pandas as pd


# =========================
# Bayesian-ness checks
# =========================

import re
import numpy as np
import pandas as pd

# --- 1) Model → factor levels -------------------------------------------------


def model_factors(name: str):
    """Return tuple (is_bayesian, is_weber, is_sequential) from a model name."""
    n = name.lower()

    # Bayesian vs Non-Bayesian
    # Treat anything starting with 'linear' as non-bayesian; all others are bayesian.
    is_bayes = not n.startswith("linear")

    # Weber vs Basic (by suffix token)
    is_weber = "weber" in n
    # Some models may be neither "basic" nor "weber"; this treats those as "basic" by default.

    # Sequential vs Non-sequential
    is_seq = n.startswith("seq") or "seq_" in n

    # Gain factor vs no gain factor
    has_gain = "gain" in n

    return is_bayes, is_weber, is_seq, has_gain


# --- 2) Convert AIC/ΔAIC to unnormalized likelihoods -------------------------


def likelihood_from_aic(df: pd.DataFrame, aic_col="aic"):
    """
    Add a column 'L' with unnormalized likelihoods from AIC.
    Uses ΔAIC within each scope: L = exp(-0.5 * ΔAIC).
    If a 'delta_aic' column already exists, it will be used.
    """
    if "delta_aic" in df.columns:
        d = df["delta_aic"].astype(float).to_numpy()
        L = np.exp(-0.5 * d)
        return df.assign(L=L)

    # compute ΔAIC per scope for stability
    out = []
    for scope, g in df.groupby("scope", sort=False, dropna=False):
        aic = g[aic_col].astype(float).to_numpy()
        d = aic - np.nanmin(aic)
        L = np.exp(-0.5 * d)
        out.append(g.assign(L=L))
    return pd.concat(out, ignore_index=True)


# --- 3) Factor evidence (size-neutral averaging inside each factor) ----------


def factor_evidence(
    df: pd.DataFrame, scope_col="scope", model_col="model", aic_col="aic"
):
    """
    Returns a tidy DataFrame with P(level | data) for each factor and scope.


      - For each factor (Bayesian/Weber/Sequential), we treat the other factors
        as "nuisance" and compute evidence by:
          (i) aggregating evidence WITHIN each nuisance "cell"
              (one exact combo of nuisance-factor levels), then
         (ii) averaging EQUALLY across cells (so adding many variants in a cell
              doesn't increase its weight).
      - Inside a cell we use MEAN(L) (uniform prior over variants in that cell).
        If you prefer "best-in-cell", switch to cell["L"].max() below.

    Output columns: [scope, factor, level, prob]
    """
    d = df.copy()

    # Attach factor levels from model name
    bayes, weber, seq, gain = zip(*d[model_col].map(model_factors))
    d["is_bayesian"] = np.array(bayes, dtype=bool)
    d["is_weber"] = np.array(weber, dtype=bool)
    d["is_sequential"] = np.array(seq, dtype=bool)
    d["has_gain"] = np.array(gain, dtype=bool)

    # Likelihood-like evidence from AIC (adds column 'L', uses delta_aic if present)
    d = likelihood_from_aic(d, aic_col=aic_col)

    # import pdb

    # pdb.set_trace()

    def nuisance_for(factor_col):
        if factor_col == "is_bayesian":
            return ["is_weber"]  # shared across families; exclude seq/gain here
        if factor_col == "is_weber":
            return [
                "is_bayesian",
                "has_gain",
                "is_sequential",
            ]  # these are the cells that contain both values of is_weber
        if factor_col == "is_sequential":
            return ["is_weber", "has_gain"]  # test seq inside where defined
        if factor_col == "has_gain":
            return ["is_weber", "is_sequential", "is_bayesian"]

    # Helper to compute family evidence for a given factor column
    def _family_prob_for_factor(
        g_scope: pd.DataFrame, factor_name: str, factor_col: str
    ):
        # Choose nuisance columns = the other two factors
        all_cols = ["is_bayesian", "is_weber", "is_sequential", "has_gain"]
        nuisance_cols = nuisance_for(factor_col)

        # Compute cell-level aggregates separately for level=True and level=False
        cell_L_true = {}
        cell_L_false = {}

        # Level = True
        g_true = g_scope[
            g_scope[factor_col] == True
        ]  # ie those rows where the factor of interest is True
        if not g_true.empty:
            for key, cell in g_true.groupby(nuisance_cols, sort=False, dropna=False):
                # average of within-cell (uniform prior over variants in that cell) wh
                cell_L_true[key] = float(cell["L"].max())
        # Level = False
        g_false = g_scope[g_scope[factor_col] == False]
        if not g_false.empty:
            for key, cell in g_false.groupby(nuisance_cols, sort=False, dropna=False):
                cell_L_false[key] = float(cell["L"].max())

        # Use the INTERSECTION of available cells for a fair compare; if empty, fall back to union
        keys_true = set(cell_L_true.keys())
        keys_false = set(cell_L_false.keys())
        # import pdb

        # pdb.set_trace()
        keys_use = keys_true & keys_false
        if not keys_use:
            keys_use = keys_true | keys_false  # graceful fallback if grids differ

        # Average equally across cells (each cell gets one vote)
        def avg_over_cells(cell_map, keys):
            if not keys:
                return 0.0
            vals = [cell_map.get(k, 0.0) for k in keys]
            return float(np.mean(vals)) if vals else 0.0

        L_true_avg = avg_over_cells(cell_L_true, keys_use)
        L_false_avg = avg_over_cells(cell_L_false, keys_use)

        Z = L_true_avg + L_false_avg
        p_true = 0.0 if Z == 0.0 else L_true_avg / Z
        p_false = 0.0 if Z == 0.0 else L_false_avg / Z
        return [
            {"factor": factor_name, "level": "True", "prob": p_true},
            {"factor": factor_name, "level": "False", "prob": p_false},
        ]

    rows = []
    for scope, g in d.groupby(scope_col, sort=False, dropna=False):
        for factor_name, col in [
            ("Bayesian", "is_bayesian"),
            ("Weber", "is_weber"),
            ("Sequential", "is_sequential"),
            ("Gain", "has_gain"),
        ]:
            out = _family_prob_for_factor(g, factor_name, col)
            for r in out:
                r[scope_col] = scope
                rows.append(r)

    return pd.DataFrame(rows)


# =========================
# Output tables
# =========================


def merge_with_suffix(left, right, suffix):
    # Use only the last suffix for the right DataFrame
    return pd.merge(left, right, on=["scope", "factor", "level"], suffixes=("", suffix))


def print_run_tables(
    merged_df: pd.DataFrame,
    run_id: str,
    exp_name: str,
    scopes: Iterable[str] = ("overall", "range:short", "range:medium", "range:long"),
    title_prefix: str = "",
):
    """
    Prints a title line and then one table per scope for this run.
    Expects merged_df to already contain the columns you merged
    (e.g., 'scope', 'factor', 'level', 'prob_*' etc.).
    """
    title = f"{title_prefix}{exp_name} — {run_id}"
    print("\n" + title)
    print("=" * len(title))

    # nice display (optional)
    with pd.option_context(
        "display.max_columns",
        200,
        "display.width",
        200,
        "display.float_format",
        lambda x: f"{x:6.2f}",
    ):
        for sc in scopes:
            if (merged_df["scope"] == sc).any():
                # print(f"\n[ scope = {sc} ]")
                # Put the 6 rows in a tidy order
                order = pd.CategoricalDtype(
                    [
                        "Bayesian",
                        "Non-Bayesian",
                        "Logarithmic",
                        "Non-Logarithmic",
                        "Sequential",
                        "Static",
                    ],
                    ordered=True,
                )
                df_sc = merged_df.loc[merged_df["scope"] == sc].copy()
                if "factor" in df_sc.columns and "level" in df_sc.columns:
                    labels = (
                        df_sc["factor"].astype(str) + "|" + df_sc["level"].astype(str)
                    )
                    mapper = {
                        "Bayesian|True": "Bayesian",
                        "Bayesian|False": "Non-Bayesian",
                        "Weber|True": "Logarithmic",
                        "Weber|False": "Non-Logarithmic",
                        "Sequential|True": "Sequential",
                        "Sequential|False": "Static",
                    }
                    df_sc["row"] = labels.map(mapper).fillna(labels)
                    df_sc["row"] = df_sc["row"].astype(order)
                    # Keep only readable cols: row label first, then your prob_* columns
                    prob_cols = [c for c in df_sc.columns if c.startswith("prob")]
                    show = df_sc.sort_values("row")[["row"] + prob_cols].rename(
                        columns={"row": "row_label"}
                    )
                else:
                    # fallback (if you already transformed to row labels)
                    show = df_sc
                print(show.to_string(index=False))

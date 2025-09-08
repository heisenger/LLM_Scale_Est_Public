import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel, norm
import itertools
import json
from analysis.model_fit import preprocess_dataframe
from analysis.data_processing import convert_numpy_types


def analyze_psychophysics_laws(data: pd.DataFrame, save_path=None) -> Dict:
    """
    Systematic analysis of psychophysics laws in the data

    Args:
        data: DataFrame with columns ['trial', 'sample', 'correct', 'response', 'range_category']

    Returns:
        Dictionary with analysis results for each law
    """

    data = data.dropna(subset=["correct", "response"])

    results = {"per_range": {}, "overall": {}}

    # 1. Regression Effect Analysis
    results["per_range"]["regression_effect"] = analyze_regression_effect(data)

    # 2. Weber Law Analysis
    results["per_range"]["weber_law"] = analyze_weber_law(data)

    # 3. Range Effect Analysis
    results["overall"]["range_effect"] = analyze_range_effect(data)

    # 4. Sequential Effect Analysis
    results["per_range"]["sequential_effect"] = analyze_sequential_effect(data)

    if save_path:
        # Save results to the specified path
        with open(save_path, "w") as f:
            json.dump(convert_numpy_types(results), f, indent=2)
    return results


def summarize_psychophysics_laws(results: dict, save_path=None) -> pd.DataFrame:
    """
    Summarize the psychophysics law analysis results into a DataFrame.
    Rows: short, medium, long (range_category)
    Columns: regression angle, scalar variability (mean_cv),
             short vs medium t-statistics, medium vs long t-statistics,
             response auto correlation (r_resp_weighted), stimulus auto correlation (r_stim_weighted)
    """
    # Extract per-range results
    regression = results["per_range"]["regression_effect"]
    weber = results["per_range"]["weber_law"]
    sequential = results["per_range"]["sequential_effect"]
    # Extract overall range effect t-statistics
    range_effect = results["overall"]["range_effect"]["pairwise_comparison_on_overlap"]

    # Prepare mapping for t-statistics
    t_stat_map = {}
    for pair, res in range_effect.items():
        if "t_statistic" in res:
            t_stat_map[pair] = res["t_statistic"]
        else:
            t_stat_map[pair] = None

    # Prepare sequential effect as dict for easy lookup
    if isinstance(sequential, list):
        seq_df = pd.DataFrame(sequential)
    else:
        seq_df = sequential

    seq_dict = seq_df.set_index("range_category").to_dict(orient="index")

    # Build the summary table
    categories = ["short", "medium", "long"]
    summary = []
    for cat in categories:
        reg_angle = regression.get(cat, {}).get("regression_angle", None)
        scalar_var = weber.get(cat, {}).get("mean_cv", None)
        # t-statistics: short vs medium, medium vs long
        t_short_medium = (
            t_stat_map.get("short_vs_medium", None) if cat == "short" else None
        )
        t_medium_long = (
            t_stat_map.get("medium_vs_long", None) if cat == "medium" else None
        )
        r_resp = seq_dict.get(cat, {}).get("r_resp_weighted", None)
        r_stim = seq_dict.get(cat, {}).get("r_stim_weighted", None)
        summary.append(
            {
                "regression_angle": reg_angle,
                "scalar_variability": scalar_var,
                "short_vs_medium_t": t_short_medium,
                "medium_vs_long_t": t_medium_long,
                "response_autocorr": r_resp,
                "stimulus_autocorr": r_stim,
            }
        )

    df = pd.DataFrame(summary, index=categories)
    df.index.name = "range_category"
    if save_path:
        df.to_csv(save_path)
    return df


# =========================
# Regression effect
# =========================


def analyze_regression_effect(data: pd.DataFrame) -> Dict[str, dict]:
    """
    Analyze regression to the mean effect for each range_category.
    Returns a dict: {range_category: {slope, intercept, r_squared, p_value, regression_strength, regression_strength_angular}}
    """
    results = {}
    for rc, group in data.groupby("range_category"):
        if group["correct"].nunique() > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                group["correct"], group["response"]
            )
            regression_strength_slope = 1 - slope
            angle_perfect = np.rad2deg(np.arctan(1))  # 45 degrees
            angle_regression = np.rad2deg(np.arctan(slope))
            regression_strength_angular = (
                abs(angle_perfect - angle_regression) / angle_perfect
            )
        else:
            slope = intercept = r_value = p_value = std_err = np.nan
            regression_strength_slope = regression_strength_angular = np.nan

        results[rc] = {
            "regression_angle": angle_regression,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2 if not np.isnan(r_value) else np.nan,
            "p_value": p_value,
            "regression_strength": regression_strength_slope,
            "regression_strength_angular": regression_strength_angular,
        }
    return results


# =========================
# Weber law / scalar variability
# =========================
def analyze_weber_law(data: pd.DataFrame) -> Dict:
    """
    Analyze Weber's Law (scalar variability) per range_category.
    Returns a dict: {range_category: {bin_magnitudes, coefficient_variations, mean_cv, cv_magnitude_correlation, p_value}}
    """
    results = {}
    for rc, group in data.groupby("range_category"):
        # Create magnitude bins
        n_bins = max(3, min(8, len(group) // 4))
        if n_bins < 2:
            results[rc] = {
                "error": "Insufficient data for Weber law analysis",
                "follows_weber_law": False,
            }
            continue

        bin_edges = np.linspace(
            group["correct"].min(), group["correct"].max(), n_bins + 1
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        cvs = []
        magnitudes = []

        for i in range(n_bins):
            mask = (group["correct"] >= bin_edges[i]) & (
                group["correct"] < bin_edges[i + 1]
            )
            if np.sum(mask) > 1:
                responses = group.loc[mask, "response"]
                if responses.mean() != 0:
                    cv = responses.std() / abs(responses.mean())
                    cvs.append(cv)
                    magnitudes.append(bin_centers[i])

        if len(cvs) > 3:
            correlation, p_value = stats.pearsonr(magnitudes, cvs)
            results[rc] = {
                "mean_cv": np.mean(cvs),
                "cv_magnitude_correlation": correlation,
                "bin_magnitudes": magnitudes,
                "coefficient_variations": cvs,
                "p_value": p_value,
            }
        else:
            results[rc] = {
                "error": "Insufficient data for Weber law analysis",
                "follows_weber_law": False,
            }
    return results


# =========================
# Range effect
# =========================


def analyze_range_effect(data: pd.DataFrame) -> Dict:
    """
    Analyzes range effect through two methods:
    1. Correlation between range width and regression strength.
    2. Pairwise t-tests on the mean responses in overlapping stimulus regions.
    """
    range_categories = data["range_category"].unique()
    if len(range_categories) < 2:
        return {"error": "Need at least 2 range categories for range effect analysis."}

    # --- Part 1: Correlation Analysis ---
    range_analyses = {}
    regression_strengths = []
    mean_magnitudes = []

    for category in range_categories:
        cat_data = data[data["range_category"] == category]
        if len(cat_data) < 2:
            continue

        # Compute slope through linear regression
        slope, intercept, r_value, p_value, _ = stats.linregress(
            cat_data["correct"], cat_data["response"]
        )

        mean_magnitude = cat_data["correct"].mean()

        range_analyses[category] = {
            "slope": slope,
            "regression_strength": 1 - slope,
            "r_squared": r_value**2,
            "mean_stimulus_magnitude": mean_magnitude,
        }
        regression_strengths.append(1 - slope)
        mean_magnitudes.append(mean_magnitude)  # Changed from range_widths

    # Check if regression strength correlates with mean stimulus magnitude
    correlation_results = {}
    if len(regression_strengths) > 2:
        # Use mean_magnitudes for the correlation
        corr, p_val = stats.pearsonr(mean_magnitudes, regression_strengths)
        correlation_results = {
            "correlation_magnitude_regression": corr,
            "p_value": p_val,
            "interpretation": f"Regression strength {'increases' if corr > 0 else 'decreases'} with stimulus magnitude (r={corr:.3f})",
        }
    pairwise_results = _run_all_pairwise_comparisons(data)

    return {
        "range_analyses_by_category": range_analyses,
        "correlation_analysis": correlation_results,
        "pairwise_comparison_on_overlap": pairwise_results,
    }


def _prepare_df_for_comparison(
    df_pair: pd.DataFrame,
    bin_width: float = 0.1,  # 0.1,  # <--- This is the parameter we want to make dynamic
) -> Optional[pd.DataFrame]:
    """Helper to find overlapping bins between two categories."""
    correct_col, response_col, range_col = "correct", "response", "range_category"
    min_val, max_val = df_pair[correct_col].min(), df_pair[correct_col].max()

    # Ensure min_val and max_val are actual numbers and range is valid
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return None  # Cannot create bins for empty or zero-range data

    # Calculate edges based on the dynamic bin_width
    edges = np.arange(np.floor(min_val), np.ceil(max_val) + bin_width, bin_width)

    # Ensure there are at least two edges to form one bin.
    # If the range is very small, bin_width might be larger than max_val - min_val
    # In such cases, np.arange might return fewer than 2 edges.
    if len(edges) < 2:
        # If range is tiny, just create two edges at min and max
        edges = np.array(
            [min_val, max_val + 1e-9]
        )  # Add a tiny epsilon to ensure a bin

    df_pair = df_pair.copy()
    # Add observed=False to pd.cut to handle cases where not all bins are present in specific data
    df_pair["bin"] = pd.cut(
        df_pair[correct_col], bins=edges, include_lowest=True  # , observed=False
    )

    pivot = (
        df_pair.groupby(["bin", range_col], observed=False).size().unstack(fill_value=0)
    )
    common_bins = pivot[(pivot > 0).all(axis=1)].index
    df_common = df_pair[df_pair["bin"].isin(common_bins)]

    if df_common.empty:
        return None

    # Use dropna(how='all') if you want to keep bins where at least one category has data
    # but for t-test, you need both. So dropna() on the unstacked means is correct.
    collapsed_df = (
        df_common.groupby(["bin", range_col], observed=False)[response_col]
        .mean()
        .unstack()
        .dropna()  # Drop rows where either category has no data in that bin
    )
    return collapsed_df if not collapsed_df.empty else None


def _run_pairwise_comparison(collapsed_df: pd.DataFrame) -> Dict:
    """Helper to run a paired t-test."""
    cat1, cat2 = collapsed_df.columns
    # It's good practice to check for sufficient data here too,
    # though dropna() above should handle it for common bins.
    if len(collapsed_df[cat1]) < 2 or len(collapsed_df[cat2]) < 2:
        return {
            "categories": [cat1, cat2],
            "error": "Insufficient data in common bins for t-test.",
            "t_statistic": np.nan,
            "p_value": np.nan,
            "is_significant": False,
            "n_overlapping_bins": 0,
        }

    t_statistic, p_value = ttest_rel(collapsed_df[cat1], collapsed_df[cat2])
    return {
        "categories": [cat1, cat2],
        "t_statistic": t_statistic,
        "p_value": p_value,
        "is_significant": p_value < 0.05,
        "n_overlapping_bins": len(collapsed_df),
        f"mean_{cat1}": collapsed_df[cat1].mean(),
        f"mean_{cat2}": collapsed_df[cat2].mean(),
    }


def _run_all_pairwise_comparisons(data: pd.DataFrame) -> Dict:
    """Main helper for pairwise range effect analysis."""
    categories = data["range_category"].unique()
    category_pairs = list(itertools.combinations(categories, 2))
    all_results = {}

    # --- SMALLest CHANGE TO MAKE BIN DYNAMIC HERE ---
    # 1. Calculate a dynamic bin width based on the entire 'correct' data range.
    #    Using Freedman-Diaconis rule for robustness to outliers.
    all_correct_values = data["correct"].dropna()
    if len(all_correct_values) > 1:
        Q1, Q3 = np.percentile(all_correct_values, [25, 75])
        IQR = Q3 - Q1
        n = len(all_correct_values)
        if IQR > 0:
            # Freedman-Diaconis bin width
            global_bin_width = 2 * IQR / (n ** (1 / 3))
            global_bin_width = (
                global_bin_width * 0.1
            )  # the computed bins are too large normally
        else:
            # Fallback if IQR is zero (e.g., all values are the same)
            # Use a small default or adapt based on the range itself
            global_bin_width = (
                (all_correct_values.max() - all_correct_values.min()) / 10
                if all_correct_values.max() - all_correct_values.min() > 0
                else 1.0
            )
    else:
        # Fallback for very small or empty datasets
        global_bin_width = 1.0  # Default if no data to calculate from

    # Ensure bin_width is not tiny or zero, causing too many bins or errors
    if global_bin_width == 0:
        global_bin_width = 1.0  # Prevent division by zero if all values are identical

    # You could also consider a minimum bin width if your data values are always integers,
    # e.g., global_bin_width = max(1.0, global_bin_width)

    for cat1, cat2 in category_pairs:
        pair_name = f"{cat1}_vs_{cat2}"
        df_pair = data[data["range_category"].isin([cat1, cat2])]

        # Pass the dynamically calculated bin_width to the helper function
        collapsed_df = _prepare_df_for_comparison(df_pair, bin_width=global_bin_width)

        if collapsed_df is not None:
            all_results[pair_name] = _run_pairwise_comparison(collapsed_df)
        else:
            # Improved error message to reflect the reason for no overlap
            all_results[pair_name] = {
                "error": f"No overlapping data found or insufficient data after binning for {cat1} vs {cat2}."
            }
    return all_results


# =========================
# Sequential partial corr
# =========================


def _lag1_autocorr(x: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    x1, x0 = x[1:], x[:-1]
    if x1.std(ddof=0) == 0 or x0.std(ddof=0) == 0:
        return np.nan
    return float(np.corrcoef(x1, x0)[0, 1])


def _fisher_z(r: np.ndarray) -> np.ndarray:
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)


def _inv_fisher_z(z: float) -> float:
    return float(np.tanh(z))


def analyze_sequential_effect(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per range_category with average lag-1 autocorrelation for
    response and stimulus. Uses your preprocess_dataframe() to respect run resets.
    """
    # 1) add run_id via your existing logic
    df = (
        preprocess_dataframe(data)
        .sort_values(["range_category", "run_id", "trial"])
        .copy()
    )

    # 2) per-run autocorrs
    rows = []
    for (rc, rid), g in df.groupby(["range_category", "run_id"], dropna=False):
        s = g["correct"].to_numpy(float)
        y = g["response"].to_numpy(float)
        n = len(s)
        r_resp = _lag1_autocorr(y)
        r_stim = _lag1_autocorr(s)
        rows.append(
            {
                "range_category": rc,
                "run_id": int(rid),
                "n": n,
                "r_resp": r_resp,
                "r_stim": r_stim,
                "z_resp": np.nan if not np.isfinite(r_resp) else _fisher_z(r_resp),
                "z_stim": np.nan if not np.isfinite(r_stim) else _fisher_z(r_stim),
            }
        )
    per_run = pd.DataFrame(rows)

    # 3) summarize to one row per range
    out_rows = []
    for rc, g in per_run.groupby("range_category", dropna=False):
        g_valid_resp = g.dropna(subset=["r_resp"])
        g_valid_stim = g.dropna(subset=["r_stim"])

        # simple means
        r_resp_mean = (
            float(g_valid_resp["r_resp"].mean()) if len(g_valid_resp) else np.nan
        )
        r_stim_mean = (
            float(g_valid_stim["r_stim"].mean()) if len(g_valid_stim) else np.nan
        )

        # Fisher-z weighted means (weights = n-3; minimum 1)
        def _weighted_r(z_col):
            gv = g.dropna(subset=[z_col, "n"])
            if not len(gv):
                return np.nan
            w = np.maximum(gv["n"].to_numpy() - 3, 1)
            zbar = float((gv[z_col].to_numpy() * w).sum() / w.sum())
            return _inv_fisher_z(zbar)

        r_resp_weighted = _weighted_r("z_resp")
        r_stim_weighted = _weighted_r("z_stim")

        out_rows.append(
            {
                "range_category": rc,
                "n_runs": int(g["run_id"].nunique()),
                "N_total": int(g["n"].fillna(0).sum()),
                "r_resp_mean": r_resp_mean,
                "r_stim_mean": r_stim_mean,
                "r_resp_weighted": r_resp_weighted,
                "r_stim_weighted": r_stim_weighted,
            }
        )

    return out_rows

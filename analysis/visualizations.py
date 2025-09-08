import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import sys
import json
import ast
from pathlib import Path
from analysis.data_validation import validate_data
from analysis.psychophysics_laws import analyze_psychophysics_laws

# --- Helper Functions ---


def load_experiment_data(experiment_base_path, exp_name, model):
    csv_path = f"{experiment_base_path}/{exp_name}/{model.replace('/', '_')}/experimental_data.csv"
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def load_model_metrics(experiment_base_path, exp_name, model):
    csv_path = f"{experiment_base_path}/{exp_name}/{model.replace('/', '_')}/model_metrics_aggregate.csv"
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def plot_annotated_bar(ax, x, height, width, color, label=None, alpha=1.0):
    bar = ax.bar(x, height, width, color=color, label=label, alpha=alpha)
    if not np.isnan(height):
        ax.text(x, height, f"{height:.2f}", ha="center", va="bottom", fontsize=8)
    return bar


def get_color_map(keys, palette="Set2"):
    colors = sns.color_palette(palette, len(keys))
    return {k: colors[i] for i, k in enumerate(keys)}


# --- Constants ---

BEHAVIOR_MODELS = [
    "Basic Single-Stage",
    "Basic Two-Stage",
    "Weber Single-Stage",
    "Weber Two-Stage",
    "Linear Regression",
]
BEHAVIOR_MODEL_COLORS = {
    "Basic Single-Stage": "#1f77b4",
    "Basic Two-Stage": "#ff7f0e",
    "Weber Single-Stage": "#2ca02c",
    "Weber Two-Stage": "#d62728",
    "Linear Regression": "#9467bd",
}
RANGE_CATEGORIES = ["short", "medium", "long"]
RANGE_CATEGORY_COLORS = {
    "short": "#17becf",
    "medium": "#bcbd22",
    "long": "#e377c2",
}

# --- Refactored Plotting Functions ---


def plot_bias_variance_bars_multimodal(
    model_list: List[str],
    experiment_names: List[str],
    experiment_base_path: str,
    compute_bias_variance,
    figsize=(14, 10),
    experiment_label_map: Optional[Dict[str, str]] = None,
    model_name_map: Optional[Dict[str, str]] = None,
    ax: Optional[plt.Axes] = None,
    log_scale: bool = True,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if experiment_label_map is None:
        experiment_label_map = {name: name for name in experiment_names}
    if model_name_map is None:
        model_name_map = {name: name for name in model_list}

    bar_width = 0.12
    n_exps = len(experiment_names)
    n_models = len(model_list)
    n_bars_per_model = n_exps * 2 + 2
    indices = np.arange(n_models)
    exp_colors = sns.color_palette("Set2", n_exps)
    combo_color = "#888888"

    for model_idx, model in enumerate(model_list):
        model_bias_vals, model_var_vals, dfs = [], [], []
        for exp_idx, exp_name in enumerate(experiment_names):
            df = load_experiment_data(experiment_base_path, exp_name, model)
            if df is not None:
                bias_sq, var_arr = compute_bias_variance(df)
                model_bias_vals.append(np.mean(bias_sq))
                model_var_vals.append(np.mean(var_arr))
                dfs.append(df)
            else:
                model_bias_vals.append(np.nan)
                model_var_vals.append(np.nan)
                dfs.append(None)
        # Plot bars and annotate
        for exp_idx in range(n_exps):
            plot_annotated_bar(
                ax,
                indices[model_idx] + bar_width * (exp_idx * 2),
                model_bias_vals[exp_idx],
                bar_width,
                exp_colors[exp_idx],
                label=(
                    f"{experiment_label_map[experiment_names[exp_idx]]} Bias²"
                    if model_idx == 0
                    else None
                ),
            )
            plot_annotated_bar(
                ax,
                indices[model_idx] + bar_width * (exp_idx * 2 + 1),
                model_var_vals[exp_idx],
                bar_width,
                exp_colors[exp_idx],
                label=(
                    f"{experiment_label_map[experiment_names[exp_idx]]} Variance"
                    if model_idx == 0
                    else None
                ),
                alpha=0.6,
            )
        # Combo bars
        if len(dfs) >= 2 and dfs[0] is not None and dfs[1] is not None:
            combo_bias2, combo_var = compute_variance_weighted_combo(dfs[0], dfs[1])
            plot_annotated_bar(
                ax,
                indices[model_idx] + bar_width * (n_exps * 2),
                combo_bias2,
                bar_width,
                combo_color,
                label="Optimal Combo Bias²" if model_idx == 0 else None,
            )
            plot_annotated_bar(
                ax,
                indices[model_idx] + bar_width * (n_exps * 2 + 1),
                combo_var,
                bar_width,
                combo_color,
                label="Optimal Combo Variance" if model_idx == 0 else None,
                alpha=0.6,
            )
    group_width = bar_width * n_bars_per_model
    xtick_positions = indices + group_width / 2 - bar_width / 2
    pretty_names = [model_name_map.get(m, m) for m in model_list]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(pretty_names, rotation=30, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Bias² and Variance per Model and Experiment (with Optimal Combo)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=True)
    ax.grid(axis="y", alpha=0.3)
    if log_scale:
        ax.set_yscale("log")
    if ax is None:
        return fig, ax
    else:
        return ax


def plot_model_akaike_weights_grid_multimodal(
    model_list: List[str],
    experiment_names: List[str],
    experiment_base_path: str,
    model_name_map: Optional[Dict[str, str]] = None,
    experiment_label_map: Optional[Dict[str, str]] = None,
    behavior_models: Optional[List[str]] = None,
    figsize=(14, 10),
    log_scale: bool = False,
    ax: Optional[plt.Axes] = None,
):
    if model_name_map is None:
        model_name_map = {name: name for name in model_list}
    if experiment_label_map is None:
        experiment_label_map = {name: name for name in experiment_names}
    if behavior_models is None:
        behavior_models = BEHAVIOR_MODELS
    colors = [BEHAVIOR_MODEL_COLORS[m] for m in behavior_models]
    n_exps = len(experiment_names)
    n_models = len(model_list)
    n_behav = len(behavior_models)
    if ax is None:
        fig, axs = plt.subplots(n_exps, 1, figsize=figsize, sharex=True)
        if n_exps == 1:
            axs = [axs]
    else:
        axs = [ax]
    indices = np.arange(n_models)
    bar_width = 0.12
    for exp_idx, exp_name in enumerate(experiment_names):
        weights_matrix = np.full((n_models, n_behav), np.nan)
        for llm_idx, llm_model in enumerate(model_list):
            df = load_model_metrics(experiment_base_path, exp_name, llm_model)
            if df is not None:
                aic_values = []
                for behav_model in behavior_models:
                    row = df[df["model"] == behav_model]
                    aic_values.append(row["aic"].values[0] if not row.empty else np.nan)
                aic_values = np.array(aic_values)
                weights = compute_relative_akaike_weights(aic_values)
                weights_matrix[llm_idx, :] = weights
        ax = axs[exp_idx]
        for behav_idx, behav_model in enumerate(behavior_models):
            bar_positions = indices + bar_width * behav_idx
            bar_heights = weights_matrix[:, behav_idx]
            bars = ax.bar(
                bar_positions,
                bar_heights,
                bar_width,
                color=colors[behav_idx],
                label=behav_model if exp_idx == 0 else None,
            )
            for xpos, height in zip(bar_positions, bar_heights):
                if not np.isnan(height):
                    ax.text(
                        xpos,
                        height + 0.03,
                        f"{height:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
        ax.set_ylabel("Akaike Weight")
        ax.set_title(experiment_label_map.get(exp_name, exp_name))
        if exp_idx != n_exps - 1:
            ax.set_xticklabels([])
        else:
            pretty_names = [model_name_map.get(m, m) for m in model_list]
            ax.set_xticks(indices + bar_width * (n_behav / 2))
            ax.set_xticklabels(pretty_names, rotation=30, ha="right")
        ax.set_ylim(0, 1.10)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        if exp_idx == 0:
            ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    if ax is None:
        return fig, ax
    else:
        return ax


def plot_response_vs_stimulus_grid(
    model_list: List[str],
    experiment_names: List[str],
    experiment_base_path: str,
    model_name_map: Optional[Dict[str, str]] = None,
    experiment_label_map: Optional[Dict[str, str]] = None,
    figsize=(14, 10),
    alpha=0.5,
    s=12,
    plot_model_predictions: bool = False,
    ax: Optional[plt.Axes] = None,
):
    if model_name_map is None:
        model_name_map = {name: name for name in model_list}
    if experiment_label_map is None:
        experiment_label_map = {name: name for name in experiment_names}
    n_exps = len(experiment_names)
    n_models = len(model_list)
    if ax is None:
        fig, axs = plt.subplots(
            n_exps, n_models, figsize=figsize, sharex=True, sharey=True
        )
        if n_exps == 1:
            axs = np.array([axs])
        if n_models == 1:
            axs = axs.reshape((n_exps, 1))
    else:
        axs = np.array(ax).reshape((len(experiment_names), len(model_list)))
    if ax is None:
        pass
    else:
        if hasattr(axs, "flat"):
            fig = axs.flat[0].figure
        else:
            fig = axs.figure
    range_id_map = RANGE_CATEGORY_COLORS
    marker = "o"
    stim_min, stim_max = np.inf, -np.inf
    for exp_idx, exp_name in enumerate(experiment_names):
        for model_idx, model in enumerate(model_list):
            ax = axs[exp_idx, model_idx]
            df = load_experiment_data(experiment_base_path, exp_name, model)
            if df is not None:
                if "stimulus" in df.columns:
                    x = df["stimulus"]
                elif "sample" in df.columns:
                    x = df["sample"]
                else:
                    x = df.iloc[:, 0]
                y = df["response"]
                stim_min = min(stim_min, x.min())
                stim_max = max(stim_max, x.max())
                if "range_category" in df.columns:
                    group_stats = compute_group_means_vars(df)
                    if not plot_model_predictions:
                        for range_val, color in range_id_map.items():
                            mask = df["range_category"] == range_val
                            ax.scatter(
                                x[mask],
                                y[mask],
                                alpha=0.3,
                                s=s,
                                color=color,
                                marker=marker,
                                edgecolor="none",
                                label=(
                                    range_val
                                    if (model_idx == 0 and exp_idx == 0)
                                    else None
                                ),
                            )
                    for range_val, color in range_id_map.items():
                        means = group_stats[group_stats["range_category"] == range_val]
                        means = means.sort_values("correct_mean")
                        ax.scatter(
                            means["correct_mean"],
                            means["response_mean"],
                            alpha=1.0,
                            s=s * 2.2,
                            color=color,
                            marker=marker,
                            linewidth=1.2,
                            zorder=10,
                        )
                else:
                    if not plot_model_predictions:
                        ax.scatter(
                            x,
                            y,
                            alpha=alpha,
                            s=s,
                            color="gray",
                            marker=marker,
                            edgecolor="none",
                        )
            if plot_model_predictions:
                json_path = f"{experiment_base_path}/{exp_name}/{model.replace('/', '_')}/model_comparison_aggregate.json"
                try:
                    with open(json_path, "r") as f:
                        model_results = json.load(f)
                        for behav_model in BEHAVIOR_MODELS:
                            color_pred = BEHAVIOR_MODEL_COLORS[behav_model]
                            if behav_model in model_results:
                                preds = model_results[behav_model]["predictions"]
                                if isinstance(preds, str):
                                    try:
                                        preds = ast.literal_eval(preds)
                                    except Exception:
                                        preds = np.fromstring(
                                            preds.strip("[]"), sep=" "
                                        )
                                if "correct_mean" in group_stats.columns:
                                    x_pred = group_stats["correct_mean"].values
                                else:
                                    x_pred = np.arange(len(preds))
                                if len(preds) == len(x_pred):
                                    ax.scatter(
                                        x_pred,
                                        preds,
                                        label=f"{behav_model} (pred)",
                                        color=color_pred,
                                        s=s,
                                        marker="x",
                                        linewidth=2.5,
                                        alpha=1.0,
                                        zorder=12,
                                    )
                except Exception as e:
                    print(f"Could not plot model predictions for {model}: {e}")
            if exp_idx == 0:
                ax.set_title(model_name_map.get(model, model), fontsize=11)
            if model_idx == 0:
                ax.set_ylabel(experiment_label_map.get(exp_name, exp_name), fontsize=11)
            if exp_idx == n_exps - 1:
                ax.set_xlabel("Stimulus")
            ax.grid(alpha=0.3)
            ax.set_ylim(stim_min * 0.5, stim_max * 2)
    handles, labels = axs[-1, -1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.02),
            title="Range Category",
            frameon=True,
        )
    plt.tight_layout()
    if ax is None:
        return fig, ax
    else:
        return ax


def plot_group_variance_by_llm_multimodal_lines(
    model_list: List[str],
    experiment_names: List[str],
    experiment_base_path: str,
    model_name_map: Optional[Dict[str, str]] = None,
    experiment_label_map: Optional[Dict[str, str]] = None,
    figsize=(14, 10),
    show_text: bool = False,
    ax: Optional[plt.Axes] = None,
):
    if model_name_map is None:
        model_name_map = {name: name for name in model_list}
    if experiment_label_map is None:
        experiment_label_map = {name: name for name in experiment_names}
    n_exps = len(experiment_names)
    n_models = len(model_list)
    if ax is None:
        fig, axs = plt.subplots(
            n_exps, n_models, figsize=figsize, sharex=True, sharey=True
        )
        if n_exps == 1:
            axs = np.array([axs])
        if n_models == 1:
            axs = axs.reshape((n_exps, 1))
    else:
        axs = np.array(ax).reshape((n_exps, n_models))
    if ax is None:
        pass
    else:
        if hasattr(axs, "flat"):
            fig = axs.flat[0].figure
        else:
            fig = axs.figure
    range_ids = ["short", "medium", "long"]
    range_id_map = RANGE_CATEGORY_COLORS
    for exp_idx, exp_name in enumerate(experiment_names):
        for model_idx, model in enumerate(model_list):
            ax = axs[exp_idx, model_idx]
            df = load_experiment_data(experiment_base_path, exp_name, model)
            if df is not None:
                group_stats = compute_group_means_vars(df)
                for range_val in range_ids:
                    sub = group_stats[group_stats["range_category"] == range_val]
                    if not sub.empty:
                        sub = sub.sort_values("stimulus_id")
                        ax.plot(
                            sub["stimulus_id"],
                            sub["response_var"],
                            marker="o",
                            color=range_id_map[range_val],
                            label=(
                                range_val if (model_idx == 0 and exp_idx == 0) else None
                            ),
                            linewidth=2,
                            alpha=0.85,
                        )
                        if show_text:
                            for x, y in zip(sub["stimulus_id"], sub["response_var"]):
                                ax.text(
                                    x,
                                    y,
                                    f"{y:.2f}",
                                    fontsize=8,
                                    ha="center",
                                    va="bottom",
                                    color=range_id_map[range_val],
                                )
            if exp_idx == 0:
                ax.set_title(model_name_map.get(model, model), fontsize=11)
            if model_idx == 0:
                ax.set_ylabel(experiment_label_map.get(exp_name, exp_name), fontsize=11)
            if exp_idx == n_exps - 1:
                ax.set_xlabel("Trial (Group)")
            n_stimuli = df["stimulus_id"].nunique() if df is not None else 0
            ax.grid(alpha=0.3)
            # Only set log scale if there are positive values
            y_data = (
                sub["response_var"]
                if df is not None and not sub.empty
                else np.array([1])
            )
            if np.any(y_data > 0):
                ax.set_yscale("log")
            else:
                ax.set_yscale("linear")
            ax.set_xticks(range(n_stimuli))
            ax.set_xticklabels([str(i + 1) for i in range(n_stimuli)])
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            title="Range Category",
            frameon=True,
        )
    plt.tight_layout()
    if ax is None:
        return fig, ax
    else:
        return ax


### Plots for text experiments ###


def safe_analyze(data, analysis_name="analysis"):
    """Safely run analysis with error handling"""
    try:
        clean_data, error = validate_data(data)
        if error:
            return None, error

        results = analyze_psychophysics_laws(clean_data)
        return results, None
    except Exception as e:
        return None, f"Analysis failed: {str(e)}..."


def plot_with_error_message(ax, message, title="Analysis Failed"):
    """Plot error message when analysis fails"""
    ax.text(
        0.5,
        0.5,
        f"❌ {message}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_psychophysics_overview(
    data: pd.DataFrame,
    title: str = "Psychophysics Data Overview",
    save_figure: bool = False,
    filename: Optional[str] = None,
    dpi: int = 300,
    format: str = "png",
    save_stats: bool = True,
    stats_filename: Optional[str] = None,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    clean_data, error = validate_data(data)

    if error:
        for ax in axes.flat:
            plot_with_error_message(ax, error)
        plt.suptitle(f"{title} - Data Validation Failed", fontsize=16)
        plt.tight_layout()
        return {"error": error}

    overall_results, overall_error = safe_analyze(clean_data, "overall")

    if overall_error:
        for ax in axes.flat:
            plot_with_error_message(ax, overall_error)
        plt.suptitle(f"{title} - Analysis Failed", fontsize=16)
        plt.tight_layout()
        return {"error": overall_error}

    category_results = {}
    for category in clean_data["range_category"].unique():
        cat_data = clean_data[clean_data["range_category"] == category]
        result, err = safe_analyze(cat_data, f"category_{category}")
        if not err:
            category_results[category] = result

    plot_response_vs_correct(axes[0, 0], clean_data)  # , category_results)
    plot_regression_effect(axes[0, 1], clean_data, category_results)
    plot_weber_law(axes[0, 2], category_results)
    plot_sequential_effects(axes[1, 0], clean_data, overall_results)
    plot_range_effect(axes[1, 1], overall_results)
    plot_response_distributions(axes[1, 2], clean_data)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_figure:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight", format=format)
        print(f"Figure saved as: {filename}")

    # plt.show()

    if save_stats:
        print("\nProcessing and saving statistics...")
        # 1. Assemble the results into the format expected by the helper
        stats_to_save = {
            "plot_info": {
                "title": title,
                "n_total_samples": len(data),
                "n_valid_samples": len(clean_data),
            },
            "overall_stats": overall_results,
            "category_stats": category_results,
        }
        # 2. Create the DataFrame
        stats_df = _create_stats_dataframe(stats_to_save)
        # 3. Save the DataFrame to a CSV file
        if not stats_df.empty:
            _save_stats_dataframe(stats_df, stats_filename, title)
        else:
            print("Warning: Statistics DataFrame was empty, not saving.")
    return {
        "overall": overall_results,
        "by_category": category_results,
    }


def plot_response_vs_correct(
    ax, data, legend=True, selected_title="Response vs. Correct (with Binned Variance)"
):
    """
    Plots response vs. correct values with raw data scatter and binned error bars.

    Args:
        ax: The matplotlib axis object to plot on.
        data: The DataFrame containing 'correct', 'response', and 'range_category' columns.
    """
    try:
        range_categories = sorted(data["range_category"].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(range_categories)))

        # --- Loop through each category to plot its data ---
        for i, category in enumerate(range_categories):
            cat_data = data[data["range_category"] == category]

            # 1. Plot the raw data as a scatter plot
            ax.scatter(
                cat_data["correct"],
                cat_data["response"],
                color=colors[i],
                alpha=0.2,  # Lighter alpha to serve as a background
                s=20,  # Smaller points
                label=f"{category} (raw data)",
            )

            # 2. Calculate and plot binned means with error bars
            if len(cat_data) > 10:  # Ensure there's enough data for meaningful bins
                # Create bins based on the stimulus range for this category
                mag_bins = np.linspace(
                    cat_data["correct"].min(), cat_data["correct"].max(), 10
                )
                mag_centers = (mag_bins[:-1] + mag_bins[1:]) / 2
                bin_means, bin_stds = [], []

                # Calculate the mean and std dev for each bin
                for j in range(len(mag_bins) - 1):
                    mask = (cat_data["correct"] >= mag_bins[j]) & (
                        cat_data["correct"] < mag_bins[j + 1]
                    )
                    if mask.sum() > 1:
                        responses = cat_data.loc[mask, "response"]
                        bin_means.append(responses.mean())
                        bin_stds.append(responses.std())
                    else:
                        bin_means.append(np.nan)
                        bin_stds.append(np.nan)

                # Remove any empty bins before plotting
                valid_mask = ~np.isnan(bin_means)
                if valid_mask.any():
                    ax.errorbar(
                        mag_centers[valid_mask],
                        np.array(bin_means)[valid_mask],
                        yerr=np.array(bin_stds)[valid_mask],
                        fmt="o-",  # Format: line with circle markers
                        color=colors[i],
                        linewidth=2.5,
                        markersize=6,
                        capsize=5,
                        label=f"{category} (binned mean ± std)",
                    )

        # 3. Add the "Perfect Accuracy" y=x line for reference
        min_val = data["correct"].min()
        max_val = data["correct"].max()
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            alpha=0.8,
            label="Perfect Accuracy",
        )

        # 4. Final styling
        ax.set_title(f"{selected_title}")
        ax.set_xlabel("Correct Stimulus")
        ax.set_ylabel("Response (Mean ± Std)")
        if legend:
            ax.legend()
        ax.grid(True, alpha=0.3)

    except Exception as e:
        plot_with_error_message(ax, f"Plot error: {str(e)}")


def plot_regression_effect(ax, data, category_results):
    try:
        range_categories = sorted(data["range_category"].unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(range_categories)))
        for i, category in enumerate(range_categories):
            if category in category_results:
                cat_data = data[data["range_category"] == category]
                reg = category_results[category]["regression_effect"]
                ax.scatter(
                    cat_data["correct"],
                    cat_data["response"],
                    color=colors[i],
                    alpha=0.4,
                    label=f"{category} (slope={reg['slope']:.2f})",
                )
                x = np.linspace(
                    cat_data["correct"].min(), cat_data["correct"].max(), 100
                )
                y = reg["slope"] * x + reg["intercept"]
                ax.plot(x, y, linestyle="--", color=colors[i])
        ax.set_title("Regression to Mean")
        ax.set_xlabel("Correct")
        ax.set_ylabel("Response")
        ax.legend()
        ax.grid(True, alpha=0.3)
    except Exception as e:
        plot_with_error_message(ax, f"Plot 2 error: {str(e)[:40]}")


def plot_weber_law(ax, category_results):
    try:
        colors = plt.cm.Set1(range(len(category_results)))
        for i, (category, results) in enumerate(category_results.items()):
            weber = results.get("weber_law", {})
            if "bin_magnitudes" in weber and weber["bin_magnitudes"]:
                ax.plot(
                    weber["bin_magnitudes"],
                    weber["coefficient_variations"],
                    "o-",
                    label=f"{category} (CV={weber['mean_cv']:.3f})",
                    color=colors[i],
                )
        ax.set_title("Weber's Law")
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("Coefficient of Variation")
        ax.legend()
        ax.grid(True, alpha=0.3)
    except Exception as e:
        plot_with_error_message(ax, f"Plot 3 error: {str(e)[:40]}")


def plot_sequential_effects(ax, data, overall_results):
    """
    Simple plot for sequential effects using the NEW analyze_sequential_effect(...) output.

    Expects:
      overall_results["sequential_effect"] -> DataFrame with columns:
        ['range_category','r_resp_weighted','r_stim_weighted',
         'r_resp_mean','r_stim_mean','n_runs','N_total']

    Draws grouped bars (Response vs Stimulus) per range.
    """
    try:
        df = overall_results["sequential_effect"]
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("sequential_effect must be a non-empty DataFrame")

        # Prefer Fisher-weighted averages if present; otherwise use simple means
        resp_col = (
            "r_resp_weighted" if "r_resp_weighted" in df.columns else "r_resp_mean"
        )
        stim_col = (
            "r_stim_weighted" if "r_stim_weighted" in df.columns else "r_stim_mean"
        )
        if resp_col not in df.columns or stim_col not in df.columns:
            raise ValueError("Expected columns not found (resp/stim weighted or mean)")

        # Keep category order as in incoming data (fallback to df order)
        cat_order = [
            c for c in data["range_category"].unique() if c in set(df["range_category"])
        ]
        if not cat_order:
            cat_order = df["range_category"].tolist()
        plot_df = df.set_index("range_category").reindex(cat_order).reset_index()

        cats = plot_df["range_category"].astype(str).tolist()
        resp_vals = plot_df[resp_col].to_numpy(dtype=float)
        stim_vals = plot_df[stim_col].to_numpy(dtype=float)

        # Bars
        x = np.arange(len(cats))
        width = 0.38
        r_plot = np.nan_to_num(resp_vals, nan=0.0)
        s_plot = np.nan_to_num(stim_vals, nan=0.0)

        ax.bar(x - width / 2, r_plot, width, label="Response")
        ax.bar(x + width / 2, s_plot, width, label="Stimulus")

        # Mark missing values as "NA"
        for i, (rv, sv) in enumerate(zip(resp_vals, stim_vals)):
            if not np.isfinite(rv):
                ax.text(x[i] - width / 2, 0, "NA", ha="center", va="bottom", fontsize=8)
            if not np.isfinite(sv):
                ax.text(x[i] + width / 2, 0, "NA", ha="center", va="bottom", fontsize=8)

        # Cosmetics
        ax.axhline(0, ls="--", lw=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.set_ylabel("Lag-1 autocorrelation (r)")
        title_tag = (
            "Fisher-weighted mean" if resp_col.endswith("weighted") else "simple mean"
        )
        ax.set_title(f"Serial correlation by range ({title_tag})")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

        # Clamp y to [-1, 1] with a touch of padding
        lo = np.nanmin([np.nanmin(resp_vals), np.nanmin(stim_vals), 0.0])
        hi = np.nanmax([np.nanmax(resp_vals), np.nanmax(stim_vals), 0.0])
        if np.isfinite(lo) and np.isfinite(hi):
            pad = 0.05
            ax.set_ylim(max(-1.0, lo - pad), min(1.0, hi + pad))

    except Exception as e:
        # Fallback message on the axes (assumes your helper exists)
        try:
            plot_with_error_message(ax, f"Sequential plot error: {str(e)[:80]}")
        except Exception:
            ax.clear()
            ax.text(0.5, 0.5, f"Sequential plot error:\n{e}", ha="center", va="center")
            ax.axis("off")


def plot_range_effect(ax: plt.Axes, overall_results: Dict):
    """
    Plots the range effect by showing:
    1. A bar chart of regression strength for each ordered category.
    2. A text summary of the pairwise t-test results.
    """
    try:
        # 1. Extract the necessary data from the results dictionary
        if "range_effect" not in overall_results:
            raise ValueError("Missing range effect data in results.")

        range_effect_data = overall_results["range_effect"]
        category_data = range_effect_data.get("range_analyses_by_category", {})
        pairwise_data = range_effect_data.get("pairwise_comparison_on_overlap", {})

        if not category_data:
            raise ValueError("Missing category analysis data.")

        # --- Bar Chart: Regression Strength ---
        # 2. Define a specific order and sort the categories
        category_order = ["short", "medium", "long"]
        # Filter and order the categories that are actually present in the data
        categories_to_plot = [cat for cat in category_order if cat in category_data]

        regression_strengths = [
            category_data[cat]["regression_strength"] for cat in categories_to_plot
        ]
        colors = plt.cm.viridis(np.linspace(0, 1, len(categories_to_plot)))

        # 3. Plot the ordered bars
        ax.bar(categories_to_plot, regression_strengths, color=colors, alpha=0.8)

        # --- Text Summary: Pairwise Comparisons ---
        # 4. Build the text summary for t-statistics and p-values
        stats_text = "Pairwise Comparisons:\n"
        for key, result in pairwise_data.items():
            if "short_vs_long" in key or "long_vs_short" in key:
                continue
            if "error" in result:
                stats_text += f"\n- {key}:\n  {result['error']}"
            else:
                p_val = result["p_value"]
                t_stat = result["t_statistic"]
                # Add an asterisk for significant results
                significance_marker = "*" if p_val < 0.05 else ""
                stats_text += f"\n- {key}:\n  t-stat = {t_stat:.2f}, p = {p_val:.3f}{significance_marker}"

        # 5. Add the text to the plot
        # Move the axis up to make space for the text
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + 0.1, pos.width, pos.height - 0.12])
        ax.text(
            0.5,
            0.02,  # Move text to the bottom (2% from bottom)
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
        )

        # After plotting, adjust the bottom margin to make space for the text
        plt.subplots_adjust(bottom=0.75)  # Increase if needed
        # --- Final Styling ---
        ax.set_title("Range Effect: Regression Strength")
        ax.set_ylabel("Regression Strength (1 - slope)")
        ax.set_xlabel("Range Category")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")  # Add a line at y=0
        ax.grid(True, alpha=0.3, axis="y")

        # Adjust plot margins to make space for the text
        plt.subplots_adjust(right=0.75)

    except Exception as e:
        plot_with_error_message(ax, f"Range effect plot error: {str(e)}")


def plot_response_distributions(ax, data):
    try:
        categories = sorted(data["range_category"].unique())
        colors = plt.cm.Set1(range(len(categories)))
        for i, category in enumerate(categories):
            cat_data = data[data["range_category"] == category]
            ax.hist(
                cat_data["response"],
                bins=20,
                alpha=0.5,
                label=str(category),
                color=colors[i],
                density=True,
            )
        ax.set_title("Response Distributions")
        ax.set_xlabel("Response")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
    except Exception as e:
        plot_with_error_message(ax, f"Plot 6 error: {str(e)[:40]}")


def _create_stats_dataframe(stats_results: dict) -> pd.DataFrame:
    """
    Creates a flattened DataFrame from the nested analysis results dictionary.
    """
    rows = []
    plot_info = stats_results.get("plot_info", {})

    # Flatten and add overall stats
    if "overall_stats" in stats_results and stats_results["overall_stats"]:
        flat_overall = pd.json_normalize(
            stats_results["overall_stats"], sep="_"
        ).to_dict(orient="records")[0]
        rows.append(
            {
                "analysis_level": "overall",
                "category": "all",
                **flat_overall,
                **plot_info,
            }
        )

    # Flatten and add category stats
    for category, cat_stats in stats_results.get("category_stats", {}).items():
        if cat_stats:
            flat_cat = pd.json_normalize(cat_stats, sep="_").to_dict(orient="records")[
                0
            ]
            rows.append(
                {
                    "analysis_level": "category",
                    "category": category,
                    **flat_cat,
                    **plot_info,
                }
            )

    return pd.DataFrame(rows)


def _generate_safe_filename(title: str, extension: str) -> str:
    """Generates a safe filename from a title string."""
    safe_title = "".join(
        c for c in title if c.isalnum() or c in (" ", "-", "_")
    ).rstrip()
    return f"{safe_title.replace(' ', '_').lower()}{extension}"


def _save_stats_dataframe(
    stats_df: pd.DataFrame, stats_filename: Optional[str], title: str
):
    """Saves the statistics DataFrame to a CSV file."""
    if stats_filename is None:
        stats_filename = _generate_safe_filename(title, "_stats.csv")

    try:
        stats_df.to_csv(stats_filename, index=False)
        print(f"Statistics saved as: {stats_filename}")
    except Exception as e:
        print(f"Failed to save statistics: {e}")


def plot_scatter_by_range(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    range_col: str = "range_category",
    ax: plt.Axes = None,
    color_map: str = "Set1",
    color_dict: dict = None,
    alpha: float = 0.7,
    legend: bool = True,
    title: str = "Scatter Plot by Range",
    xlabel: str = None,
    ylabel: str = None,
):
    """
    Plots a scatter chart of x_col vs y_col, colored by range_col.
    Standardizes color choices using a color map or a provided color_dict.

    Args:
        df: DataFrame containing the data.
        x_col: Name of the column for x-axis.
        y_col: Name of the column for y-axis.
        range_col: Column to use for coloring (default: "range_category").
        ax: Matplotlib axis to plot on (optional).
        color_map: Name of matplotlib colormap to use if color_dict not provided.
        color_dict: Optional dict mapping range values to colors.
        alpha: Scatter point transparency.
        legend: Whether to show legend.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    categories = sorted(df[range_col].unique())
    if color_dict is None:
        cmap = plt.get_cmap(color_map)
        colors = [cmap(i) for i in range(len(categories))]
        color_dict = {cat: colors[i] for i, cat in enumerate(categories)}

    for cat in categories:
        cat_data = df[df[range_col] == cat]
        ax.scatter(
            cat_data[x_col],
            cat_data[y_col],
            label=str(cat),
            color=color_dict[cat],
            alpha=alpha,
            s=40,
            edgecolor="k",
            linewidth=0.5,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel else x_col)
    ax.set_ylabel(ylabel if ylabel else y_col)
    if legend:
        ax.legend(title=range_col)
    ax.grid(True, alpha=0.3)


# --------
# Plot functions for cue combination
import os, json, glob
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- I/O helpers ----------


@dataclass
class RunData:
    overall: pd.DataFrame
    per_range: pd.DataFrame
    preds: pd.DataFrame
    w_range: pd.DataFrame
    w_stim: pd.DataFrame
    w_global: Dict


def load_run(run_dir: str) -> RunData:
    overall = pd.read_csv(os.path.join(run_dir, "overall.csv"))
    per_range = pd.read_csv(os.path.join(run_dir, "per_range.csv"))
    preds = pd.read_parquet(os.path.join(run_dir, "predictions.parquet"))
    w_range = pd.read_csv(os.path.join(run_dir, "weights_bayes_per_range.csv"))
    w_stim = pd.read_csv(os.path.join(run_dir, "weights_bayes_per_stimulus.csv"))
    with open(os.path.join(run_dir, "weights_global.json")) as f:
        w_global = json.load(f)
    return RunData(overall, per_range, preds, w_range, w_stim, w_global)


def find_runs(results_root: str, exp: Optional[str] = None) -> List[str]:
    pat = os.path.join(results_root, f"exp={exp}" if exp else "*", "llm=*")
    return sorted(glob.glob(pat))


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def savefig(fig, outpath: str):
    ensure_dir(outpath)
    fig.savefig(outpath, bbox_inches="tight", dpi=180)
    plt.close(fig)


# ---------- small metrics ----------


def _ensure_aic_weight(df: pd.DataFrame) -> pd.DataFrame:
    if "aic_weight" in df.columns and df["aic_weight"].notna().any():
        return df
    d = df.dropna(subset=["aic"]).copy()
    if d.empty:
        df["aic_weight"] = np.nan
        return df
    aic_min = d["aic"].min()
    w = np.exp(-0.5 * (d["aic"] - aic_min))
    d["aic_weight"] = w / w.sum()
    return df.merge(d[["model", "aic_weight"]], on="model", how="left")


def _bias_var(yhat, ytrue):
    err = yhat - ytrue
    mse = float(np.mean(err**2))
    bias = float(np.mean(err))
    return mse, bias**2, float(mse - bias**2)


# ---------- PLOT 1: model evidence ----------


def plot_model_evidence(
    run_dir: str, outpath: Optional[str] = None, title: Optional[str] = None
):
    rd = load_run(run_dir)
    df = _ensure_aic_weight(rd.overall.copy())
    df = df.dropna(subset=["aic_weight"])
    df = df.sort_values("aic_weight", ascending=False)

    fig = plt.figure(figsize=(8, 4))
    plt.bar(df["model"], df["aic_weight"])
    plt.ylim(0, 1)
    plt.ylabel("AIC weight (model evidence)")
    ttl = title or os.path.basename(run_dir)
    plt.title(f"{ttl}: Model evidence")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    if outpath:
        savefig(fig, outpath)
    else:
        plt.show()
    return df[["model", "aic_weight"]]


# ---------- PLOT 2: bias–variance vs truth ----------

_KEEP_MODELS = [
    "Bayes(range)",
    "Equal",
    "EmpiricalLinear",
    "DynamicLogit",
    "LLM(Combined)",
]


def plot_bias_variance(
    run_dir: str,
    include_bayes_stimulus: bool = True,
    outpath: Optional[str] = None,
    title: Optional[str] = None,
):
    rd = load_run(run_dir)
    keep = _KEEP_MODELS.copy()
    if include_bayes_stimulus and (rd.overall["model"] == "Bayes(stimulus)").any():
        keep.insert(1, "Bayes(stimulus)")

    ytrue = rd.preds.loc[rd.preds["model"] == "LLM(Combined)", "correct"].to_numpy()
    rows = []
    for m in keep:
        yhat = rd.preds.loc[rd.preds["model"] == m, "yhat"].to_numpy()
        mse, b2, varc = _bias_var(yhat, ytrue)
        rows.append({"model": m, "mse": mse, "bias2": b2, "variance": varc})
    df = pd.DataFrame(rows)

    fig = plt.figure(figsize=(9, 5))
    x = np.arange(len(df))
    plt.bar(x, df["bias2"], label="bias²")
    plt.bar(x, df["variance"], bottom=df["bias2"], label="variance")
    ttl = title or os.path.basename(run_dir)
    plt.title(f"{ttl}: Bias–variance vs truth")
    plt.ylabel("MSE components")
    plt.xticks(x, df["model"], rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()

    if outpath:
        savefig(fig, outpath)
    else:
        plt.show()
    return df[["model", "mse", "bias2", "variance"]]


# ---------- PLOT 3: mean image weight ----------


def plot_mean_image_weight(
    run_dir: str,
    include_bayes_stimulus: bool = True,
    outpath: Optional[str] = None,
    title: Optional[str] = None,
):
    rd = load_run(run_dir)
    meta = rd.preds[rd.preds["model"] == "LLM(Combined)"][
        ["range_category", "stimulus_id", "y_text", "y_img", "correct"]
    ]

    weights = []

    # Bayes(range)
    w_map = rd.w_range.set_index("range_category")["w_img_bayes"].to_dict()
    w_img_r = meta["range_category"].map(w_map).to_numpy()
    weights.append({"model": "Bayes(range)", "w_img_mean": float(np.mean(w_img_r))})

    # Bayes(stimulus)
    if include_bayes_stimulus and not rd.w_stim.empty:
        ws = rd.w_stim[["range_category", "stimulus_id", "w_img_bayes"]]
        meta_ws = meta.merge(ws, on=["range_category", "stimulus_id"], how="left")
        weights.append(
            {
                "model": "Bayes(stimulus)",
                "w_img_mean": float(meta_ws["w_img_bayes"].mean()),
            }
        )

    # Equal
    weights.append({"model": "Equal", "w_img_mean": 0.5})

    # EmpiricalLinear
    w_img_emp = rd.w_global["empirical_linear"].get("w_img_norm", float("nan"))
    weights.append({"model": "EmpiricalLinear", "w_img_mean": float(w_img_emp)})

    # DynamicLogit
    b0 = rd.w_global["dynamic_logit"]["b0"]
    b1 = rd.w_global["dynamic_logit"]["b1"]
    b2 = rd.w_global["dynamic_logit"]["b2"]
    b3 = rd.w_global["dynamic_logit"]["b3"]
    X1 = np.abs(meta["y_img"].to_numpy() - meta["y_text"].to_numpy())
    X2 = np.abs(meta["y_img"].to_numpy() - meta["correct"].to_numpy())
    X3 = np.abs(meta["y_text"].to_numpy() - meta["correct"].to_numpy())
    z = b0 + b1 * X1 + b2 * X2 - b3 * X3
    w_img_dyn = 1.0 / (1.0 + np.exp(-z))
    weights.append({"model": "DynamicLogit", "w_img_mean": float(np.mean(w_img_dyn))})

    df = pd.DataFrame(weights)

    fig = plt.figure(figsize=(8, 4))
    plt.bar(df["model"], df["w_img_mean"])
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.ylim(0, 1)
    ttl = title or os.path.basename(run_dir)
    plt.title(f"{ttl}: Effective image weight")
    plt.ylabel("Mean weight on image (w_img)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    if outpath:
        savefig(fig, outpath)
    else:
        plt.show()
    return df


# ---------- registry / dispatcher ----------

PLOTS = {
    "evidence": plot_model_evidence,
    "biasvar": plot_bias_variance,
    "weights": plot_mean_image_weight,
}


def run_plots_for_experiment(
    results_root: str,
    exp: str,
    plot_names: List[str] = ("evidence", "biasvar", "weights"),
    include_bayes_stimulus: bool = True,
    out_root: Optional[str] = None,
):
    """
    Run selected plots for all LLMs in a given experiment folder.
    """
    run_dirs = find_runs(results_root, exp)
    for rd in run_dirs:
        llm = os.path.basename(rd).split("=", 1)[1]
        for pname in plot_names:
            fn = PLOTS[pname]
            outpath = None
            if out_root:
                outdir = os.path.join(out_root, f"exp={exp}", f"llm={llm}")
                outpath = os.path.join(outdir, f"{pname}.png")
            fn(
                rd,
                include_bayes_stimulus=include_bayes_stimulus,
                outpath=outpath,
                title=f"{exp} — {llm}",
            )


if __name__ == "__main__":
    # Example usage
    ##### multi modal experiments #####
    model_list = [
        "google/gemini-2.5-flash-lite",  # hundreds of B (lightweight Gemini)
        "openai/gpt-4.1-mini",  # ~8–10B (GPT-4 mini variant)
        "meta-llama/llama-4-maverick",  # ~17B
        "mistralai/mistral-small-3.2-24b-instruct",  # ~24B
        "qwen/qwen-vl-plus",  # ~235B total, ~22B activated
        "openai/gpt-4o-2024-08-06",  # est. ~200B–1T
        "anthropic/claude-3.7-sonnet",  # ~150–250B
    ]
    model_name_map = {
        "meta-llama/llama-4-maverick": "Llama 4 Maverick",
        "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
        "qwen/qwen-vl-plus": "Qwen VL Plus",
        "mistralai/mistral-small-3.2-24b-instruct": "Mistral Small 3.2",
        "openai/gpt-4.1-mini": "GPT-4.1 Mini",
        "openai/gpt-4o-2024-08-06": "GPT-4o",
        "anthropic/claude-3.7-sonnet": "Claude 3.7 Sonnet",
    }

    experiment_names = ["310725_image", "310725_text", "310725_text_image"]
    label_map = {
        "310725_image": "Image Only",
        "310725_text": "Text Only",
        "310725_text_image": "Image + Text",
    }
    experiment_base_path = "../experiments/image_studies/maze_distance/experiment_runs"

    # ### text based experiments ###
    # model_list = [
    #     "meta-llama/llama-3.1-8b-instruct",  # parameter count: 8B
    #     "meta-llama/llama-4-maverick",  # parameter count: 17B
    #     "qwen/qwen3-32b",  # parameter count: 32B
    #     "openai/gpt-3.5-turbo",  # parameter count: 20-40B
    #     "meta-llama/llama-3.3-70b-instruct",  # parameter count: 70B
    #     "openai/gpt-3.5-turbo",  # parameter count: 20-40B
    #     "meta-llama/llama-3.3-70b-instruct",  # parameter count: 70B
    #     "anthropic/claude-3.7-sonnet",  # parameter count: 150-250B
    #     # "google/gemini-2.5-pro",  # parameter count: 25B
    # ]

    # model_name_map = {
    #     "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B",
    #     "meta-llama/llama-4-maverick": "Llama 4 Maverick",
    #     "qwen/qwen3-32b": "Qwen3 32B",
    #     "openai/gpt-3.5-turbo": "GPT-3.5 Turbo",
    #     "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
    #     "anthropic/claude-3.7-sonnet": "Claude 3.7 Sonnet",
    #     # If you want to add Gemini later:
    #     # "google/gemini-2.5-pro": "Gemini 2.5 Pro (25B)",
    # }

    # experiment_names = ["310725"]
    # label_map = {
    #     "310725": "Text Only",
    # }
    # experiment_base_path = (
    #     "../experiments/text_studies/subtitle_duration/experiment_runs"
    # )

    ## plotting functions ###
    ax = plot_bias_variance_bars_multimodal(
        model_list=model_list,
        experiment_names=experiment_names,
        experiment_base_path=experiment_base_path,
        compute_bias_variance=compute_bias_variance_overall,
        experiment_label_map=label_map,
        model_name_map=model_name_map,
    )
    ax = plot_model_akaike_weights_grid_multimodal(
        model_list=model_list,
        experiment_names=experiment_names,
        experiment_base_path=experiment_base_path,
        # compute_bias_variance=compute_bias_variance_overall,
        experiment_label_map=label_map,
        model_name_map=model_name_map,
    )
    # ax = plot_group_variance_by_llm_multimodal(
    #     model_list=model_list,
    #     experiment_names=experiment_names,
    #     experiment_base_path=experiment_base_path,
    #     block_size=10,
    #     model_name_map=model_name_map,
    #     experiment_label_map=label_map,
    # )
    ax = plot_response_vs_stimulus_grid(
        model_list=model_list,
        experiment_names=experiment_names,
        experiment_base_path=experiment_base_path,
        model_name_map=model_name_map,
        experiment_label_map=label_map,
        plot_model_predictions=False,
    )

    ax = plot_response_vs_stimulus_grid(
        model_list=model_list,
        experiment_names=experiment_names,
        experiment_base_path=experiment_base_path,
        model_name_map=model_name_map,
        experiment_label_map=label_map,
        plot_model_predictions=True,
    )

    ax = plot_group_variance_by_llm_multimodal_lines(
        model_list=model_list,
        experiment_names=experiment_names,
        experiment_base_path=experiment_base_path,
        model_name_map=model_name_map,
        experiment_label_map=label_map,
    )

    plt.show()

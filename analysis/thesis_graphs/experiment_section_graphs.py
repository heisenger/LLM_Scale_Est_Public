import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import sys

from analysis.thesis_graphs.config import MODEL_NAME, MODEL_PLOT_ORDER, ABLATION_LABEL

tol_colors = [
    "#332288",  # dark blue
    "#117733",  # green
    "#44AA99",  # teal
    "#88CCEE",  # light blue
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#AA4499",  # purple
    "#882255",  # wine
]


tol_bright = [
    "#EE7733",  # orange
    "#0077BB",  # blue
    "#33BBEE",  # sky blue
    "#009988",  # teal
    "#CC3311",  # red
    "#EE3377",  # pink/magenta
    "#BBBBBB",  # gray
]

tol_muted = [
    "#CC6677",  # rose
    "#DDCC77",  # sand / mustard
    "#117733",  # green
    "#88CCEE",  # light blue
    "#332288",  # indigo
    "#AA4499",  # purple
    "#44AA99",  # teal
    "#999933",  # olive
    "#882255",  # wine
]


tol_warm_mix = [
    "#EE7733",  # warm orange
    "#CC6677",  # rose
    "#DDCC77",  # sand / mustard
    "#882255",  # wine
    "#117733",  # green (as cool accent)
    "#332288",  # indigo (as cool anchor)
]

w_weight_map = {
    "EmpiricalLinear": "empirical_linear_w_img_norm",
    "Combo_NonOracle": "combo_non_oracle_w_img_norm",
    "Combo_Oracle": "combo_oracle_w_img_norm",
}


def add_w_img_column(cue_com_df, cue_com_weights, w_weight_map):
    # Start with all NaN
    cue_com_df = cue_com_df.copy()
    cue_com_df["w_img"] = np.nan

    for idx, row in cue_com_df.iterrows():
        model = row["model"]
        llm = row["LLM_model"]
        base = row["base"]

        if model in w_weight_map:
            col = w_weight_map[model]
            match = cue_com_weights[
                (cue_com_weights["LLM_model"] == llm)
                & (cue_com_weights["base"] == base)
            ]
            if not match.empty:
                cue_com_df.at[idx, "w_img"] = match.iloc[0][col]
        elif model == "Equal":
            cue_com_df.at[idx, "w_img"] = 0.5

    return cue_com_df


def add_rebased_rmse(cue_com_df):
    cue_com_df = cue_com_df.copy()
    cue_com_df["rebased_RMSE"] = np.nan

    # Group by base and LLM_model
    for (base, llm), group in cue_com_df.groupby(["base", "LLM_model"]):
        # Find the RMSE for LLM(Combined) in this group
        combined_row = group[group["model"] == "LLM(Combined)"]
        if not combined_row.empty and combined_row["rmse_vs_truth"].values[0] != 0:
            combined_rmse = combined_row["rmse_vs_truth"].values[0]
            idxs = group.index
            cue_com_df.loc[idxs, "rebased_RMSE"] = (
                group["rmse_vs_truth"] / combined_rmse
            )
        else:
            # If no combined or combined RMSE is zero, leave as NaN
            continue
    return cue_com_df


def systematic_reversal_diff_by_stim(
    df,
    exp_forward="base_text_image",
    exp_reverse="base_8_text_image",
    group_cols=("range_category", "llm_model"),
    stim_col="stimulus_id",
    value_col="response",
):
    """
    To compute the squared difference between forward and reversed experiments, ie, for ablation 8
    """
    # slice
    fwd = df[df["exp_id"] == exp_forward].copy()
    rev = df[df["exp_id"] == exp_reverse].copy()

    # average across runs by stimulus
    fwd_mean = (
        fwd.groupby(list(group_cols) + [stim_col])[value_col].mean().reset_index()
    )
    rev_mean = (
        rev.groupby(list(group_cols) + [stim_col])[value_col].mean().reset_index()
    )

    # figure out N per group (number of stimuli)
    N = fwd_mean.groupby(list(group_cols))[stim_col].max() + 1
    N = N.rename("N").reset_index()

    # attach N
    fwd_mean = fwd_mean.merge(N, on=list(group_cols), how="left")

    # compute partner stimulus id under reversal
    fwd_mean["stim_rev"] = fwd_mean["N"] - 1 - fwd_mean[stim_col]

    # merge forward with reversed using stimulus partner
    merged = fwd_mean.merge(
        rev_mean,
        left_on=list(group_cols) + ["stim_rev"],
        right_on=list(group_cols) + [stim_col],
        suffixes=("_fwd", "_rev"),
    )

    # compute squared diff
    merged["sqdiff"] = (merged[f"{value_col}_fwd"] - merged[f"{value_col}_rev"]) ** 2

    # average across stimuli for summary
    summary = merged.groupby("llm_model")["sqdiff"].mean().reset_index()
    summary.rename(columns={"sqdiff": f"{value_col}_systematic_sqdiff"}, inplace=True)

    return summary, merged


if __name__ == "__main__":
    exp_name = "marker_location"  # "line_length_ratio"  # "maze_distance"  #
    plot_appendix = True  # False

    output_dir = f"outputs/thesis/{exp_name}"
    os.makedirs(output_dir, exist_ok=True)

    output_dir_df = f"outputs/thesis/{exp_name}/dataframes"
    os.makedirs(output_dir_df, exist_ok=True)

    model_names = [
        "GPT-5 Mini",
        "anthropic_claude-3.7-sonnet",
        "openai_gpt-4o-2024-08-06",
        "qwen_qwen2.5-vl-32b-instruct",
        "google_gemini-2.5-flash-lite",
        "meta-llama_llama-4-maverick",
        "mistralai_mistral-small-3.2-24b-instruct",
        "gemma-3-4b-it",
        "microsoft_phi-4-multimodal-instruct",
    ]

    base_path = f"experiments/text_image/{exp_name}/artifacts/"
    exp_df = pd.read_csv(f"{base_path}/experimental_data.csv")
    exp_summary_df = pd.read_csv(f"{base_path}/experimental_data_summary.csv")
    bayes_factor_df = pd.read_csv(f"{base_path}/{exp_name}_summary_factor_bayesian.csv")
    seq_factor_df = pd.read_csv(f"{base_path}/{exp_name}_summary_factor_sequential.csv")
    nrmse_df = pd.read_csv(f"{base_path}/{exp_name}_summary_nrmse_observed.csv")
    cue_com_df = pd.read_parquet(f"{base_path}/overall_cue_combo.parquet")
    cue_com_weights = pd.read_parquet(f"{base_path}/weights_global.parquet")

    # Define fixed colors for range_category
    color_dict = {
        "short": "#1f77b4",  # blue
        "medium": "#ff7f0e",  # orange
        "long": "#2ca02c",  # green
    }
    tol_colors = tol_muted

    cue_com_df = add_w_img_column(cue_com_df, cue_com_weights, w_weight_map)
    cue_com_df = add_rebased_rmse(cue_com_df)
    cue_com_df.to_csv(f"{output_dir_df}/aug_cue_combo.csv", index=False)
    ablation_8_sys, ablation_8_merged = systematic_reversal_diff_by_stim(exp_df)
    ablation_8_sys.to_csv(f"{output_dir_df}/ablation_8_system.csv", index=False)

    #########################
    # Chart set up
    #########################

    # sort model order based in increasing NRMSE
    model_order = nrmse_df[
        (nrmse_df.ablation == "base") & (nrmse_df.modality == "text_image")
    ].sort_values("nrmse", ascending=True)["llm_model"]

    clustered_bars_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    twin_clustered_bars_colors = [
        "#1f77b4",  # dark blue
        "#6baed6",  # light blue
        "#d62728",  # dark red
        "#fb6a4a",  # light red
    ]

    twin_clustered_bars_colors_2 = [
        "#2ca02c",  # dark green
        "#98df8a",  # light green
        "#ff7f0e",  # dark orange
        "#ffbb78",  # light orange
    ]
    #########################
    # Overview Scattered Chart
    #########################

    row_names = ["base_text", "base_image", "base_text_image"]
    row_subtitle = ["Text Input", "Image Input", "Text & Image Input"]
    col_names = [
        "GPT-5 Mini",
        "anthropic_claude-3.7-sonnet",
        "openai_gpt-4o-2024-08-06",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)

    for i, row in enumerate(row_names):
        for j, col in enumerate(col_names):
            ax = axes[i, j]
            # Filter exp_df for the current row and col
            df_sub = exp_df[(exp_df["exp_id"] == row) & (exp_df["llm_model"] == col)]

            # Plot each range_category separately for fixed colors
            for cat, color in color_dict.items():
                df_cat = df_sub[df_sub["range_category"] == cat]
                ax.scatter(
                    df_cat["correct"],
                    df_cat["response"],
                    alpha=0.3,
                    s=10,
                    label=None,
                    color=color,
                )

            df_mean_sub = exp_summary_df[
                (exp_summary_df["exp_id"] == row) & (exp_summary_df["llm_model"] == col)
            ]
            # Plot each range_category separately for fixed colors
            for cat, color in color_dict.items():
                df_mean_cat = df_mean_sub[df_mean_sub["range_category"] == cat]
                ax.scatter(
                    df_mean_cat["mean_correct"],
                    df_mean_cat["mean_response"],
                    alpha=1.0,
                    edgecolors="black",
                    label=cat,
                    color=color,
                )

            # Add 45-degree dotted line
            all_correct = pd.concat([df_sub["correct"], df_mean_sub["mean_correct"]])
            all_response = pd.concat([df_sub["response"], df_mean_sub["mean_response"]])
            min_val = min(all_correct.min(), all_response.min())
            max_val = max(all_correct.max(), all_response.max())
            ax.plot(
                [min_val, max_val], [min_val, max_val], ls=":", color="gray", zorder=1
            )

            # set title
            ax.set_title(f"{MODEL_NAME[col]} - {row_subtitle[i]}", fontsize=9)
            if i == 2:
                ax.set_xlabel("correct")
            if j == 0:
                ax.set_ylabel("response")
            if i == 0 and j == 2:
                ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/base_scatter_grid.pdf", dpi=300, bbox_inches="tight")
    # plt.show()

    # Clear out assigned lists
    row_names.clear()
    row_subtitle.clear()
    col_names.clear()
    #########################
    # Bayesian model evidence and NRMSE : Base
    #########################

    modalities = ["text", "image", "text_image"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    for i, modality in enumerate(modalities):
        ax = axes[i]
        bayes_mod = bayes_factor_df[
            (bayes_factor_df["modality"] == modality)
            & (bayes_factor_df["ablation"] == "base")
        ]
        nrmse_mod = nrmse_df[
            (nrmse_df["modality"] == modality) & (nrmse_df["ablation"] == "base")
        ]

        bayes_mod = bayes_mod.set_index("llm_model").reindex(model_order)
        nrmse_mod = nrmse_mod.set_index("llm_model").reindex(model_order)

        x = np.arange(len(model_order))
        width = 0.35

        # Convert to percent for plotting
        prob_bayes = bayes_mod["prob"].values * 100
        nrmse_vals = nrmse_mod["nrmse"].values * 100

        bars1 = ax.bar(
            x - width / 2, prob_bayes, width, label="Bayes prob (%)", color="#1f77b4"
        )
        ax2 = ax.twinx()
        bars2 = ax2.bar(
            x + width / 2,
            nrmse_vals,
            width,
            label="NRMSE (%)",
            color="#ff7f0e",
            alpha=0.7,
        )

        # Add dotted line at 50% for Bayes prob
        ax.axhline(50, ls=":", color="gray", linewidth=1)

        # Add percentage labels above Bayes prob bars
        max_prob = np.nanmax(prob_bayes)
        for idx, bar in enumerate(bars1):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                f"{height:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#1f77b4",
                fontweight="bold",
            )

        # Add percentage labels above NRMSE bars
        max_nrmse = np.nanmax(nrmse_vals)
        for idx, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                f"{height:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#ff7f0e",
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_NAME[m] for m in model_order], rotation=30)
        ax.set_ylabel("Bayes prob (%)")
        ax2.set_ylabel("NRMSE (%)")
        ax.set_title(f"Modality: {modality}")

        # Set y-limits with extra space for labels
        ax.set_ylim(0, max_prob * 1.5)
        ax2.set_ylim(0, max_nrmse * 1.5)

        if i == 0:
            ax.legend(loc="upper left")  # , bbox_to_anchor=(0, -0.08)
            ax2.legend(loc="upper right")  # , bbox_to_anchor=(1, -0.08)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/base_bayes_prob.pdf", dpi=300, bbox_inches="tight")
    # plt.show()

    # Clear out assigned lists
    modalities.clear()

    #########################
    # Cue Combination Fit and Image weight: Base Case
    #########################
    plot_models = ["EmpiricalLinear", "Equal", "Combo_NonOracle", "Combo_Oracle"]
    plot_labels = [
        "Linear Model",
        "Equal",
        "Bayesian Combo (Non-Oracle)",
        "Bayesian Combo (Oracle)",
    ]

    bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    x = np.arange(len(model_order))
    width = 0.2  # 0.18

    # --- Row 1: Cue Combination Model Prob ---
    ax = axes[0]
    aic_weights = []
    for model in plot_models:
        vals = []
        for llm in model_order:
            val = cue_com_df[
                (cue_com_df["model"] == model)
                & (cue_com_df["LLM_model"] == llm)
                & (cue_com_df["base"] == "base")
            ]["aic_weight"]
            vals.append(val.values[0] * 100 if len(val) > 0 else np.nan)
        aic_weights.append(vals)
    for i, (model, vals) in enumerate(zip(plot_labels, aic_weights)):
        bars = ax.bar(
            x + (i - 1.5) * width, vals, width, label=model, color=bar_colors[i]
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                f"{height:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                color=bar.get_facecolor(),  # <-- use the bar color for the label
            )
    ax.set_ylabel("%")
    ax.set_title("Cue Combination Model Evidence")
    ax.set_ylim(0, np.nanmax(aic_weights) * 1.2)

    # --- Row 2: RMSE vs Actual (%) ---
    ax = axes[1]
    rmse_vals = []
    for model in plot_models:
        vals = []
        for llm in model_order:
            rmse = cue_com_df[
                (cue_com_df["model"] == model)
                & (cue_com_df["LLM_model"] == llm)
                & (cue_com_df["base"] == "base")
            ]["rebased_RMSE"]
            vals.append(rmse.values[0] * 100 if len(rmse) > 0 else np.nan)
        rmse_vals.append(vals)
    for i, (model, vals) in enumerate(zip(plot_labels, rmse_vals)):
        bars = ax.bar(
            x + (i - 1.5) * width, vals, width, label=model, color=bar_colors[i]
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                f"{height:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                color=bar.get_facecolor(),  # <-- use the bar color for the label
            )
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Reference–Relative Efficiency")
    flat_rmse = np.array(rmse_vals).flatten()
    if np.isfinite(flat_rmse).any():
        ax.set_ylim(0, np.nanmax(flat_rmse) * 1.2)
    else:
        ax.set_ylim(0, 1)

    # --- Row 3: Image Weight (%) ---
    ax = axes[2]
    w_img_vals = []
    for model in plot_models:
        vals = []
        for llm in model_order:
            val = cue_com_df[
                (cue_com_df["model"] == model)
                & (cue_com_df["LLM_model"] == llm)
                & (cue_com_df["base"] == "base")
            ]["w_img"]
            vals.append(val.values[0] * 100 if len(val) > 0 else np.nan)
        w_img_vals.append(vals)
    for i, (model, vals) in enumerate(zip(plot_labels, w_img_vals)):
        bars = ax.bar(
            x + (i - 1.5) * width, vals, width, label=model, color=bar_colors[i]
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                f"{height:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                color=bar.get_facecolor(),  # <-- use the bar color for the label
            )
    ax.set_ylabel("Image Weight (%)")
    ax.set_title("Fitted Weighing to Image Modality")
    ax.set_ylim(0, np.nanmax(w_img_vals) * 1.2)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels([MODEL_NAME[m] for m in model_order], rotation=30)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),  # X>1 pushes it outside; Y=0.5 centers vertically
        ncol=len(labels),
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{output_dir}/base_cue_combo.pdf", dpi=300, bbox_inches="tight")

    # Clear out assigned lists
    plot_models.clear()
    plot_labels.clear()
    bar_colors.clear()

    #########################
    # Steering Ablation
    #########################

    exp_ids = ["base", "base_1", "base_2", "base_6"]

    #########################
    # Bayesian Model Evidence

    x = np.arange(len(model_order))
    width = 0.2  # bar width

    ## Prepare Bayes prob and NRMSE data for each exp_id and model
    bayes_probs = []
    nrmse_vals = []
    seq_probs = []
    for exp_id in exp_ids:
        bayes_mod = (
            bayes_factor_df[
                (bayes_factor_df["modality"] == "text_image")
                & (bayes_factor_df["ablation"] == exp_id)
            ]
            .set_index("llm_model")
            .reindex(model_order)
        )
        nrmse_mod = (
            nrmse_df[
                (nrmse_df["modality"] == "text_image")
                & (nrmse_df["ablation"] == exp_id)
            ]
            .set_index("llm_model")
            .reindex(model_order)
        )
        seq_mod = (
            seq_factor_df[
                (seq_factor_df["modality"] == "text_image")
                & (seq_factor_df["ablation"] == exp_id)
            ]
            .set_index("llm_model")
            .reindex(model_order)
        )
        bayes_probs.append(bayes_mod["prob"].values * 100)
        nrmse_vals.append(nrmse_mod["nrmse"].values * 100)
        seq_probs.append(seq_mod["prob"].values * 100)

    # Colors and labels for each exp_id
    bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    # bar_colors =
    bar_labels = [
        "Base",
        "Verbal Steer",
        "Unbiased Numerical Steer",
        "Biased Numerical Steer",
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    # --- Chart 1: Bayes prob ---
    ax = axes[0]
    for i, (vals, label, color) in enumerate(zip(bayes_probs, bar_labels, bar_colors)):
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                f"{height:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )
    ax.set_ylabel("Bayes prob (%)")
    ax.set_title("Model Evidence for Bayesian Behavior")
    ax.set_ylim(0, np.nanmax(bayes_probs) * 1.2)

    # --- Chart 2: Sequence prob ---
    ax = axes[1]
    for i, (vals, label, color) in enumerate(zip(seq_probs, bar_labels, bar_colors)):
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                f"{height:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )
    ax.set_ylabel("Sequence prob (%)")
    ax.set_title("Model Evidence for Sequential Behavior")
    ax.set_ylim(0, np.nanmax(seq_probs) * 1.2)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([MODEL_NAME[m] for m in model_order], rotation=30)

    # Legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        fontsize=11,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(
        f"{output_dir}/steering_bayes_evidence.pdf", dpi=300, bbox_inches="tight"
    )

    # Clear out assigned lists
    bar_colors.clear()
    bar_labels.clear()

    #########################
    # Biased-Variance Decomposition
    ablations = [
        "base",
        "base_1",
        "base_2",
        "base_6",
    ]
    ablation_labels = [
        "Base",
        "Verbal Steer",
        "Unbiased Numerical Steer",
        "Biased Numerical Steer",
    ]
    marker_styles = ["o", "s", "D", "^"]  # circle, square, diamond, triangle
    model_colors = tol_colors[: len(model_order)]

    fig, ax = plt.subplots(figsize=(7, 7))

    for m_idx, model in enumerate(model_order):
        # import pdb

        # pdb.set_trace()
        color = model_colors[m_idx]  #  % len(model_colors)]
        for a_idx, (ablation, marker) in enumerate(zip(ablations, marker_styles)):
            df_plot = nrmse_df[
                (nrmse_df["llm_model"] == model)
                & (nrmse_df["ablation"] == ablation)
                & (nrmse_df["modality"] == "text_image")
            ]
            if not df_plot.empty:
                ax.scatter(
                    df_plot["n_sqrt_bias_squared"],
                    df_plot["n_sqrt_variance"],
                    color=color,
                    marker=marker,
                    s=100,
                    edgecolors="black",
                    label=(
                        f"{MODEL_NAME.get(model, model)} ({ablation_labels[a_idx]})"
                        if a_idx == 0
                        else None
                    ),
                )

    # RMSE contours
    x = np.linspace(0, nrmse_df["n_sqrt_bias_squared"].max() * 1.2, 1000)
    y = np.linspace(0, nrmse_df["n_sqrt_variance"].max() * 1.2, 1000)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(X**2 + Y**2)
    contour_levels = [0.1 * i for i in range(1, 11)]
    cs = ax.contour(X, Y, Z, levels=contour_levels, colors="gray", linestyles="dotted")
    # ax.clabel(cs, fmt="NRMSE=%.2f", fontsize=8)

    # Legend for marker shapes (ablations)
    from matplotlib.lines import Line2D

    ablation_legend_elements = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="gray",
            label=label,
            markersize=10,
            linestyle="None",
            markeredgecolor="black",
        )
        for marker, label in zip(marker_styles, ablation_labels)
    ]

    # Legend for model colors
    model_legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            label=MODEL_NAME.get(model, model),
            markersize=10,
            linestyle="None",
            markeredgecolor="black",
        )
        for model, color in zip(model_order, model_colors)
    ]
    plt.tight_layout(rect=[0, 0, 0.60, 1])  # leave ~22% of figure width on the right

    # Right-side legends (stacked)
    fig.legend(
        handles=model_legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.98, 0.98),  # x in figure coords, y near top
        ncol=1,
        frameon=True,
        fontsize=10,
        title="Model (colour)",
    )

    fig.legend(
        handles=ablation_legend_elements,
        loc="lower left",
        bbox_to_anchor=(0.98, 0.05),  # y near bottom
        ncol=1,
        frameon=True,
        fontsize=10,
        title="Ablation (shape)",
    )

    ax.set_xlabel("Normalized |Bias|")
    ax.set_ylabel("Normalized sqrt(Variance)")
    ax.set_title("Bias–Variance Decomposition (contours: constant NRMSE)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # # Place legends above the plot, split into two rows
    # fig.legend(
    #     handles=model_legend_elements,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.15),
    #     ncol=len(model_legend_elements),
    #     frameon=True,
    #     fontsize=12,
    #     title="Model Indicated with Colour",
    # )
    # fig.legend(
    #     handles=ablation_legend_elements,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.08),
    #     ncol=len(ablation_legend_elements),
    #     frameon=True,
    #     fontsize=12,
    #     title="Ablation Indicated with Shape",
    # )

    # ax.set_xlabel("Normalized |Bias|")
    # ax.set_ylabel("Normalized sqrt(Variance)")
    # ax.set_title("Bias-Variance Decomposition. Contours show constant NRMSE")
    # ax.set_xlim(0, 1)  # or adjust based on your data range
    # ax.set_ylim(0, 1)
    # ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(
        f"{output_dir}/steering_var_bias_decompose.pdf", dpi=300, bbox_inches="tight"
    )
    # plt.show()
    # Clear out assigned lists
    # ablations.clear()
    # ablation_labels.clear()
    # marker_styles.clear()

    #########################
    # Noise Ablation
    #########################

    exp_ids = ["base_3", "base_5"]

    #########################
    # Behavioral Evidence
    bar_labels = [ABLATION_LABEL[exp_id] for exp_id in exp_ids]

    modality = "image"

    x = np.arange(len(model_order))
    width = 0.2  # bar width

    # Prepare Bayes prob and NRMSE data for each exp_id and model
    seq_probs = []
    bayes_probs = []
    nrmse_vals = []
    for exp_id in exp_ids:
        bayes_mod = (
            bayes_factor_df[
                (bayes_factor_df["modality"] == modality)
                & (bayes_factor_df["ablation"] == exp_id)
            ]
            .set_index("llm_model")
            .reindex(model_order)
        )
        nrmse_mod = (
            nrmse_df[
                (nrmse_df["modality"] == modality) & (nrmse_df["ablation"] == exp_id)
            ]
            .set_index("llm_model")
            .reindex(model_order)
        )
        seq_mod = (
            seq_factor_df[
                (seq_factor_df["modality"] == modality)
                & (seq_factor_df["ablation"] == exp_id)
            ]
            .set_index("llm_model")
            .reindex(model_order)
        )
        seq_probs.append(seq_mod["prob"].values * 100)
        bayes_probs.append(bayes_mod["prob"].values * 100)
        nrmse_vals.append(nrmse_mod["nrmse"].values * 100)

    # Colors and labels for each exp_id
    bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # --- Chart 1: Bayes prob ---
    ax = axes[0]
    for i, (vals, label, color) in enumerate(zip(bayes_probs, bar_labels, bar_colors)):
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                f"{height:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                fontweight="bold",
            )
    ax.set_ylabel("Bayesian probability (%)")
    ax.set_title("Bayesian Model Evidence - Image Only Modality")
    ax.set_ylim(0, np.nanmax(bayes_probs) * 1.2)

    # --- Chart 2: Sequential prob --
    ax = axes[1]
    for i, (vals, label, color) in enumerate(zip(seq_probs, bar_labels, bar_colors)):
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                f"{height:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                fontweight="bold",
            )
    ax.set_ylabel("Sequential probability (%)")
    ax.set_title("Sequential Model Evidence - Image Only Modality")
    ax.set_ylim(0, np.nanmax(seq_probs) * 1.2)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([MODEL_NAME[m] for m in model_order], rotation=30)

    # Legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        fontsize=11,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(f"{output_dir}/noise_bayes_seq.pdf", dpi=300, bbox_inches="tight")

    # clear out assigned lists
    bar_labels.clear()

    #########################
    # NRMSE and image weighing
    bar_labels = [ABLATION_LABEL[exp_id] for exp_id in exp_ids]

    x = np.arange(len(model_order))
    width = 0.2  # width of each bar

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ---------- Panel 1: NRMSE Δ ----------
    nrmse_vals = []
    for exp_id in exp_ids:
        vals_img, vals_txtimg = [], []
        for llm in model_order:
            img_base = nrmse_df[
                (nrmse_df["llm_model"] == llm)
                & (nrmse_df["ablation"] == "base")
                & (nrmse_df["modality"] == "image")
            ]["nrmse"]
            img_abl = nrmse_df[
                (nrmse_df["llm_model"] == llm)
                & (nrmse_df["ablation"] == exp_id)
                & (nrmse_df["modality"] == "image")
            ]["nrmse"]
            vals_img.append(
                (img_abl.values[0] - img_base.values[0])
                if (len(img_base) > 0 and len(img_abl) > 0)
                else np.nan
            )

            ti_base = nrmse_df[
                (nrmse_df["llm_model"] == llm)
                & (nrmse_df["ablation"] == "base")
                & (nrmse_df["modality"] == "text_image")
            ]["nrmse"]
            ti_abl = nrmse_df[
                (nrmse_df["llm_model"] == llm)
                & (nrmse_df["ablation"] == exp_id)
                & (nrmse_df["modality"] == "text_image")
            ]["nrmse"]
            vals_txtimg.append(
                (ti_abl.values[0] - ti_base.values[0])
                if (len(ti_base) > 0 and len(ti_abl) > 0)
                else np.nan
            )
        nrmse_vals.append((vals_img, vals_txtimg))

    ax = axes[0]
    for i, (vals_img, vals_txtimg) in enumerate(nrmse_vals):
        bars1 = ax.bar(
            x + i * 2 * width,
            vals_img,
            width,
            label=f"{bar_labels[i]} (Image)",
            color=twin_clustered_bars_colors_2[i * 2],
        )
        bars2 = ax.bar(
            x + i * 2 * width + width,
            vals_txtimg,
            width,
            label=f"{bar_labels[i]} (Text+Image)",
            color=twin_clustered_bars_colors_2[i * 2 + 1],
        )
        # add labels
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            if np.isnan(h):
                continue
            va = "bottom" if h >= 0 else "top"
            offset = 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (offset if h >= 0 else -offset),
                f"{h:.1f}",
                ha="center",
                va=va,
                fontsize=8,
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("NRMSE Δ (ppt)")
    ax.set_title("Change in NRMSE vs Base Case")
    ax.legend(
        fontsize=9,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=len(labels),
        borderaxespad=0,
    )

    # ---------- Panel 2: w_img ----------
    w_img_vals = []
    for exp_id in exp_ids:
        vals_emp, vals_non = [], []
        for llm in model_order:
            emp_base = cue_com_weights[
                (cue_com_weights["LLM_model"] == llm)
                & (cue_com_weights["base"] == "base")
            ]["empirical_linear_w_img_norm"]
            emp = cue_com_weights[
                (cue_com_weights["LLM_model"] == llm)
                & (cue_com_weights["base"] == exp_id)
            ]["empirical_linear_w_img_norm"]
            non_base = cue_com_weights[
                (cue_com_weights["LLM_model"] == llm)
                & (cue_com_weights["base"] == "base")
            ]["combo_non_oracle_w_img_norm"]
            non = cue_com_weights[
                (cue_com_weights["LLM_model"] == llm)
                & (cue_com_weights["base"] == exp_id)
            ]["combo_non_oracle_w_img_norm"]
            vals_emp.append(
                (emp.values[0] - emp_base.values[0]) * 100 if len(emp) > 0 else np.nan
            )
            vals_non.append(
                (non.values[0] - non_base.values[0]) * 100 if len(non) > 0 else np.nan
            )
        w_img_vals.append((vals_emp, vals_non))

    ax = axes[1]
    for i, (vals_emp, vals_non) in enumerate(w_img_vals):
        bars1 = ax.bar(
            x + i * 2 * width,
            vals_emp,
            width,
            label=f"{bar_labels[i]} Fitted Empirical Model",
            color=twin_clustered_bars_colors[i * 2],
        )
        bars2 = ax.bar(
            x + i * 2 * width + width,
            vals_non,
            width,
            label=f"{bar_labels[i]} Non Oracle Bayesian Model",
            color=twin_clustered_bars_colors[i * 2 + 1],
        )
        # add labels
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            if np.isnan(h):
                continue
            va = "bottom" if h >= 0 else "top"
            offset = 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (offset if h >= 0 else -offset),
                f"{h:.0f}",
                ha="center",
                va=va,
                fontsize=8,
            )

    ax.set_ylabel("Image Weight Δ (%)")
    ax.set_title("Change in Fitted Image Weights vs Base Case")
    ax.legend(
        fontsize=9,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.55),
        ncol=len(labels),
        borderaxespad=0,
    )

    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(
        [MODEL_NAME[m] for m in model_order], rotation=30, ha="right"
    )

    plt.tight_layout()
    fig.savefig(f"{output_dir}/noise_ablation.pdf", dpi=300, bbox_inches="tight")

    #########################
    # Bias-Variance Breakdown

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharex=True, sharey=True)

    modalities = ["image", "text_image"]
    ablations = ["base", "base_3", "base_5"]
    ablation_labels = [ABLATION_LABEL[ab] for ab in ablations]
    titles = [
        "Impact of Noise on Bias-Variance - Image Only",
        "Impact of Noise on Bias-Variance - Text+Image",
    ]

    for ax, modality, title in zip(axes, modalities, titles):
        for m_idx, model in enumerate(model_order):
            color = model_colors[m_idx % len(model_colors)]
            for a_idx, (ablation, marker) in enumerate(zip(ablations, marker_styles)):
                df_plot = nrmse_df[
                    (nrmse_df["llm_model"] == model)
                    & (nrmse_df["ablation"] == ablation)
                    & (nrmse_df["modality"] == modality)
                ]
                if not df_plot.empty:
                    ax.scatter(
                        df_plot["n_sqrt_bias_squared"],
                        df_plot["n_sqrt_variance"],
                        color=color,
                        marker=marker,
                        s=100,
                        edgecolors="black",
                    )

        # RMSE contours
        x = np.linspace(0, nrmse_df["n_sqrt_bias_squared"].max() * 1.2, 500)
        y = np.linspace(0, nrmse_df["n_sqrt_variance"].max() * 1.2, 500)
        X, Y = np.meshgrid(x, y)
        Z = np.sqrt(X**2 + Y**2)
        contour_levels = [0.1 * i for i in range(1, 11)]
        ax.contour(X, Y, Z, levels=contour_levels, colors="gray", linestyles="dotted")

        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    axes[0].set_xlabel("Normalized |Bias|")
    axes[1].set_xlabel("Normalized |Bias|")
    axes[0].set_ylabel("Normalized sqrt(Variance)")

    # --- Shared legends ---
    ablation_legend_elements = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="gray",
            label=label,
            markersize=10,
            linestyle="None",
            markeredgecolor="black",
        )
        for marker, label in zip(marker_styles, ablation_labels)
    ]

    model_legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            label=MODEL_NAME.get(model, model),
            markersize=10,
            linestyle="None",
            markeredgecolor="black",
        )
        for model, color in zip(model_order, model_colors)
    ]

    fig.legend(
        handles=model_legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(model_legend_elements),
        frameon=True,
        fontsize=12,
        title="Model Indicated with Colour",
    )
    fig.legend(
        handles=ablation_legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(ablation_legend_elements),
        frameon=True,
        fontsize=12,
        title="Ablation Indicated with Shape",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        f"{output_dir}/noise_var_bias_decompose.pdf", dpi=300, bbox_inches="tight"
    )
    # plt.show()

    # # clear out assigned lists
    # modalities.clear()
    # ablations.clear()
    # ablation_labels.clear()
    # marker_styles.clear()

    #########################
    # Context Ablation
    #########################

    exp_ids = ["base", "base_4", "base_7", "base_8"]
    bar_labels = [ABLATION_LABEL[exp_id] for exp_id in exp_ids]

    #########################
    # Context Ablation
    x = np.arange(len(model_order))
    width = 0.22  # bar width

    # Colors for the 3 deltas (not for base)
    bar_colors = clustered_bars_colors[: len(bar_labels) - 1]
    delta_labels = []  # will fill with bar_labels[1:]

    # ----- Collect data (percent) -----
    bayes_abs = []  # absolute % values per exp_id
    nrmse_abs = []
    for exp_id in exp_ids:
        bayes_mod = (
            bayes_factor_df[
                (bayes_factor_df["modality"] == "text_image")
                & (bayes_factor_df["ablation"] == exp_id)
            ]
            .set_index("llm_model")
            .reindex(model_order)
        )
        nrmse_mod = (
            nrmse_df[
                (nrmse_df["modality"] == "text_image")
                & (nrmse_df["ablation"] == exp_id)
            ]
            .set_index("llm_model")
            .reindex(model_order)
        )
        bayes_abs.append(bayes_mod["prob"].values * 100.0)
        nrmse_abs.append(nrmse_mod["nrmse"].values * 100.0)

    # base = first entry
    bayes_base = bayes_abs[0]
    nrmse_base = nrmse_abs[0]

    # deltas vs base for the 3 variants
    bayes_deltas = [vals - bayes_base for vals in bayes_abs[1:]]
    nrmse_deltas = [vals - nrmse_base for vals in nrmse_abs[1:]]
    delta_labels = bar_labels[
        1:
    ]  # ["Shorter Context", "Longer Context", "Reversed Order"]

    def symmetric_ylim(arr_list, pad=1.1, min_span=5):
        """Make a nice symmetric y-limit around zero from a list of arrays (with NaNs)."""
        vals = np.concatenate([np.asarray(a, float).ravel() for a in arr_list])
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return (-1, 1)
        m = np.nanmax(np.abs(finite))
        m = max(m * pad, min_span)  # ensure some span
        return (-m, m)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # ---------- Chart 1: Δ Bayesian "evidence" (factor prob) vs Base ----------
    ax = axes[0]
    for i, (vals, label, color) in enumerate(
        zip(bayes_deltas, delta_labels, bar_colors)
    ):
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height
                + (
                    0.1 if height >= 0 else -0.1
                ),  # above for positive, below for negative
                f"{height:.1f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=9,
                color=color,
                fontweight="bold",
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ Evidence (pp)")  # percentage points
    ax.set_title("Bayesian Model Evidence - Delta Compared to Base Case")
    ax.set_ylim(*symmetric_ylim(bayes_deltas))

    # ---------- Chart 2: Δ NRMSE vs Base ----------
    ax = axes[1]
    for i, (vals, label, color) in enumerate(
        zip(nrmse_deltas, delta_labels, bar_colors)
    ):
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (0.1 if height >= 0 else -0.1),
                f"{height:.1f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=9,
                color=color,
                fontweight="bold",
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ NRMSE (pp)")
    ax.set_title("NRMSE - Delta Compared to Base Case")
    ax.set_ylim(*symmetric_ylim(nrmse_deltas))

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [MODEL_NAME[m] for m in model_order], rotation=30, ha="right"
    )

    # Single legend on the right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(labels),
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(
        f"{output_dir}/context_bayesian_NRMSE_delta.pdf", dpi=300, bbox_inches="tight"
    )

    # clear out assigned lists

    #########################
    # Reversal Effect
    value_col = "response"

    # ------------------------
    # Plot 1: Bar chart (systematic sqdiff per model)
    # ------------------------
    x = np.arange(len(model_order))
    vals = (
        ablation_8_sys.set_index("llm_model")
        .reindex(model_order)[f"{value_col}_systematic_sqdiff"]
        .values
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(x, vals, width=0.6, color="#1f77b4")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Systematic squared diff")
    ax.set_title("Systematic reversal effect (mean over stimuli)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAME[m] for m in model_order], rotation=30, ha="right")

    # add simple labels above bars
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 0.02 * max(1e-12, np.nanmax(vals)),
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )

    # pad y-limit a bit for labels
    ymax = np.nanmax(vals)
    ax.set_ylim(0, ymax * 1.15 if np.isfinite(ymax) else 1)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/context_reversal_bar.pdf", dpi=300, bbox_inches="tight")
    # plt.show()

    # ------------------------
    # Plot 2: Square scatter (per-stim matched means)
    # x: forward mean; y: reverse mean
    # ------------------------
    # Prepare colors per model (consistent, simple palette)
    cmap = plt.cm.get_cmap("tab20", len(model_order))
    model_to_color = {m: cmap(i) for i, m in enumerate(model_order)}

    fig, ax = plt.subplots(figsize=(12, 12))

    # merged has columns: ... value_fwd, value_rev, sqdiff, llm_model, etc.
    xcol = f"{value_col}_fwd"
    ycol = f"{value_col}_rev"

    for m in model_order:
        dfm = ablation_8_merged[ablation_8_merged["llm_model"] == m]
        if dfm.empty:
            continue
        ax.scatter(
            dfm[xcol],
            dfm[ycol],
            s=50,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
            color=model_to_color[m],
            label=MODEL_NAME.get(m, m),
        )

    # 45-degree reference
    xy = np.concatenate(
        [
            ablation_8_merged[xcol].to_numpy(float),
            ablation_8_merged[ycol].to_numpy(float),
        ]
    )
    finite = xy[np.isfinite(xy)]
    if finite.size > 0:
        lo = np.nanmin(finite)
        hi = np.nanmax(finite)
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        lo, hi = lo - pad, hi + pad
    else:
        lo, hi = 0.0, 1.0
    ax.plot([lo, hi], [lo, hi], linestyle=":", color="gray", linewidth=1)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{value_col} stimuli presented in forward order")
    ax.set_ylabel(f"{value_col} stimuli presented in reverse order")
    ax.set_title("Stimuli Response in Forward vs Reverse Order")

    # legend on the right
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 0.86, 1])
    fig.savefig(
        f"{output_dir}/context_reversal_scatter.pdf", dpi=300, bbox_inches="tight"
    )

    #############
    # Scoring Functions
    #############

    #############
    # Synthetic Rank

    # Combine the three dataframes on 'llm_model'
    df_nrmse = nrmse_df.groupby("llm_model")["nrmse"].mean().reset_index()
    df_seq_var = (
        seq_factor_df.groupby("llm_model")["prob"]
        .var()
        .reset_index()
        .rename(columns={"prob": "seq_var"})
    )
    df_bayes_var = (
        bayes_factor_df.groupby("llm_model")["prob"]
        .var()
        .reset_index()
        .rename(columns={"prob": "bayes_var"})
    )

    combined_df = df_nrmse.merge(df_seq_var, on="llm_model").merge(
        df_bayes_var, on="llm_model"
    )

    # Compute the score column
    combined_df["score"] = (
        (combined_df["seq_var"] + combined_df["bayes_var"]) / 2 + 1
    ) / (combined_df["nrmse"] + 1)

    # Sort by score descending
    combined_df = combined_df.sort_values("score", ascending=False)
    combined_df["model_label"] = combined_df["llm_model"].map(MODEL_NAME)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(combined_df["model_label"], combined_df["score"], color="#1f77b4")
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1f77b4",
            fontweight="bold",
        )

    ax.set_ylabel("Score")
    ax.set_title("Score Based on Behavioral Variance and NRMSE")
    ax.set_xticklabels(combined_df["model_label"], rotation=30)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/synthetic_rank.pdf", dpi=300, bbox_inches="tight")
    # plt.show()

    # Bar chart: RMSE vs Actual Combo Non-Oracle, ranked from lowest to highest

    ablation = "base"  # or whichever ablation you want to plot

    # Get RMSE vs Actual (%) for Combo Non-Oracle
    rmse_vals = []
    for llm in model_order:
        rmse = cue_com_df[
            (cue_com_df["model"] == "Combo_NonOracle")
            & (cue_com_df["LLM_model"] == llm)
            & (cue_com_df["base"] == ablation)
        ]["rmse_vs_truth"]
        rmse_llm = cue_com_df[
            (cue_com_df["model"] == "LLM(Combined)")
            & (cue_com_df["LLM_model"] == llm)
            & (cue_com_df["base"] == ablation)
        ]["rmse_vs_truth"]
        rmse_vals.append(
            rmse_llm.values[0] - rmse.values[0] if len(rmse) > 0 else np.nan
        )

    # Prepare data for ranking
    model_labels = [MODEL_NAME[m] for m in model_order]
    rmse_arr = np.array(rmse_vals)
    sorted_idx = np.argsort(rmse_arr)
    sorted_rmse = rmse_arr[sorted_idx]
    sorted_labels = np.array(model_labels)[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(sorted_labels)), sorted_rmse, color="#2ca02c")
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#2ca02c",
        )

    ax.set_ylabel("Excess RMSE over Bayes")
    ax.set_title("Cue Combination Performance - Excess RMSE over Bayes")
    ax.set_xticks(range(len(sorted_labels)))
    ax.set_xticklabels(sorted_labels, rotation=30)
    plt.tight_layout()
    fig.savefig(
        f"{output_dir}/combo_nonoracle_rmse_ranked.pdf", dpi=300, bbox_inches="tight"
    )
    # plt.show()

    #############
    # Combined plot of RMSE rank and model evidence
    ablation = "base"
    model_order = list(model_names)

    # Get RMSE and evidence values
    rmse_vals, evidence_nonoracle, evidence_oracle = [], [], []
    for llm in model_order:
        rmse = cue_com_df[
            (cue_com_df["model"] == "Combo_NonOracle")
            & (cue_com_df["LLM_model"] == llm)
            & (cue_com_df["base"] == ablation)
        ]["rmse_vs_truth"]
        rmse_llm = cue_com_df[
            (cue_com_df["model"] == "LLM(Combined)")
            & (cue_com_df["LLM_model"] == llm)
            & (cue_com_df["base"] == ablation)
        ]["rmse_vs_truth"]
        rmse_vals.append(
            rmse_llm.values[0] - rmse.values[0] if len(rmse) > 0 else np.nan
        )
        nonoracle = cue_com_df[
            (cue_com_df["model"] == "Combo_NonOracle")
            & (cue_com_df["LLM_model"] == llm)
            & (cue_com_df["base"] == ablation)
        ]["aic_weight"]
        oracle = cue_com_df[
            (cue_com_df["model"] == "Combo_Oracle")
            & (cue_com_df["LLM_model"] == llm)
            & (cue_com_df["base"] == ablation)
        ]["aic_weight"]
        evidence_nonoracle.append(
            nonoracle.values[0] * 100 if len(nonoracle) > 0 else np.nan
        )
        evidence_oracle.append(oracle.values[0] * 100 if len(oracle) > 0 else np.nan)

    # Sort by RMSE
    model_labels = [MODEL_NAME[m] for m in model_order]
    rmse_arr = np.array(rmse_vals)
    evidence_nonoracle_arr = np.array(evidence_nonoracle)
    evidence_oracle_arr = np.array(evidence_oracle)
    sorted_idx = np.argsort(rmse_arr)
    sorted_rmse = rmse_arr[sorted_idx]
    sorted_evidence_nonoracle = evidence_nonoracle_arr[sorted_idx]
    sorted_evidence_oracle = evidence_oracle_arr[sorted_idx]
    sorted_labels = np.array(model_labels)[sorted_idx]

    x = np.arange(len(sorted_labels))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Row 1: RMSE bar chart
    ax1 = axes[0]
    bars_rmse = ax1.bar(x, sorted_rmse, width, color="#2ca02c")
    for bar in bars_rmse:
        height = bar.get_height()
        # ax1.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}", ha="center", va="bottom", fontsize=9, color=bar.get_facecolor())
        offset = 0.01 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + (offset if height >= 0 else -offset - 0.004),
            f"{height:.3f}",
            ha="center",
            va=va,
            fontsize=9,
            color=bar.get_facecolor(),
        )
    ax1.set_ylabel("Excess RMSE over Bayes")
    ax1.set_title("Cue Combination Performance - Excess RMSE over Bayes")

    # Row 2: Evidence clustered bar chart
    ax2 = axes[1]
    bars_nonoracle = ax2.bar(
        x - width / 2,
        sorted_evidence_nonoracle,
        width,
        label="Combo Non-Oracle Evidence",
        color="#1f77b4",
        alpha=0.8,
    )
    bars_oracle = ax2.bar(
        x + width / 2,
        sorted_evidence_oracle,
        width,
        label="Combo Oracle Evidence",
        color="#d62728",
        alpha=0.8,
    )
    for bar in bars_nonoracle:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color=bar.get_facecolor(),
        )
    for bar in bars_oracle:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color=bar.get_facecolor(),
        )
    ax2.set_ylabel("Model Evidence (%)")
    ax2.set_title("Model Evidence for Combo Non-Oracle & Oracle")
    ax2.legend(loc="upper right")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sorted_labels, rotation=30)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/combo_summary_ranked.pdf", dpi=300, bbox_inches="tight")

    #############
    # Synthetic Rank - Breakdown
    # --- Aggregate ---
    df_nrmse = nrmse_df.groupby("llm_model")["nrmse"].mean().reset_index()
    df_seq_var = (
        seq_factor_df.groupby("llm_model")["prob"]
        .var()
        .reset_index()
        .rename(columns={"prob": "seq_var"})
    )
    df_bayes_var = (
        bayes_factor_df.groupby("llm_model")["prob"]
        .var()
        .reset_index()
        .rename(columns={"prob": "bayes_var"})
    )

    # Merge
    combined_df = df_nrmse.merge(df_bayes_var, on="llm_model").merge(
        df_seq_var, on="llm_model"
    )

    # Composite score
    combined_df["score"] = (
        (combined_df["seq_var"] + combined_df["bayes_var"]) / 2 + 1
    ) / (combined_df["nrmse"] + 1)

    combined_df.to_csv(
        f"{output_dir}/dataframes/synthetic_rank_breakdown.csv", index=False
    )

    # Labels
    combined_df["model_label"] = combined_df["llm_model"].map(MODEL_NAME)

    # ---- Ranking: sort by mean NRMSE (ascending) ----
    combined_df = combined_df.sort_values("nrmse", ascending=True).reset_index(
        drop=True
    )

    # ---- Plot: 1x4 panels horizontally ----
    # --- plotting ---
    metrics = [
        ("nrmse", "Mean NRMSE", "#1f77b4", "{:.3f}"),
        ("bayes_var", "Bayesian factor variance", "#ff7f0e", "{:.3f}"),
        ("seq_var", "Sequential factor variance", "#2ca02c", "{:.3f}"),
        ("score", "Composite score", "#d62728", "{:.3f}"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 8), sharey=True)

    y = np.arange(len(combined_df))
    labels = combined_df["model_label"].tolist()

    for ax_idx, (ax, (col, title, color, fmt)) in enumerate(zip(axes, metrics)):
        vals = combined_df[col].values
        bars = ax.barh(y, vals, color=color)

        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(
                v + 0.02 * (np.nanmax(vals) - np.nanmin(vals)),
                b.get_y() + b.get_height() / 2,
                fmt.format(v),
                ha="left",
                va="center",
                fontsize=9,
            )

        ax.set_title(title)
        ax.grid(axis="x", linewidth=0.3, alpha=0.5)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel("Models (ranked by mean NRMSE)")
            ax.invert_yaxis()

    fig.suptitle(
        "Model comparison across metrics (ranked by mean NRMSE)", y=0.98, fontsize=16
    )
    plt.subplots_adjust(left=0.25, right=0.98, top=0.9, bottom=0.05)
    fig.savefig(
        f"{output_dir}/synthetic_rank_breakdown.pdf", dpi=300, bbox_inches="tight"
    )

    # #################
    # # Appendix Graphs
    # #################
    if plot_appendix:
        modalities = ["text", "image", "text_image"]
        ablations = [
            "base",
            "base_1",
            "base_2",
            "base_3",
            "base_4",
            "base_5",
            "base_6",
            "base_7",
            "base_8",
        ]  # e.g., ["base", "base_1", "base_2", "base_6"]
        os.makedirs(f"{output_dir}/appendix", exist_ok=True)
        for model_to_plot in model_names:
            fig, axes = plt.subplots(
                len(ablations),
                len(modalities),
                figsize=(12, 2 * len(ablations)),
                sharex=True,
                sharey=True,
            )

            fig.suptitle(
                f"Scatter Grid for {MODEL_NAME[model_to_plot]} - {exp_name}",
                fontsize=16,
                y=0.98,
            )  # Add this line

            for i, ablation in enumerate(ablations):
                for j, modality in enumerate(modalities):
                    ax = axes[i, j]
                    # For noise ablations, always use base_text for base_3_text and base_5_text
                    exp_id = f"{ablation}_{modality}"

                    # For the noise ablations, the text inputs are unchanged. Revert to base text input in those cases
                    if exp_id in ["base_3_text", "base_5_text"]:
                        exp_id = "base_text"
                    # Filter for this ablation, modality, and model
                    df_sub = exp_df[
                        (exp_df["exp_id"] == exp_id)
                        & (exp_df["llm_model"] == model_to_plot)
                    ]
                    # Scatter for each range_category
                    for cat, color in color_dict.items():
                        df_cat = df_sub[df_sub["range_category"] == cat]
                        ax.scatter(
                            df_cat["correct"],
                            df_cat["response"],
                            alpha=0.3,
                            s=10,
                            label=None,
                            color=color,
                        )
                    # Means: thicker, border
                    df_mean_sub = exp_summary_df[
                        (exp_summary_df["exp_id"] == exp_id)
                        & (exp_summary_df["llm_model"] == model_to_plot)
                    ]
                    for cat, color in color_dict.items():
                        df_mean_cat = df_mean_sub[df_mean_sub["range_category"] == cat]
                        ax.scatter(
                            df_mean_cat["mean_correct"],
                            df_mean_cat["mean_response"],
                            alpha=1.0,
                            edgecolors="black",
                            label=cat,
                            color=color,
                        )
                    # 45-degree dotted line
                    all_correct = pd.concat(
                        [df_sub["correct"], df_mean_sub["mean_correct"]]
                    )
                    all_response = pd.concat(
                        [df_sub["response"], df_mean_sub["mean_response"]]
                    )
                    if not all_correct.empty:
                        all_correct_min = 0
                        all_correct_max = all_correct.max()
                        ax.set_xlim(all_correct_min, 1.1 * all_correct_max)
                        ax.set_ylim(all_correct_min, 1.5 * all_correct_max)
                        # 45-degree line from (0,0) to (1.5*max_correct, 1.5*max_correct)
                        ax.plot(
                            [all_correct_min, all_correct_max],
                            [all_correct_min, all_correct_max],
                            ls=":",
                            color="gray",
                            zorder=1,
                        )
                    if i == 0:
                        ax.set_title(modality.replace("_", " ").title())
                    if j == 0:
                        ax.set_ylabel(ABLATION_LABEL.get(ablation, ablation))
                    if i == len(ablations) - 1:
                        ax.set_xlabel("correct")
                    if i == 0 and j == len(modalities) - 1:
                        ax.legend(loc="upper left", fontsize=8)
            plt.tight_layout(rect=[0, 0, 1, 0.99])  # Adjust rect so suptitle is visible
            plt.savefig(
                f"{output_dir}/appendix/{model_to_plot}_full_response_scatter_grid.pdf",
                dpi=300,
            )
            # plt.show()

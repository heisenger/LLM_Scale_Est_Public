import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from analysis.thesis_graphs.config import MODEL_NAME, ABLATION_LABEL

# Define fixed colors for range_category
color_dict = {
    "short": "#1f77b4",  # blue
    "medium": "#ff7f0e",  # orange
    "long": "#2ca02c",  # green
}


if __name__ == "__main__":
    exp_name = "line_length_ratio"  #  "maze_distance"  # "marker_location"  #
    base_path = f"experiments/text_image/{exp_name}/artifacts"
    output_dir = f"outputs/thesis/general"
    os.makedirs(output_dir, exist_ok=True)

    exp_df = pd.read_csv(f"{base_path}/experimental_data.csv")
    exp_summary_df = pd.read_csv(f"{base_path}/experimental_data_summary.csv")
    bayes_factor_df = pd.read_csv(f"{base_path}/{exp_name}_summary_factor_bayesian.csv")
    seq_factor_df = pd.read_csv(f"{base_path}/{exp_name}_summary_factor_sequential.csv")
    nrmse_df = pd.read_csv(f"{base_path}/{exp_name}_summary_nrmse_observed.csv")
    cue_com_df = pd.read_parquet(f"{base_path}/overall_cue_combo.parquet")
    cue_com_weights = pd.read_parquet(f"{base_path}/weights_global.parquet")
    behavioral_model_df = pd.read_parquet(f"{base_path}/behavioral_models.parquet")

    # Best candidate for the main model classes
    best_models = behavioral_model_df.loc[
        behavioral_model_df.groupby(["llm_model", "exp_id"])["aic"].idxmin()
    ]

    # Best models in each category
    static_best = best_models[best_models["model_name"] == "Static"]
    static_gain_best = best_models[best_models["model_name"] == "StaticGain"]
    seq_gain_best = best_models[best_models["model_name"] == "SeqGain"]
    linear_best = best_models[best_models["model_name"] == "Linear"]

    llm_list = [
        "GPT-5 Mini",
        "openai_gpt-4o-2024-08-06",
        "anthropic_claude-3.7-sonnet",
        "meta-llama_llama-4-maverick",
    ]

    static_sel = static_best[static_best.llm_model.isin(llm_list)].iloc[:1]
    seq_sel = seq_gain_best[seq_gain_best.llm_model.isin(llm_list)].iloc[:1]
    linear_sel = linear_best[linear_best.llm_model.isin(llm_list)].iloc[:1]

    ##############################
    # Pie chart of best fitting models
    ##############################

    nice_exp_name = ["Line Length Ratio", "Marker Location", "Maze Distance Est."]
    exp_names = ["line_length_ratio", "marker_location", "maze_distance"]

    # Gather all model types for consistent coloring
    all_model_types = set()
    model_counts_list = []
    for exp_name in exp_names:
        behavioral_model_df_by_exp = pd.read_parquet(
            f"experiments/text_image/{exp_name}/artifacts/behavioral_models.parquet"
        )
        best_models_by_exp = behavioral_model_df_by_exp.loc[
            behavioral_model_df_by_exp.groupby(["llm_model", "exp_id"])["aic"].idxmin()
        ]
        counts = best_models_by_exp["model_name"].value_counts()
        all_model_types.update(counts.index)
        model_counts_list.append(counts)

    all_model_types = sorted(all_model_types)
    color_map = {
        m: plt.cm.Set2.colors[i % len(plt.cm.Set2.colors)]
        for i, m in enumerate(all_model_types)
    }

    fig, axes = plt.subplots(1, len(exp_names), figsize=(6 * len(exp_names), 6))

    for i, (ax, exp_name, counts) in enumerate(zip(axes, exp_names, model_counts_list)):
        sizes = [counts.get(m, 0) for m in all_model_types]
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            autopct="%1.1f%%",
            startangle=90,
            colors=[color_map[m] for m in all_model_types],
            radius=1.0,
            textprops={"fontsize": 14},
        )
        for autotext in autotexts:
            autotext.set_fontsize(14)
        ax.set_title(
            f"{nice_exp_name[i]}", fontsize=20
        )  # Use nice_exp_name as sub-heading

    # Add a legend below all pies
    fig.legend(
        [
            plt.Line2D(
                [0], [0], color=color_map[m], marker="o", linestyle="", markersize=15
            )
            for m in all_model_types
        ],
        all_model_types,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(all_model_types),
        fontsize=20,
    )
    # fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])  # Leave space for legend and titles
    fig.savefig(
        f"{output_dir}/best_model_type_split.pdf", bbox_inches="tight"
    )  # plt.show()
    plt.close(fig)

    ##############################
    # Example Plots for behavioral models fitting LLM responses
    ##############################

    ##############################
    # Static Bayes
    sel_df = exp_df[
        (exp_df["llm_model"] == static_sel["llm_model"].values[0])
        & (exp_df["exp_id"] == static_sel["exp_id"].values[0])
    ]
    model_pred = static_sel["pred"].values[0]
    exp_name = ABLATION_LABEL["_".join(static_sel["exp_id"].values[0].split("_")[:2])]

    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot raw points for each range
    for i, range_val in enumerate(sorted(sel_df["range_category"].unique())):
        idx = sel_df["range_category"] == range_val
        ax.scatter(
            sel_df.loc[idx, "correct"],
            sel_df.loc[idx, "response"],
            alpha=0.3,
            s=20,
            label=None,
            color=color_dict[range_val],
        )

    # Plot mean points by range and stimulus_id
    grouped = (
        sel_df.groupby(["range_category", "stimulus_id"])
        .agg({"correct": "mean", "response": "mean"})
        .reset_index()
    )
    for i, range_val in enumerate(sorted(grouped["range_category"].unique())):
        df_mean = grouped[grouped["range_category"] == range_val]
        ax.scatter(
            df_mean["correct"],
            df_mean["response"],
            alpha=1.0,
            edgecolors="black",
            s=40,
            label=f"Mean {range_val}",
            color=color_dict[range_val],
            zorder=3,
        )

    # Plot model prediction line (black, only one legend entry)
    pred_line_drawn = False
    for i, range_val in enumerate(sorted(sel_df["range_category"].unique())):
        idx = sel_df["range_category"] == range_val
        correct_vals = sel_df.loc[idx, "correct"]
        pred_vals = model_pred[idx]
        sort_idx = np.argsort(correct_vals)
        if not pred_line_drawn:
            ax.plot(
                correct_vals.iloc[sort_idx],
                pred_vals[sort_idx],
                linestyle="-",
                alpha=0.7,
                color="black",
                label="Static Bayesian Model",
                zorder=2,
            )
            pred_line_drawn = True
        else:
            ax.plot(
                correct_vals.iloc[sort_idx],
                pred_vals[sort_idx],
                linestyle="-",
                alpha=0.7,
                color="black",
                zorder=2,
            )

    # Add 45-degree dotted line
    all_correct = pd.concat([sel_df["correct"], grouped["correct"]])
    all_response = pd.concat([sel_df["response"], grouped["response"]])
    min_val = min(all_correct.min(), all_response.min())
    max_val = max(all_correct.max(), all_response.max())
    ax.plot([min_val, max_val], [min_val, max_val], ls=":", color="gray", zorder=1)

    ax.set_xlabel("Stimulus")
    ax.set_ylabel("Response")
    ax.legend()
    ax.set_title(
        f"Model Fit for {MODEL_NAME[static_sel['llm_model'].values[0]]} - {exp_name}"
    )
    fig.tight_layout()
    fig.savefig(f"{output_dir}/model_fit_{static_sel['llm_model'].values[0]}.pdf")

    ##############################
    # Seq Bayes
    sel_df = exp_df[
        (exp_df["llm_model"] == seq_sel["llm_model"].values[0])
        & (exp_df["exp_id"] == seq_sel["exp_id"].values[0])
    ]
    model_pred = seq_sel["pred"].values[0]
    exp_name = ABLATION_LABEL["_".join(seq_sel["exp_id"].values[0].split("_")[:2])]

    grouped = (
        sel_df.groupby(["range_category", "stimulus_id"])
        .agg({"correct": "mean", "response": "mean"})
        .reset_index()
    )

    # --- Trajectory plot with color_dict ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for range_val in grouped["range_category"].unique():
        df_range = grouped[grouped["range_category"] == range_val]
        # Sort by stimulus_id for line connection
        df_range_sorted = df_range.sort_values("stimulus_id")
        x = df_range_sorted["correct"].values
        y = df_range_sorted["response"].values
        ax.plot(
            x,
            y,
            marker="o",
            label=f"Range {range_val}",
            linestyle="-",
            color=color_dict[range_val],
        )
        # Add arrows between consecutive points
        for i in range(len(x) - 1):
            ax.annotate(
                "",
                xy=(x[i + 1], y[i + 1]),
                xytext=(x[i], y[i]),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
            )

    ax.set_xlabel("Correct (mean)")
    ax.set_ylabel("Response (mean)")
    ax.legend()
    ax.set_title(
        f"Mean Response Trajectory (arrow shows sequence) - {seq_sel['llm_model'].values[0]} - {exp_name}"
    )
    fig.tight_layout()
    fig.savefig(
        f"{output_dir}/mean_response_trajectory_{seq_sel['llm_model'].values[0]}.pdf"
    )

    # --- LLM output and model prediction lines with color_dict ---
    fig, ax = plt.subplots(figsize=(10, 6))

    range_indices = [(0, 10), (50, 60), (100, 110)]
    range_labels = ["short", "medium", "long"]  # match keys in color_dict

    for i, (start, end) in enumerate(range_indices):
        # LLM output (response)
        llm_vals = sel_df.iloc[start:end]["response"].values
        # Model predictions (preds)
        pred_vals = model_pred[start:end]
        x = np.arange(end - start)
        ax.plot(
            x,
            llm_vals,
            label=f"LLM Response {range_labels[i]}",
            color=color_dict[range_labels[i]],
            linestyle="-",
        )
        # Model prediction line always black, label only once
        if i == 2:
            ax.plot(
                x,
                pred_vals,
                label="Sequential Bayesian Model",
                color="black",
                linestyle="--",
            )
        else:
            ax.plot(x, pred_vals, color="black", linestyle="--")

    ax.set_xlabel("Stimulus Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.set_title(
        f"Mean Response by Stimulus Index - {seq_sel['llm_model'].values[0]} - {exp_name}"
    )
    fig.tight_layout()
    fig.savefig(
        f"{output_dir}/mean_response_by_stimulus_index_{seq_sel['llm_model'].values[0]}.pdf"
    )

# analysis/thesis/graph_codes/run_all.py
from __future__ import annotations
from .util import ensure_dir
from analysis.factor_summary import save_factor_csvs
from .cue_combo_summary import save_cue_combo_csv
from .plot_heatmaps import plot_all_from_csvs
from pathlib import Path
import pandas as pd
from .plot_bars import (
    plot_bayesness_bars,
    plot_observed_vs_bayes_bars,
    plot_observed_vs_bayes_bars_two_rows,
)
from analysis.thesis_graphs.prompt_pages_one import (
    make_prompt_pages_from_master_one_interaction,
)
from analysis.nrmse_summary import save_nrmse_observed_csv
from .plot_scatter import (
    plot_nrmse_vs_bayes,
    plot_ratio_scatter_nrmse_vs_bayes,
    plot_response_vs_stimulus_grid_base_two_models,
    plot_response_vs_stimulus_grid_all_ablations,
)
import os


base_path = Path(__file__).parent

# --- set once per experiment ---
EXP_MODE = "text_image"  # "text"  #
EXP_TAG = "marker_location"  # "line_length_ratio"  # "maze_distance"  #      "subtitle_duration"  #      change per experiment
PATH_EXP_DATA = f"{base_path}/../../experiments/{EXP_MODE}/{EXP_TAG}/artifacts/experimental_data.csv"  # /runs/prior/170825
PATH_BEH = f"{base_path}/../../experiments/{EXP_MODE}/{EXP_TAG}/artifacts/behavioral_models.parquet"
PATH_CUE = (
    f"{base_path}/../../experiments/{EXP_MODE}/{EXP_TAG}/artifacts/overall.parquet"
)
MASTER_TURNS = f"{base_path}/../../experiments/{EXP_MODE}/{EXP_TAG}/artifacts/llm_prompt_preview.turns.parquet"  # your master parquet
ARTIFACTS_ROOT = f"{base_path}/../../experiments/{EXP_MODE}/{EXP_TAG}/artifacts/"  # folder that contains weights_* artifacts
OUT_SUMMARY = f"{base_path}/../../outputs/thesis/summaries/{EXP_TAG}"
OUT_FIGS = f"{base_path}/../../outputs/thesis/graphs/{EXP_TAG}"

MULTI_MODAL = (
    True  # False  #     ONLY USE FALSE WHEN SINGLE MODAL - IE WHEN DOING SUBTITLE
)


def main():
    ensure_dir(OUT_SUMMARY)
    ensure_dir(OUT_FIGS)

    save_nrmse_observed_csv(
        str(PATH_EXP_DATA), str(OUT_SUMMARY), EXP_TAG, baseline="experiment_mean"
    )

    # 1) Factor summaries
    save_factor_csvs(str(PATH_BEH), str(OUT_SUMMARY), EXP_TAG)

    # 2) Cue-combo summary
    if MULTI_MODAL:
        save_cue_combo_csv(
            str(PATH_EXP_DATA),
            str(PATH_CUE),
            str(ARTIFACTS_ROOT),
            str(OUT_SUMMARY),
            EXP_TAG,
            baseline="experiment_mean",
        )

    # 3) heatmaps (now includes NRMSE triptych)
    plot_all_from_csvs(
        str(OUT_SUMMARY), EXP_TAG, str(OUT_FIGS), multi_modal=MULTI_MODAL
    )

    # --- 🔎 SANITY CHECK (add this block) ---
    if MULTI_MODAL:
        cue_csv = Path(OUT_SUMMARY) / f"{EXP_TAG}_summary_cue_combo_long.csv"
        cue_long = pd.read_csv(cue_csv)

        emp = cue_long.query("combo_model == 'EmpiricalLinear'")
        print(f"[sanity] EmpiricalLinear rows: {len(emp)}")
        if emp.empty:
            print(
                "[sanity] WARNING: no EmpiricalLinear rows in summary; "
                "its triptych will be blank."
            )
        else:
            print(
                "[sanity] EmpiricalLinear ablations:",
                sorted(emp["ablation"].dropna().unique()),
            )
            print(
                "[sanity] EmpiricalLinear models:",
                emp["model_pretty"].dropna().unique().tolist(),
            )
            print(
                "[sanity] Any NaNs in prob/nrmse?",
                emp["prob"].isna().any(),
                emp["nrmse"].isna().any(),
            )

    # # 3) Plots
    # plot_all_from_csvs(str(OUT_SUMMARY), EXP_TAG, str(OUT_FIGS))

    # 4) Bar charts
    plot_bayesness_bars(
        OUT_SUMMARY,
        EXP_TAG,
        save_as=Path(OUT_FIGS)
        / f"{EXP_TAG}_bars_bayesian_base_vs_numsteer_image_textimage.pdf",
    )

    if MULTI_MODAL:
        plot_observed_vs_bayes_bars_two_rows(
            str(OUT_SUMMARY),
            EXP_TAG,
            str(PATH_EXP_DATA),
            save_as=os.path.join(
                OUT_FIGS, f"{EXP_TAG}_bars_observed_vs_bayesrange_base__two_rows.pdf"
            ),
        )

    plot_nrmse_vs_bayes(
        str(OUT_SUMMARY),
        EXP_TAG,
        save_as=os.path.join(OUT_FIGS, f"{EXP_TAG}_scatter_nrmse_vs_bayesianess.pdf"),
    )

    exp = pd.read_csv(PATH_EXP_DATA)
    plot_response_vs_stimulus_grid_all_ablations(
        exp,
        model_pretty_list=(
            "GPT 5 Mini",
            "GPT 4o",
            "Claude 3.7 Sonnet",
            "Llama 4 Maverick",
            "Gemini 2.5 Flash Lite",
            "Mistral 24B",
            "Phi 4 Multimodal",
            "Gemma 3 4B it",
            "Qwen2.5 VL 32B",
        ),
        save_dir=os.path.join(OUT_FIGS, "scatter_plot"),
    )

    # plot_response_vs_stimulus_grid_base_two_models(
    #     exp,
    #     model_pretty_list=(
    #         "GPT 5 Mini",
    #         "GPT 4o",
    #         "Claude 3.7 Sonnet",
    #     ),
    #     save_as=os.path.join(
    #         OUT_FIGS, f"{EXP_TAG}_grid_resp_vs_stim__base__o4_vs_claude.pdf"
    #     ),
    # )

    make_prompt_pages_from_master_one_interaction(
        MASTER_TURNS,
        os.path.join(OUT_FIGS, f"{EXP_TAG}_base_prompt_pages__one_interaction.pdf"),
        title_prefix=f"{EXP_TAG} — Base",
        wrap=76,
    )
    # plot_ratio_scatter_nrmse_vs_bayes(
    #     str(OUT_SUMMARY),
    #     EXP_TAG,
    #     save_as=os.path.join(OUT_FIGS, f"{EXP_TAG}_scatter_ratio_nrmse_vs_bayes.pdf"),
    #     include_base=False,  # set True if you want base included in the averages
    # )

    # plot_observed_vs_bayes_bars(
    #     OUT_SUMMARY,
    #     EXP_TAG,
    #     PATH_EXP_DATA,
    #     save_as=Path(OUT_FIGS) / f"{EXP_TAG}_bars_observed_vs_bayesrange_base.pdf",
    # )


if __name__ == "__main__":
    main()

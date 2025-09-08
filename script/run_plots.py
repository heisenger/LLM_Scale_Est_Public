import sys
import numpy as np
import pandas as pd
from pathlib import Path
from utils.mappings import model_name_map

curr_path = Path(__file__).parent


sys.path.append("../")
import os
import matplotlib.pyplot as plt

from analysis.visualizations import (
    plot_bias_variance_bars_multimodal,
    plot_model_akaike_weights_grid_multimodal,
    plot_response_vs_stimulus_grid,
    plot_group_variance_by_llm_multimodal_lines,
    compute_bias_variance_overall,
)


def run_plots(
    base_path: str,
    experiment_names: list,
    model_list: list,
    model_name_map: dict,
    experiment_label_map: dict,
    dir_name: str = "310725",
):
    experiment_base_path = os.path.join(base_path, "experiment_runs")
    plot_dir = os.path.join(experiment_base_path, dir_name, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Just call the plotting functions with the full experiment_names list
    ax = plot_bias_variance_bars_multimodal(
        model_list=model_list,
        experiment_names=experiment_names,
        experiment_base_path=experiment_base_path,
        compute_bias_variance=compute_bias_variance_overall,
        experiment_label_map=experiment_label_map,
        model_name_map=model_name_map,
    )
    fig = ax.figure
    fig.savefig(os.path.join(plot_dir, "bias_variance_bars.png"))
    plt.close(fig)

    ax = plot_model_akaike_weights_grid_multimodal(
        model_list=model_list,
        experiment_names=experiment_names,
        experiment_base_path=experiment_base_path,
        experiment_label_map=experiment_label_map,
        model_name_map=model_name_map,
    )
    fig = ax.figure
    fig.savefig(os.path.join(plot_dir, "akaike_weights.png"))
    plt.close(fig)

    ax = plot_response_vs_stimulus_grid(
        model_list=model_list,
        experiment_names=experiment_names,
        experiment_base_path=experiment_base_path,
        model_name_map=model_name_map,
        experiment_label_map=experiment_label_map,
        plot_model_predictions=False,
    )
    fig = ax.figure
    fig.savefig(os.path.join(plot_dir, "response_vs_stimulus.png"))
    plt.close(fig)

    ax = plot_response_vs_stimulus_grid(
        model_list=model_list,
        experiment_names=experiment_names,
        experiment_base_path=experiment_base_path,
        model_name_map=model_name_map,
        experiment_label_map=experiment_label_map,
        plot_model_predictions=True,
    )
    fig = ax.figure
    fig.savefig(os.path.join(plot_dir, "response_vs_stimulus_with_predictions.png"))
    plt.close(fig)

    ax = plot_group_variance_by_llm_multimodal_lines(
        model_list=model_list,
        experiment_names=experiment_names,
        experiment_base_path=experiment_base_path,
        model_name_map=model_name_map,
        experiment_label_map=experiment_label_map,
    )
    fig = ax.figure
    fig.savefig(os.path.join(plot_dir, "group_variance_lines.png"))
    plt.close(fig)

    print(f"Plots saved for all experiments in {plot_dir}")


if __name__ == "__main__":
    experiment_names = [
        "070825_" + suffix
        for suffix in ["image_sys_prompt", "text_sys_prompt", "text_image_sys_prompt"]
    ]
    # experiment_names = ["310725_text_image_shifted"]
    label_map = {
        "070825_image_sys_prompt": "Image Only",
        "070825_text_sys_prompt": "Text Only",
        "070825_text_image_sys_prompt": "Image + Text",
    }
    # label_map = {
    #     "310725_text_image_shifted": "Image + Text (Shifted Upwards by 0.1)",
    # }

    model_list = [
        "google/gemini-2.5-flash-lite",  # hundreds of B (lightweight Gemini)
        # "openai/gpt-4.1-mini",  # ~8–10B (GPT-4 mini variant)
        "meta-llama/llama-4-maverick",  # ~17B
        "mistralai/mistral-small-3.2-24b-instruct",  # ~24B
        # "qwen/qwen-vl-plus",  # ~235B total, ~22B activated
        # "openai/gpt-4o-2024-08-06",  # est. ~200B–1T
        # "anthropic/claude-3.7-sonnet",  # ~150–250B
    ]
    experiment_base_path = "experiments/image_studies/maze_distance/"

    # experiment_base_path = "../experiments/image_studies/marker_location/"
    # experiment_base_path = "../experiments/image_studies/line_length_ratio/"

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
    # experiment_base_path = "../experiments/text_studies/subtitle_duration/"

    run_plots(
        base_path=experiment_base_path,
        experiment_names=experiment_names,
        model_list=model_list,
        model_name_map=model_name_map,
        experiment_label_map=label_map,
        dir_name="Maze_Plots_070825",
    )

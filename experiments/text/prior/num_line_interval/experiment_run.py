from typing import List, Dict

from tqdm import tqdm
import re, time
import math
import pandas as pd
import numpy as np

import sys

sys.path.append("../")

from utils.evaluate import magnitude_experiment_run


def pred_extractor(response_text: str) -> int | None:

    def extract_pred_percentage(response_text: str) -> float | None:
        """Extract predicted percentage from LLM response text."""
        # Try extracting explicitly from "Final Answer:"
        matches = re.findall(r"Final Answer:\s*([0-9]*\.?[0-9]+)", response_text)
        if matches:
            return float(matches[-1])  # use last Final Answer if multiple

        # Fallback: extract *last* number anywhere in string
        fallback_matches = re.findall(r"([0-9]*\.?[0-9]+)", response_text)
        if fallback_matches:
            return float(fallback_matches[-1])

        return None

    def convert_to_percentage_if_needed(value: float) -> float:
        """
        Convert decimal to percentage if the value is between 0 and 1.

        Args:
            value: The predicted value

        Returns:
            Value converted to percentage if needed, otherwise original value

        Example:
            0.45 -> 45.0 (converted to percentage)
            45.0 -> 45.0 (already in percentage form)
            0.85 -> 85.0 (converted to percentage)
        """
        if value is None:
            return None

        # If value is between 0 and 1 (exclusive), convert to percentage
        if 0 < value < 1:
            return value * 100
        else:
            return value

    pred_percentage_raw = extract_pred_percentage(response_text)
    return convert_to_percentage_if_needed(pred_percentage_raw)


def data_generator(experiment_df, prompt_string, i):
    """Generate user prompt for the current sample."""

    ground_truth = experiment_df.iloc[i]["actual_percentage"]
    x_start = experiment_df.iloc[i]["x_start"]
    x_end = experiment_df.iloc[i]["x_end"]
    y_start = experiment_df.iloc[i]["y_start"]
    y_end = experiment_df.iloc[i]["y_end"]
    distance_x = experiment_df.iloc[i]["distance_x"]
    distance_y = experiment_df.iloc[i]["distance_y"]
    number_line_max = experiment_df.iloc[i]["number_line_max"]

    prompt_string = prompt_string.format(
        number_line_max=number_line_max,
        x_start=int(x_start),
        x_end=int(x_end),
        y_start=int(y_start),
        y_end=int(y_end),
    )

    input_data = f"x_start: {x_start}, x_end: {x_end}, y_start: {y_start}, y_end: {y_end}, distance_x: {distance_x}, distance_y: {distance_y}"
    return input_data, prompt_string, ground_truth


def run_experiment(
    models: List[str],
    llm_object,
    experiment_file: str = "../experiments/distance_estimation/data/experiment_files/5_20_100samples.csv",
    config: type = None,
) -> pd.DataFrame:
    """
    Args:
        models: List of model names to test
        llm_object: LLM interface class
        experiment_file: Path to CSV file with text samples
        config: Experiment configuration object

    Returns:
        pd.DataFrame with columns: ['model', 'sample_id', 'text', 'ground_truth',
                                   'prediction', 'prompt_sent', 'response']
    """

    results_df = magnitude_experiment_run(
        models=models,
        llm_object=llm_object,
        experiment_file=experiment_file,
        config=config,
        task="number_line_distance_estimation",
        data_generator=data_generator,
        pred_extractor=pred_extractor,
    )

    return results_df

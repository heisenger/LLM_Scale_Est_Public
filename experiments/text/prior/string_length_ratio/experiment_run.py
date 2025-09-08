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
        # Try extracting explicitly from "Final Answer:" - get FIRST match
        matches = re.findall(r"Final Answer:\s*([0-9]*\.?[0-9]+)", response_text)
        if matches:
            return float(matches[0])  # use FIRST Final Answer instead of last

        # Fallback: extract *FIRST* number anywhere in string
        fallback_matches = re.findall(r"([0-9]*\.?[0-9]+)", response_text)
        if fallback_matches:
            return float(fallback_matches[0])  # use FIRST number instead of last

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

    actual_percentage = experiment_df.iloc[i]["actual_percentage"]
    line_a = experiment_df.iloc[i]["line_a"]
    line_b = experiment_df.iloc[i]["line_b"]

    line_a_length = experiment_df.iloc[i]["line_a_length"]
    line_b_length = experiment_df.iloc[i]["line_b_length"]

    # Generate noisy observations
    line_a_1 = int(np.random.normal(line_a_length, line_a_length * 0.1)) * "-"
    line_a_2 = int(np.random.normal(line_a_length, line_a_length * 0.1)) * "-"
    line_a_3 = int(np.random.normal(line_a_length, line_a_length * 0.1)) * "-"

    input_data = f"{line_a} | {line_b}"
    prompt_string = prompt_string.format(
        line_a=line_a,
        line_b=line_b,
        line_a_1=line_a_1,
        line_a_2=line_a_2,
        line_a_3=line_a_3,
    )
    ground_truth = actual_percentage

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
        task="line_length_estimation",
        data_generator=data_generator,
        pred_extractor=pred_extractor,
    )

    return results_df

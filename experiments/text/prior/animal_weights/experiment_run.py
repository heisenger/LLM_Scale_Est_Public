from typing import List, Dict

from tqdm import tqdm
import re, time
import math
import pandas as pd
import numpy as np

import sys

sys.path.append("../")

from utils.evaluate import magnitude_experiment_run


def pred_extractor(response_text: str) -> float | None:

    def extract_pred_position(response_text: str) -> float | None:
        """Extract predicted position from LLM response text."""
        # Try extracting explicitly from "Final Answer:"
        matches = re.findall(r"Final Answer:\s*([-+]?[0-9]*\.?[0-9]+)", response_text)
        if matches:
            return float(matches[-1])  # use last Final Answer if multiple

        # Fallback: extract *last* number (including negative) anywhere in string
        fallback_matches = re.findall(r"([-+]?[0-9]*\.?[0-9]+)", response_text)
        if fallback_matches:
            return float(fallback_matches[-1])

        return None

    pred_position = extract_pred_position(response_text)

    return pred_position


def data_generator(experiment_df, prompt_string, i, block_size=10):
    """Generate user prompt for the current sample."""
    i = i % block_size
    ground_truth = experiment_df.iloc[i]["normalized_weight"]
    animal_name = experiment_df.iloc[i]["animal_name"]

    prompt_string = prompt_string.format(
        animal_name=animal_name,
    )

    return animal_name, prompt_string, ground_truth


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
        task="animal_weight_estimation",
        data_generator=data_generator,
        pred_extractor=pred_extractor,
    )

    return results_df

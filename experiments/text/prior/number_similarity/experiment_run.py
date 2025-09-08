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
    """Extract similarity score from LLM response text."""
    # Try extracting explicitly from "Final Answer:"
    matches = re.findall(r"Final Answer:\s*([0-9]*\.?[0-9]+)", response_text)
    if matches:
        try:
            return float(matches[-1])  # use last Final Answer if multiple
        except ValueError:
            pass

    # Try extracting from common similarity patterns
    similarity_patterns = [
        r"similarity[:\s]+([0-9]*\.?[0-9]+)",
        r"score[:\s]+([0-9]*\.?[0-9]+)",
        r"rating[:\s]+([0-9]*\.?[0-9]+)",
        r"([0-9]*\.?[0-9]+)\s*out\s*of\s*10",
        r"([0-9]*\.?[0-9]+)\s*/\s*10",
    ]

    for pattern in similarity_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue

    # Fallback: extract *last* number anywhere in string
    fallback_matches = re.findall(r"([0-9]*\.?[0-9]+)", response_text)
    if fallback_matches:
        try:
            return float(fallback_matches[-1])
        except ValueError:
            pass

    return None


def data_generator(experiment_df, prompt_string, i):
    """Generate user prompt for the current sample."""

    row = experiment_df.iloc[i]
    number_1 = row["number_1"]
    number_2 = row["number_2"]

    input_data = f"Number 1: {number_1}, Number 2: {number_2}"
    prompt_string = prompt_string.format(number_1=number_1, number_2=number_2)
    ground_truth = min(number_1, number_2) / max(number_1, number_2)

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
        task="number_similarity_estimation",
        data_generator=data_generator,
        pred_extractor=pred_extractor,
    )

    return results_df

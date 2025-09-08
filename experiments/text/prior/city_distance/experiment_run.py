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
    """Extract predicted distance from LLM response text."""
    # Try extracting explicitly from "Final Answer:" - handle commas
    matches = re.findall(r"Final Answer:\s*([0-9,]+\.?[0-9]*)", response_text)
    if matches:
        # Remove commas and convert to float
        number_str = matches[-1].replace(",", "").strip()
        if number_str:  # Check if not empty after cleaning
            try:
                return float(number_str)
            except ValueError:
                pass

    # Fallback: extract *last* number anywhere in string - handle commas
    fallback_matches = re.findall(r"([0-9,]+\.?[0-9]*)", response_text)
    if fallback_matches:
        number_str = fallback_matches[-1].replace(",", "").strip()
        if number_str:  # Check if not empty after cleaning
            try:
                return float(number_str)
            except ValueError:
                pass

    return None


def data_generator(experiment_df, prompt_string, i):
    """Generate user prompt for the current sample."""

    # 1. Extract the data for the current route
    row = experiment_df.iloc[i]
    city_chain_str = row["city_chain"]
    country_chain_str = row["country_chain"]
    actual_total_distance = row["total_distance"]

    # 2. Process the chains into a readable format
    cities = city_chain_str.split(", ")
    countries = country_chain_str.split(", ")

    # Combine city and country for clarity, e.g., "Libreville, Gabon"
    full_stops = [f"{city}, {country}" for city, country in zip(cities, countries)]

    # Create a final route description string, e.g., "Libreville, Gabon -> Accra, Ghana -> ..."
    route_description = " -> ".join(full_stops)

    actual_distance = experiment_df.iloc[i]["total_distance"]

    prompt_string = prompt_string.format(route_description=route_description)

    input_data = route_description

    return input_data, prompt_string, actual_distance


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
        task="city_distance_estimation",
        data_generator=data_generator,
        pred_extractor=pred_extractor,
    )

    return results_df

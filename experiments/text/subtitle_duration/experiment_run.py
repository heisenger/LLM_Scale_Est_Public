from typing import List, Dict

# from .data_generator import extract_speech_text_from_vtt
from tqdm import tqdm
import re, time
import math
import pandas as pd
import numpy as np

import sys

sys.path.append("../")

from utils.evaluate import magnitude_experiment_run, pred_extractor


def data_generator(
    experiment_df, sys_prompt_string, usr_prompt_string, i, block_size=10
):
    """Generate user prompt for the current sample."""
    i = i % block_size
    input_data = experiment_df.iloc[i]["text"]
    usr_prompt_string = usr_prompt_string.format(sentence=input_data)
    ground_truth = experiment_df.iloc[i]["actual_duration"]

    return input_data, sys_prompt_string, usr_prompt_string, ground_truth, ground_truth


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
        task="speech_duration_estimation",
        data_generator=data_generator,
        pred_extractor=pred_extractor,
        multi_modal=config.multi_modal,
    )

    return results_df

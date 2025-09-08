from typing import List, Dict, Optional
import base64

from tqdm import tqdm
import re, time
import math
import pandas as pd
import numpy as np
import os
from pathlib import Path
from functools import partial
import ast

from utils.evaluate import magnitude_experiment_run, pred_extractor
from utils.audio_process import mp3_to_base64

file_path = Path(__file__)  # or Path("your/file/path.py")
parent_path = file_path.parent


def data_generator(
    experiment_df,
    prompt_string,
    i,
    block_size=10,
    multi_modal: bool = False,
    process_factor: float = 0.0,
    audio_mode: str = "normal",  # "blurred"
    experiment_file: str = "normal",
    numerical_steer: bool = False,
):
    """Generate user prompt for the current sample."""
    i = i % block_size
    ground_truth = experiment_df.iloc[i]["actual_length"]
    input_data = ground_truth
    input_value = ground_truth

    audio_file = "audio_path"

    format_kwargs = {"text_representation": experiment_df.iloc[i].text}
    # if process_factor != 0:
    #     format_kwargs["process_factor"] = process_factor
    #     input_value = ground_truth * (1 + process_factor)

    if numerical_steer:
        match = re.search(r"/([\d.]+_[\d.]+)/", str(experiment_file))
        if match:
            range_str = match.group(1)  # e.g., '0.3_0.8'
            start, end = map(float, range_str.split("_"))
            prior = f"{start} to {end}"
        format_kwargs["value_range"] = prior

    if multi_modal:
        audio_uri = mp3_to_base64(
            f"{parent_path}/data/experiment_files/{experiment_df.iloc[i][audio_file]}"
        )
        format_kwargs["audio_uri"] = audio_uri
        prompt_string = prompt_string.format(**format_kwargs)
        return input_data, prompt_string, audio_uri, ground_truth, input_value

    else:
        prompt_string = prompt_string.format(**format_kwargs)
        return input_data, prompt_string, ground_truth, input_value


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
        task="line_length_ratio",
        multi_modal_type="audio",
        data_generator=partial(
            data_generator,
            multi_modal=config.multi_modal,
            audio_mode=config.audio_mode,
            experiment_file=experiment_file,
            numerical_steer=config.numerical_steer,
        ),
        pred_extractor=pred_extractor,
        multi_modal=config.multi_modal,
    )

    return results_df

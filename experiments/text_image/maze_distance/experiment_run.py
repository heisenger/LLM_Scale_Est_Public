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

file_path = Path(__file__)  # or Path("your/file/path.py")
parent_path = file_path.parent

from utils.evaluate import magnitude_experiment_run, pred_extractor
from utils.image_process import get_image_base64_data_uri


def data_generator(
    experiment_df,
    sys_prompt_string,
    usr_prompt_string,
    i,
    block_size=10,
    multi_modal: bool = False,
    process_factor: float = 0.0,
    provide_answer: bool = False,
    text_mode: str = "path_text",
    image_mode: str = "normal",  # "blurred"
    experiment_file: str = "normal",
    numerical_steer: bool = False,
    prior_steer: bool = False,
    reverse_order: bool = False,
):
    """Generate user prompt for the current sample."""
    i = i % block_size
    if reverse_order:
        i = block_size - 1 - i
    ground_truth = experiment_df.iloc[i]["actual_length"]
    if i > 0:
        ground_truth_prior = experiment_df.iloc[i - 1]["actual_length"]
    input_data = ground_truth
    input_value = ground_truth

    # Choose the right image
    if image_mode == "blurred":
        image_file = "blurred_image_path"
    elif image_mode == "blurred_low":
        image_file = "blurred_low_image_path"
    elif image_mode == "blurred_high":
        image_file = "blurred_high_image_path"
    elif image_mode == "blurred_low_sequence":
        if i > 6:
            image_file = "blurred_low_image_path"
        else:
            image_file = "image_path"
    elif image_mode == "blurred_high_sequence":
        if i > 6:
            image_file = "blurred_high_image_path"
        else:
            image_file = "image_path"
    elif image_mode == "blurred_mixed_sequence":
        if i > 7:
            image_file = "blurred_high_image_path"
        elif i > 4:
            image_file = "blurred_image_path"
        elif i > 1:
            image_file = "blurred_low_image_path"
        else:
            image_file = "image_path"
    else:
        image_file = "image_path"

    if text_mode == "path_text":
        format_kwargs = {
            "text_representation": experiment_df.iloc[i]["path_text_description"]
        }
    else:
        format_kwargs = {"text_representation": experiment_df.iloc[i]["ascii_path"]}

    if numerical_steer or prior_steer:
        match = re.search(r"/([\d.]+_[\d.]+)/", str(experiment_file))
        if match:
            range_str = match.group(1)  # e.g., '0.3_0.8'
            start, end = map(float, range_str.split("_"))
            mid_point = (start + end) / 2

            if prior_steer:
                end = mid_point

            prior = f"{start} to {end}"
            sys_prompt_string = sys_prompt_string.format(value_range=prior)

    if multi_modal:
        image_uri = get_image_base64_data_uri(
            f"{parent_path}/data/experiment_files/{experiment_df.iloc[i][image_file]}"
        )
        format_kwargs["image_uri"] = image_uri
        usr_prompt_string = usr_prompt_string.format(**format_kwargs)
        if provide_answer and i > 0:
            usr_prompt_string += f"\n\nThe ground truth answer of the previous interaction was: {ground_truth_prior}.\n"
        return (
            input_data,
            sys_prompt_string,
            usr_prompt_string,
            image_uri,
            ground_truth,
            input_value,
        )
    else:
        usr_prompt_string = usr_prompt_string.format(**format_kwargs)
        if provide_answer and i > 0:
            usr_prompt_string += f"\n\nThe ground truth answer of the previous interaction was: {ground_truth_prior}.\n"

        return (
            input_data,
            sys_prompt_string,
            usr_prompt_string,
            ground_truth,
            input_value,
        )


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
        task="maze_distance",
        data_generator=partial(
            data_generator,
            multi_modal=config.multi_modal,
            provide_answer=config.provide_answer,
            text_mode=config.text_mode,
            image_mode=config.image_mode,
            experiment_file=experiment_file,
            numerical_steer=config.numerical_steer,
            prior_steer=config.prior_steer,
            reverse_order=config.reverse_order,
        ),
        pred_extractor=pred_extractor,
        multi_modal=config.multi_modal,
    )

    return results_df

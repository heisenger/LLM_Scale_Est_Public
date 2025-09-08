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
from utils.image_process import get_image_base64_data_uri

file_path = Path(__file__)  # or Path("your/file/path.py")
parent_path = file_path.parent


def data_generator(
    experiment_df,
    sys_prompt_string,
    usr_prompt_string,
    i,
    block_size=10,
    ascii_line_choice: str = "ascii_line",
    multi_modal: bool = False,
    process_factor: float = 0.0,
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

    format_kwargs = {
        "text_representation_1": ast.literal_eval(
            experiment_df.iloc[i][ascii_line_choice]
        )[0],
        "text_representation_2": ast.literal_eval(
            experiment_df.iloc[i][ascii_line_choice]
        )[1],
    }
    if process_factor != 0:
        format_kwargs["process_factor"] = process_factor
        input_value = ground_truth * (1 + process_factor)

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
        task="line_length_ratio",
        data_generator=partial(
            data_generator,
            ascii_line_choice=config.ascii_line_choice,
            multi_modal=config.multi_modal,
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

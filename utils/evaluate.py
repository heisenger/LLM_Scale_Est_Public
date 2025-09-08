import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import pdb
import re
from pathlib import Path
import ast
import json
from configs.experiments.config import CommonConfig

base_dir = Path(__file__).parent  # Directory where this script sits


def magnitude_experiment_run(
    models: List[str],
    llm_object,
    experiment_file: str = "../experiments/distance_estimation/data/experiment_files/5_20_100samples.csv",
    config: type = None,
    task="character_count_estimation",
    data_generator=None,
    pred_extractor=None,
    multi_modal=False,
    multi_modal_type="image",  # either image or audio
    reasoning_flag=False,
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

    experiment_df = pd.read_csv(experiment_file)
    all_results = []

    for model in models:
        print(f"Running: {model}")

        # Initialize LLM request function
        request_fn = llm_object(
            provider=config.llm_provider,
            model=model,
            task=task,
        ).ask

        # Initialize conversation history
        conversation_history = []

        for i in tqdm(range(config.num_samples), desc=f"Samples for {model}"):

            # Add current user prompt. Include image_uri where true
            if multi_modal:
                (
                    input_data,
                    sys_prompt_text,
                    user_prompt_text,
                    media,
                    ground_truth,
                    input_value,
                ) = data_generator(
                    experiment_df,
                    config.system_prompt,
                    config.user_prompt_zero_shot,
                    i,
                    block_size=config.block_size,
                )

                # Build current message
                current_messages = [{"role": "system", "content": sys_prompt_text}]

                # Add conversation history if accumulating
                if config.accumulate_context and conversation_history:
                    current_messages.extend(conversation_history)

                if multi_modal_type == "image":
                    current_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt_text},
                                {"type": "image_url", "image_url": {"url": media}},
                            ],
                        }
                    )
                elif multi_modal_type == "audio":
                    current_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt_text},
                                {
                                    "type": "input_audio",
                                    "input_audio": {"data": media, "format": "mp3"},
                                },
                            ],
                        }
                    )
            else:

                (
                    input_data,
                    sys_prompt_text,
                    user_prompt_text,
                    ground_truth,
                    input_value,
                ) = data_generator(
                    experiment_df,
                    config.system_prompt,
                    config.user_prompt_zero_shot,
                    i,
                    block_size=config.block_size,
                )
                # Build current message
                current_messages = [{"role": "system", "content": sys_prompt_text}]

                # Add conversation history if accumulating
                if config.accumulate_context and conversation_history:
                    current_messages.extend(conversation_history)

                current_messages.append({"role": "user", "content": user_prompt_text})

            message_sent, response_text, _, reasoning = request_fn(
                messages=current_messages
            )

            # Extract prediction
            pred_clean = pred_extractor(response_text)

            # modify response text and prediction if necessary
            response_text_appended = response_text

            # Check if LLM is reasoning when told not to
            if (len(response_text.split()) > 10) and not reasoning_flag:
                # Further check if the answer is well formed. Filter out non-well-formed answers
                if "Final Answer" not in response_text:
                    response_text_appended = "You responded with reasoning. Do not respond with reasoning, just give the final answer."
                    pred_clean = (
                        np.nan
                    )  # Set prediction to NaN if answer is not well formed

            # Accumulate this for the next round
            if config.accumulate_context:
                if (
                    len(conversation_history) + 2 >= config.context_window * 2
                ):  # the last time a message is appended is when the current message is the 10th
                    conversation_history = (
                        []
                    )  # clear out conversation history once we hit this point.
                else:
                    if multi_modal:
                        if multi_modal_type == "image":
                            conversation_history.extend(
                                [
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": user_prompt_text},
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": media,
                                                },
                                            },
                                        ],
                                    },
                                    {
                                        "role": "assistant",
                                        "content": response_text_appended,
                                    },
                                ]
                            )
                        elif multi_modal_type == "audio":
                            conversation_history.extend(
                                [
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": user_prompt_text},
                                            {
                                                "type": "input_audio",
                                                "input_audio": {
                                                    "data": media,
                                                    "format": "mp3",
                                                },
                                            },
                                        ],
                                    },
                                    {
                                        "role": "assistant",
                                        "content": response_text_appended,
                                    },
                                ]
                            )

                    else:
                        conversation_history.extend(
                            [
                                {"role": "user", "content": user_prompt_text},
                                {
                                    "role": "assistant",
                                    "content": response_text_appended,
                                },
                            ]
                        )

            # Store result for this sample
            result = {
                "model": model,
                "sample_id": experiment_df.iloc[i % config.block_size].get(
                    "sample_id", f"sample_{i+1:03d}"
                ),
                "text": input_data,
                CommonConfig.true_column: ground_truth,
                CommonConfig.input_column: input_value,
                CommonConfig.pred_column: pred_clean,
                "prompt_sent": json.dumps(message_sent),
                "response": response_text,
                "reasoning": reasoning,
            }

            all_results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    return results_df


def pred_extractor(response_text: str) -> float | None:
    """Extracts a number from response_text, or returns None if error/failed is present."""
    if not isinstance(response_text, str):
        return None
    lowered = response_text.lower()
    if "error" in lowered or "failed" in lowered:
        return None

    matches = re.findall(r"Final Answer:\s*([0-9]*\.?[0-9]+)", response_text)
    if matches:
        return float(matches[0])

    fallback_matches = re.findall(r"([0-9]*\.?[0-9]+)", response_text)
    if fallback_matches:
        return float(fallback_matches[0])

    return None


def load_data(
    experiment="maze_distance",
    experiment_run="060825_text_image",
    experiment_type="text_image",
    model_name="anthropic_claude-3.7-sonnet",
    full_path_short=None,
    full_path_medium=None,
    full_path_long=None,
):
    """
    Loads experimental data and 1) return DataFrames for short, medium, and long range categories 2) interactions DataFrame
    """
    if full_path_short:
        short_df = pd.read_csv(full_path_short)
    if full_path_medium:
        medium_df = pd.read_csv(full_path_medium)
    if full_path_long:
        long_df = pd.read_csv(full_path_long)

    else:
        base_path = f"{base_dir}/../experiments/{experiment_type}/{experiment}/runs/prior/{experiment_run}"
        config_path = f"{base_path}/configs/used_experiment_configs.json"

        with open(config_path, "r") as f:
            config_json = json.load(f)
        range_cats = [config_json[0]["experiment_files"][i] for i in range(3)]
        short_df, medium_df, long_df = (pd.read_csv(cat) for cat in range_cats)

    interactions_path = [
        f"{base_path}/{model_name}/results_{Path(range_cats[i]).name}" for i in range(3)
    ]
    short_interaction, medium_interaction, long_interaction = (
        pd.read_csv(path) for path in interactions_path
    )

    return (
        short_df,
        medium_df,
        long_df,
        short_interaction,
        medium_interaction,
        long_interaction,
    )


def print_interaction_details(interactions, idx=0, turn_idx=-1):
    """
    Print ground truth, text prompt, and image URL for a given interaction index.
    Only prints image URL if present.

    Args:
        interactions (pd.DataFrame): DataFrame with 'ground_truth' and 'prompt_sent' columns.
        idx (int): Row index in the DataFrame.
        turn_idx (int): Which turn in the prompt_response to use (default: last).
    """
    gt = interactions.ground_truth.iloc[idx]
    prompt_response = evaluate_list_dict(interactions.prompt_sent.iloc[idx])
    turn = prompt_response[turn_idx]["content"]
    print("Ground truth:", gt)
    print(turn)
    if isinstance(turn, list):
        text_prompt = turn[0]["text"]
        img_url = turn[1]["image_url"]


# Function to evaluate list in string
def evaluate_list_num(s):
    arr = np.fromstring(s.strip("[]"), sep=" ")
    lst = arr.tolist()
    return lst


# Function to evaluate list in string
def evaluate_list_dict(s):
    d = ast.literal_eval(s)
    return d


def evaluate_list_tuple(s):
    path = ast.literal_eval(s)
    # print(path)
    return path

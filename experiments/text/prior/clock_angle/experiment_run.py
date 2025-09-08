from typing import List, Dict
from .data_generator import random_time_with_angle
from tqdm import tqdm
import re, time
import math


def extract_pred_angle(response_text: str) -> float | None:
    # Try extracting explicitly from "Final Answer:"
    matches = re.findall(r"Final Answer:\s*([0-9]*\.?[0-9]+)", response_text)
    if matches:
        return float(matches[-1])  # use last Final Answer if multiple

    # Fallback: extract *last* number anywhere in string
    fallback_matches = re.findall(r"([0-9]*\.?[0-9]+)", response_text)
    if fallback_matches:
        return float(fallback_matches[-1])

    return None


def generate_few_shot_examples(
    num_examples: int = 3,
    min_angle: float = 10,
    max_angle: float = 100.0,
    adjustment_factor: float = 1.0,
):
    """Generate few-shot examples for speech duration estimation."""
    examples = []

    for i in range(num_examples):
        # Generate diverse angles across the range
        example_angle = min_angle + (max_angle - min_angle) * i / (num_examples - 1)

        # Get example (hour, min) pairs
        example_time = random_time_with_angle(example_angle)

        # Adjust duration by a factor to create variability in the prior
        example_angle *= adjustment_factor

        # Format the example
        example = f"Pair: (h, m): ({example_time[0]}, {example_time[1]})\n Final Answer: {example_angle:.2f} degrees"
        examples.append(example)

    return examples


def run_angle_estimation(
    models: List[str],
    llm_object,
    llm_params: Dict = None,
    num_samples: int = 30,
    min_angle: float = 1.0,
    max_angle: float = 179.0,
    verbose: bool = False,
    few_shot_switch: bool = True,
    adjustment_factor: float = 1.0,
    prompt_dict: dict = None,
) -> Dict:

    all_responses = {}
    test_sentences = {}
    invalid_flags = {}
    true_times = {}
    pred_times = {}

    for model in models:
        print(f"Running for model: {model}")
        all_responses[model] = []
        test_sentences[model] = []
        true_times[model] = []
        pred_times[model] = []
        invalid_flags[model] = []
        request_fn = llm_object(
            provider=llm_params["llm_provider"], model=model, task="angle_estimation"
        ).ask

        for i in tqdm(range(num_samples), desc=f"Samples for {model}"):
            # Compute target_angle logarithmically spaced between min_angle and max_angle
            target_angle = min_angle + (max_angle - min_angle) * i / (num_samples - 1)
            target_pair = random_time_with_angle(target_angle, tolerance=1)
            if few_shot_switch:
                few_shot_examples = generate_few_shot_examples(
                    adjustment_factor=adjustment_factor,
                )
                few_shot_prompt = "\n\n".join(few_shot_examples)
                prompt = prompt_dict["user_prompt_few_shots"].format(
                    few_shot_examples=few_shot_prompt,
                    hour=target_pair[0],
                    minute=target_pair[1],
                )
            else:
                prompt = prompt_dict["user_prompt_zero_shot"].format(
                    hour=target_pair[0],
                    minute=target_pair[1],
                )

            message_sent, response_text, _ = request_fn(
                prompt, system_prompt=prompt_dict["system_prompt"]
            )
            first_word = response_text.strip().split()[0]

            while first_word == "Error":
                if verbose:
                    print(f"Invalid response for model {model}: {response_text}")
                time.sleep(10)
                invalid_flags[model].append((target_angle, response_text))
                message_sent, response_text, _ = request_fn(
                    prompt, system_prompt=prompt_dict["system_prompt"]
                )
                first_word = response_text.strip().split()[0]

            pred_time = extract_pred_angle(response_text)
            true_times[model].append(target_angle)
            pred_times[model].append(pred_time)
            test_sentences[model].append(message_sent)
            all_responses[model].append(response_text)

            time.sleep(2.1)

    return {
        "responses": all_responses,
        "true_times": true_times,
        "pred_times": pred_times,
        "sentences": test_sentences,
        "invalids": invalid_flags,
    }

import pandas as pd
import numpy as np


def clean_experiment_df(
    result_list,
    pred_column="predicted_duration",
    true_column="true_duration",
    input_column="input_values",
    block_size=10,
):
    """
    Returns a dataframe with the following columns:
    - correct: Ground truth of each stimulus
    - response: Prediction of model
    - range_category: Category / context of stimulus (short, medium, long)
    - trial: Sequential trial numbers

    """
    # Combine all results from the three experiments
    all_results = []

    range_categories = ["short", "medium", "long"]

    for i, results_df in enumerate(result_list):
        # Add range category based on which experiment file was used
        results_df["range_category"] = range_categories[i]
        results_df["trial"] = range(len(results_df))  # Sequential trial numbers
        results_df["stimulus_id"] = results_df["trial"] % block_size
        all_results.append(results_df)

    # Combine all DataFrames
    combined_results = pd.concat(all_results, ignore_index=True)
    # Create the experimental data DataFrame for psychophysics analysis
    experimental_data = pd.DataFrame(
        {
            "correct": combined_results[true_column],
            "response": combined_results[pred_column],
            "input_value": combined_results[input_column],
            "range_category": combined_results["range_category"],
            "trial": combined_results["trial"],  # Sequential trial numbers
            "stimulus_id": combined_results["stimulus_id"],
        }
    )

    return experimental_data


def convert_numpy_types(obj):
    """Recursively convert numpy types in a dict/list to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

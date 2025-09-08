import os
import numpy as np
import pandas as pd


def generate_number_line_distance_samples(
    x_start: float = 10.0,
    x_range: float = 20.0,
    y_start: float = 800.0,
    y_range: float = 200.0,
    n_samples: int = 100,
    number_line_max: int = 1000,
    output_dir: str = "../experiments/number_line_estimation/data/experiment_files",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate samples of number line distance estimation tasks.
    Creates two distance intervals where distance X is a certain percentage of distance Y.
    X distance (smaller numbers) will be SMALLER than Y distance (larger numbers).

    Args:
        min_percentage: Minimum percentage that X is of Y (e.g., 10.0)
        max_percentage: Maximum percentage that X is of Y (e.g., 90.0)
        n_samples: Number of samples to generate
        number_line_max: Maximum value on the number line (default: 1000)
        output_dir: Directory to save CSV file
        seed: Random seed for reproducibility

    Returns:
        pd.DataFrame: Generated samples with columns:
            - sample_id: Unique identifier for each sample
            - target_percentage: Target percentage (X/Y * 100)
            - x_start: Start point of distance X (smaller numbers)
            - x_end: End point of distance X (smaller numbers)
            - y_start: Start point of distance Y (larger numbers)
            - y_end: End point of distance Y (larger numbers)
            - distance_x: Actual distance X (SMALLER)
            - distance_y: Actual distance Y (LARGER)
            - actual_percentage: Actual percentage (X/Y * 100)

    Example:
        samples = generate_number_line_distance_samples(200.0, 800.0, 100)
        # X might be from 10-30 (distance=20), Y might be from 800-960 (distance=160)
        # Y/X = 160/20 = 800% - Y is much bigger than X
    """

    np.random.seed(seed)

    # Uniformly sample target percentages
    x_size_range = np.random.uniform(0, 1, n_samples)

    samples = []

    for i, x_size in enumerate(x_size_range):
        x_end = x_start + x_size * x_range

        actual_percentage = np.random.uniform(0, 1)
        y_end = y_start + x_size * x_range / actual_percentage

        samples.append(
            {
                "sample_id": f"sample_{i+1:03d}",
                "actual_percentage": round(actual_percentage, 2) * 100,
                "x_start": x_start,
                "x_end": x_end,
                "y_start": y_start,
                "y_end": y_end,
                "distance_x": x_end - x_start,
                "distance_y": y_end - y_start,
                "number_line_max": number_line_max,
            }
        )

    # Create DataFrame
    samples_df = pd.DataFrame(samples)

    # Save to CSV with informative filename
    os.makedirs(output_dir, exist_ok=True)
    filename = f"number_line_distance_{x_range}_{y_range}_{n_samples}samples.csv"
    filepath = os.path.join(output_dir, filename)
    samples_df.to_csv(filepath, index=False)

    print(f"Generated {len(samples)} number line distance samples")
    print(f"Saved to: {filepath}")

    return samples_df

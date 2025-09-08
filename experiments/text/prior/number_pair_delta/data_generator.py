import os
import numpy as np
import pandas as pd


def generate_number_pair_samples(
    min_delta: float,
    max_delta: float,
    n_samples: int,
    min_base: float = 10_000,
    max_base: float = 90_000,
    output_dir: str = "../experiments/number_size_estimation/data/experiment_files",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate samples of number pairs (A, B) where B - A equals a uniformly sampled delta,
    and A is randomly sampled from a specified range.

    Args:
        min_delta: Minimum difference between B and A
        max_delta: Maximum difference between B and A
        n_samples: Number of samples to generate
        min_base: Minimum value for the smaller number A (default: 1)
        max_base: Maximum value for the smaller number A (default: 1,000,000)
        output_dir: Directory to save CSV file
        seed: Random seed for reproducibility

    Returns:
        pd.DataFrame: Generated samples with columns:
            - sample_id: Unique identifier for each sample
            - number_a: The smaller number
            - number_b: The larger number (A + delta)
            - delta: The difference (B - A)
            - ratio: The ratio B/A for analysis

    Example:
        samples = generate_number_pair_samples(1000, 5000, 100)
        # Generates pairs like: A=123456, B=126789, delta=3333
        # Saves to: ../experiments/number_size_estimation/data/experiment_files/1000_5000_100samples.csv
    """

    np.random.seed(seed)

    # Uniformly sample deltas
    target_deltas = np.random.uniform(min_delta, max_delta, n_samples)

    # Uniformly sample base numbers (A values)
    base_numbers = np.random.uniform(min_base, max_base, n_samples)

    samples = []

    for i, (base_number, delta) in enumerate(zip(base_numbers, target_deltas)):
        number_a = round(base_number)  # Round to integer
        number_b = number_a + round(delta)  # Add delta to get B
        actual_delta = number_b - number_a
        ratio = number_b / number_a if number_a != 0 else float("inf")

        samples.append(
            {
                "sample_id": f"sample_{i+1:03d}",
                "number_pair": (number_a, number_b),
                "delta": actual_delta,
                # "ratio": round(ratio, 4),
            }
        )

    # Create DataFrame
    samples_df = pd.DataFrame(samples)

    # Save to CSV with informative filename
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{int(min_base)}_{int(max_base)}_{n_samples}samples.csv"
    filepath = os.path.join(output_dir, filename)
    samples_df.to_csv(filepath, index=False)

    print(f"Generated {len(samples)} number pair samples")
    print(f"Delta range: {min_delta:,} - {max_delta:,}")
    print(f"Base number range: {min_base:,} - {max_base:,}")
    print(f"Saved to: {filepath}")

    return samples_df

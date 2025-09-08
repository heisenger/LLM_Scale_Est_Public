import os
import numpy as np
import pandas as pd


def generate_line_length_samples(
    min_length: float = 10.0,
    max_length: float = 90.0,
    n_samples: int = 100,
    metric_length: int = 3,  # number of dashes used to represent a line of length 5
    output_dir: str = "../experiments/line_length/data/experiment_files",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate samples for line length estimation tasks.

    Args:
        min_length (float): Minimum target line length.
        max_length (float): Maximum target line length.
        n_samples (int): Number of samples to generate.
        metric_length (int): Number of dashes representing a unit length.
        output_dir (str): Directory to save the generated CSV file.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing generated samples with columns:
            - sample_id: Unique identifier for each sample.
            - target_length: Target line length (float).
            - actual_length: Actual line length used (float).
            - line_a_length: Length of line A (int).
            - five_unit_metric_length: Metric unit length (int).
            - line_a: Visualization of line A (str).
            - metric: Visualization of metric unit (str).

    Example:
        samples = generate_line_length_samples(20.0, 80.0, 100)
        # Generates samples with line lengths between 20.0 and 80.0
        # Saves to ../experiments/line_length/data/experiment_files/line_length_20.0_80.0_100samples.csv
    """

    np.random.seed(seed)

    # Uniformly sample target lengths
    target_lengths = np.random.uniform(min_length, max_length, n_samples)

    samples = []

    for i, target_length in enumerate(target_lengths):
        # Calculate line A length based on target length
        line_a_length = int(round(target_length))

        # Generate the actual lines
        line_a = "-" * int(line_a_length / 5 * metric_length)

        samples.append(
            {
                "sample_id": f"sample_{i+1:03d}",
                "target_length": round(target_length, 2),
                "actual_length": round(target_length, 2),
                "line_a_length": line_a_length,
                "five_unit_metric_length": metric_length,
                "line_a": line_a,
                "metric": "-" * metric_length,
            }
        )

    # Create DataFrame
    samples_df = pd.DataFrame(samples)

    # Save to CSV with informative filename
    os.makedirs(output_dir, exist_ok=True)
    filename = f"line_length_{min_length}_{max_length}_{n_samples}samples.csv"
    filepath = os.path.join(output_dir, filename)
    samples_df.to_csv(filepath, index=False)

    print(f"Generated {len(samples)} line length samples")
    print(f"Length range: {min_length} - {max_length}")
    print(f"Saved to: {filepath}")

    return samples_df


if __name__ == "__main__":
    # Example usage
    range_1 = (5.0, 20.0)
    range_2 = (15.0, 50.0)
    range_3 = (30.0, 100.0)

    for r in [range_1, range_2, range_3]:
        generate_line_length_samples(
            min_length=r[0],
            max_length=r[1],
            n_samples=100,
            metric_length=3,
            output_dir="../experiments/line_length/data/experiment_files/",
            seed=42,
        )

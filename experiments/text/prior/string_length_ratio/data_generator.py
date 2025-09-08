import os
import numpy as np
import pandas as pd


def generate_line_length_samples(
    min_percentage: float = 10.0,
    max_percentage: float = 90.0,
    n_samples: int = 100,
    line_b_length: int = 50,
    output_dir: str = "../experiments/line_len_estimation/data/experiment_files",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate samples of line length estimation tasks.
    Creates two lines where line A is a certain percentage of line B's length.

    Args:
        min_percentage: Minimum percentage that line A is of line B (e.g., 10.0)
        max_percentage: Maximum percentage that line A is of line B (e.g., 90.0)
        n_samples: Number of samples to generate
        line_b_length: Length of line B in characters (default: 50)
        output_dir: Directory to save CSV file
        seed: Random seed for reproducibility

    Returns:
        pd.DataFrame: Generated samples with columns:
            - sample_id: Unique identifier for each sample
            - target_percentage: Target percentage (A/B * 100)
            - line_a_length: Length of line A in characters
            - line_b_length: Length of line B in characters
            - line_a: Line A visualization (dashes)
            - line_b: Line B visualization (dashes)

    Example:
        samples = generate_line_length_samples(20.0, 80.0, 100)
        # Generates line pairs where A is 20-80% of B's length
        # Saves to: ../experiments/line_len_estimation/data/experiment_files/20.0_80.0_100samples.csv
    """

    np.random.seed(seed)

    # Uniformly sample target percentages
    target_percentages = np.random.uniform(min_percentage, max_percentage, n_samples)

    samples = []

    for i, target_percent in enumerate(target_percentages):
        # Calculate line A length based on percentage
        line_a_length = int(round((target_percent / 100.0) * line_b_length))

        # Ensure line A has at least 1 character
        line_a_length = max(1, line_a_length)

        # Generate the actual lines
        line_a = "-" * line_a_length
        line_b = "-" * line_b_length

        # Calculate actual percentage (might differ slightly due to rounding)
        actual_percentage = (line_a_length / line_b_length) * 100

        samples.append(
            {
                "sample_id": f"sample_{i+1:03d}",
                "target_percentage": round(target_percent, 2),
                "actual_percentage": round(actual_percentage, 2),
                "line_a_length": line_a_length,
                "line_b_length": line_b_length,
                "line_a": line_a,
                "line_b": line_b,
            }
        )

    # Create DataFrame
    samples_df = pd.DataFrame(samples)

    # Save to CSV with informative filename
    os.makedirs(output_dir, exist_ok=True)
    filename = f"line_length_{min_percentage}_{max_percentage}_{n_samples}samples.csv"
    filepath = os.path.join(output_dir, filename)
    samples_df.to_csv(filepath, index=False)

    print(f"Generated {len(samples)} line length samples")
    print(f"Percentage range: {min_percentage}% - {max_percentage}%")
    print(f"Line B length: {line_b_length} characters")
    print(
        f"Line A length range: {samples_df['line_a_length'].min()} - {samples_df['line_a_length'].max()} characters"
    )
    print(f"Saved to: {filepath}")

    return samples_df

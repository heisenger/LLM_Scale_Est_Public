import os
import numpy as np
import pandas as pd


def generate_line_position_samples(
    min_position_abs: float = 0.2,
    max_position_abs: float = 1.0,
    n_samples: int = 100,
    line_length: int = 13,
    output_dir: str = "../experiments/location_on_line/data/experiment_files",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate samples of horizontal line position estimation tasks.
    Samples positions from both positive and negative ranges based on absolute values.

    Args:
        min_position_abs: Minimum absolute position (e.g., 0.2 -> samples from 0.2 to max_position_abs and -0.2 to -max_position_abs)
        max_position_abs: Maximum absolute position (e.g., 1.0 -> samples up to ±1.0)
        n_samples: Number of samples to generate
        line_length: Total length of the line in characters (default: 13)
        output_dir: Directory to save CSV file
        seed: Random seed for reproducibility

    Returns:
        pd.DataFrame: Generated samples with columns:
            - sample_id: Unique identifier for each sample
            - target_position: Target position (-max_position_abs to +max_position_abs)
            - reference_line: Line with just dashes
            - fixation_line: Line with fixation marker "+"
            - marker_line: Line with both "+" and "*"

    Example:
        samples = generate_line_position_samples(0.2, 1.0, 100)
        # Generates positions from [0.2, 1.0] and [-1.0, -0.2]
        # Excludes the range [-0.2, 0.2] around zero
    """

    np.random.seed(seed)

    # Generate target positions from both positive and negative ranges
    target_positions = []

    # Split samples between positive and negative ranges
    n_positive = n_samples // 2
    n_negative = n_samples - n_positive

    # Sample from positive range [min_position_abs, max_position_abs]
    positive_positions = np.random.uniform(
        min_position_abs, max_position_abs, n_positive
    )

    # Sample from negative range [-max_position_abs, -min_position_abs]
    negative_positions = np.random.uniform(
        -max_position_abs, -min_position_abs, n_negative
    )

    # Combine and shuffle
    target_positions = np.concatenate([positive_positions, negative_positions])
    np.random.shuffle(target_positions)

    samples = []

    for i, target_pos in enumerate(target_positions):
        # Calculate center position (where + goes)
        center_pos = line_length // 2

        # Calculate where * should go based on target position
        # Map from [-1, 1] to [0, line_length-1]
        star_pos = int(round((target_pos + 1) / 2 * (line_length - 1)))

        # Generate the three lines
        reference_line = "-" * line_length

        fixation_line = list("-" * line_length)
        fixation_line[center_pos] = "+"
        fixation_line = "".join(fixation_line)

        marker_line = list("-" * line_length)
        marker_line[center_pos] = "+"
        marker_line[star_pos] = "*"
        marker_line = "".join(marker_line)

        samples.append(
            {
                "sample_id": f"sample_{i+1:03d}",
                "star_position": star_pos,
                "target_position": round(np.abs(target_pos), 3),
                "reference_line": reference_line,
                "fixation_line": fixation_line,
                "marker_line": marker_line,
            }
        )

    # Create DataFrame
    samples_df = pd.DataFrame(samples)

    # Save to CSV with informative filename
    os.makedirs(output_dir, exist_ok=True)
    filename = (
        f"line_position_{min_position_abs}_{max_position_abs}_{n_samples}samples.csv"
    )
    filepath = os.path.join(output_dir, filename)
    samples_df.to_csv(filepath, index=False)

    print(f"Generated {len(samples)} line position samples")
    print(
        f"Position ranges: [{min_position_abs}, {max_position_abs}] and [{-max_position_abs}, {-min_position_abs}]"
    )
    print(f"Excluded range: [{-min_position_abs}, {min_position_abs}]")
    print(f"Line length: {line_length} characters")
    print(f"Positive samples: {n_positive}, Negative samples: {n_negative}")
    print(f"Saved to: {filepath}")

    return samples_df

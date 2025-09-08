import os
import numpy as np
import pandas as pd


def generate_a_string_samples_with_end_char(
    min_count: int,
    max_count: int,
    n_samples: int,
    end_char: str = "C",
    output_dir: str = "../experiments/distance_estimation/data/experiment_files",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate samples with customizable end character

    Args:
        min_count: Minimum number of A's
        max_count: Maximum number of A's
        n_samples: Number of samples
        end_char: Character to append at the end (default: 'C')
        output_dir: Directory to save CSV file
        seed: Random seed
    """
    np.random.seed(seed)
    target_counts = np.random.randint(min_count, max_count + 1, n_samples)

    samples = []
    for i, target_count in enumerate(target_counts):
        text = "A" * target_count + end_char

        samples.append(
            {
                "sample_id": f"sample_{i+1:03d}",
                "target_count": target_count,
                "text": text,
                "actual_a_count": target_count,
                "end_char": end_char,
                "total_length": len(text),
            }
        )

    samples_df = pd.DataFrame(samples)

    # Save with end_char in filename
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{min_count}_{max_count}_{n_samples}samples_end{end_char}.csv"
    filepath = os.path.join(output_dir, filename)
    samples_df.to_csv(filepath, index=False)

    print(f"Generated {len(samples)} samples with format: A's + '{end_char}'")
    print(f"Saved to: {filepath}")

    return samples_df

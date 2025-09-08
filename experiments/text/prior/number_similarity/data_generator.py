import os
import numpy as np
import pandas as pd
from itertools import combinations


def generate_number_similarity_samples(
    min_value: int = 1,
    max_value: int = 10,
    n_samples: int = None,
    output_dir: str = "../experiments/number_similarity/data/experiment_files",
    seed: int = 42,
    sample_method: str = "all",  # "all", "random", "stratified"
) -> pd.DataFrame:
    """
    Generate samples of number similarity comparison tasks.
    Creates pairs of integers within a specified range for similarity estimation.

    Args:
        min_value: Minimum integer value in the range (inclusive)
        max_value: Maximum integer value in the range (inclusive)
        n_samples: Number of samples to generate (None = generate all possible pairs)
        output_dir: Directory to save CSV file
        seed: Random seed for reproducibility
        sample_method: How to sample pairs:
            - "all": Generate all possible pairs (ignores n_samples)
            - "random": Randomly sample n_samples pairs
            - "stratified": Sample evenly across similarity levels

    Returns:
        pd.DataFrame: Generated samples with columns:
            - sample_id: Unique identifier for each sample
            - number_1: First number in the pair
            - number_2: Second number in the pair
            - absolute_difference: |number_1 - number_2|
            - relative_difference: abs_diff / max(number_1, number_2)
            - similarity_category: "identical", "very_similar", "similar", "different"
            - range_min: Minimum value of the range
            - range_max: Maximum value of the range
            - range_size: Size of the range (max - min + 1)

    Example:
        # Generate all pairs from 1 to 5
        samples = generate_number_similarity_samples(1, 5)
        # Creates pairs: (1,1), (1,2), (1,3), ..., (5,5)

        # Generate 50 random pairs from 1 to 20
        samples = generate_number_similarity_samples(1, 20, n_samples=50, sample_method="random")
    """

    np.random.seed(seed)

    # Generate all possible pairs (including identical pairs and both orders)
    numbers = list(range(min_value, max_value + 1))

    if sample_method == "all":
        # Generate all possible pairs (including self-pairs and both orders)
        all_pairs = []
        for num1 in numbers:
            for num2 in numbers:
                all_pairs.append((num1, num2))
        pairs = all_pairs

    elif sample_method == "random":
        # Generate all possible pairs first, then randomly sample
        all_pairs = []
        for num1 in numbers:
            for num2 in numbers:
                all_pairs.append((num1, num2))

        if n_samples is None or n_samples >= len(all_pairs):
            pairs = all_pairs
        else:
            pairs = list(
                np.random.choice(len(all_pairs), size=n_samples, replace=False)
            )
            pairs = [all_pairs[i] for i in pairs]

    elif sample_method == "stratified":
        # Sample evenly across different similarity levels
        all_pairs = []
        for num1 in numbers:
            for num2 in numbers:
                all_pairs.append((num1, num2))

        # Categorize pairs by similarity
        pair_categories = {
            "identical": [],
            "very_similar": [],
            "similar": [],
            "different": [],
        }
        range_size = max_value - min_value + 1

        for pair in all_pairs:
            num1, num2 = pair
            abs_diff = abs(num1 - num2)

            if abs_diff == 0:
                pair_categories["identical"].append(pair)
            elif abs_diff <= range_size * 0.1:  # Within 10% of range
                pair_categories["very_similar"].append(pair)
            elif abs_diff <= range_size * 0.3:  # Within 30% of range
                pair_categories["similar"].append(pair)
            else:
                pair_categories["different"].append(pair)

        # Sample evenly from each category
        if n_samples is None:
            pairs = all_pairs
        else:
            samples_per_category = n_samples // 4
            pairs = []

            for category, category_pairs in pair_categories.items():
                if len(category_pairs) > 0:
                    n_from_category = min(samples_per_category, len(category_pairs))
                    sampled = list(
                        np.random.choice(
                            len(category_pairs), size=n_from_category, replace=False
                        )
                    )
                    pairs.extend([category_pairs[i] for i in sampled])

    # Create samples from pairs
    samples = []
    range_size = max_value - min_value + 1

    for i, (num1, num2) in enumerate(pairs):
        abs_diff = abs(num1 - num2)
        rel_diff = abs_diff / max(num1, num2) if max(num1, num2) > 0 else 0

        # Categorize similarity
        if abs_diff == 0:
            similarity_category = "identical"
        elif abs_diff <= range_size * 0.1:
            similarity_category = "very_similar"
        elif abs_diff <= range_size * 0.3:
            similarity_category = "similar"
        else:
            similarity_category = "different"

        samples.append(
            {
                "sample_id": f"sample_{i+1:04d}",
                "number_1": num1,
                "number_2": num2,
                "absolute_difference": abs_diff,
                "relative_difference": round(rel_diff, 4),
                "similarity_category": similarity_category,
                "range_min": min_value,
                "range_max": max_value,
                "range_size": range_size,
            }
        )

    # Create DataFrame
    samples_df = pd.DataFrame(samples)

    # Save to CSV with informative filename
    os.makedirs(output_dir, exist_ok=True)
    method_suffix = f"_{sample_method}" if sample_method != "all" else ""
    n_suffix = f"_{len(samples)}samples" if n_samples else f"_{len(samples)}samples"
    filename = f"number_similarity_{min_value}_{max_value}{method_suffix}{n_suffix}.csv"
    filepath = os.path.join(output_dir, filename)
    samples_df.to_csv(filepath, index=False)

    print(f"Generated {len(samples)} number similarity pairs")
    print(f"Range: {min_value} to {max_value}")
    print(f"Method: {sample_method}")
    print(f"Similarity distribution:")
    print(samples_df["similarity_category"].value_counts())
    print(f"Saved to: {filepath}")

    return samples_df


if __name__ == "__main__":
    # Example usage

    # Generate all pairs for small range
    df1 = generate_number_similarity_samples(1, 5, sample_method="all")

    # Generate random sample for larger range
    df2 = generate_number_similarity_samples(
        1, 20, n_samples=100, sample_method="random"
    )

    # Generate stratified sample
    df3 = generate_number_similarity_samples(
        1, 15, n_samples=80, sample_method="stratified"
    )

import os
import numpy as np
import pandas as pd

import pandas as pd
from pathlib import Path
import random

file_path = Path(__file__)  # or Path("your/file/path.py")
parent_path = file_path.parent

# Rough average weights (kg) for ~50 animals across a wide range
animal_weights = {
    "Mouse": 0.025,
    "Hamster": 0.1,
    "Rat": 0.35,
    "Falcon": 1.2,
    "Penguin (Little Blue)": 1.2,
    "Rabbit": 2.0,
    "Chicken": 2.5,
    "Duck": 3.0,
    "Cat": 4.5,
    "Goose": 5.0,
    "Sloth": 6.0,
    "Eagle": 6.5,
    "Turkey": 7.0,
    "Small Dog": 8.0,
    "Koala": 12.0,
    "Medium Dog": 20.0,
    "Penguin (Emperor)": 30.0,
    "Large Dog": 35.0,
    "Wolf": 45.0,
    "Chimpanzee": 50.0,
    "Goat": 60.0,
    "Leopard": 60.0,
    "Sheep": 70.0,
    "Human": 70.0,
    "Cheetah": 72.0,
    "Kangaroo": 85.0,
    "Jaguar": 95.0,
    "Panda": 100.0,
    "Deer": 120.0,
    "Ostrich": 120.0,
    "Pig": 150.0,
    "Seal": 150.0,
    "Gorilla": 160.0,
    "Lion": 190.0,
    "Dolphin": 200.0,
    "Tiger": 220.0,
    "Donkey": 250.0,
    "Sea Lion": 300.0,
    "Grizzly Bear": 360.0,
    "Moose": 400.0,
    "Polar Bear": 450.0,
    "Horse": 500.0,
    "Camel": 600.0,
    "Cow": 700.0,
    "Bison": 900.0,
    "Giraffe": 1200.0,
    "Whale (Beluga)": 1400.0,
    "Hippopotamus": 1500.0,
    "Rhinoceros": 2300.0,
    "Elephant (Asian)": 4000.0,
    "Whale (Orca)": 5600.0,
    "Elephant (African)": 6000.0,
    "Whale (Humpback)": 30000.0,
    "Whale (Blue)": 120000.0,
}


# Convert to DataFrame
df_animals = pd.DataFrame(
    list(animal_weights.items()), columns=["Animal", "AverageWeight_kg"]
)
df_animals["normalized_weight"] = np.log(df_animals["AverageWeight_kg"]) / np.log(
    df_animals["AverageWeight_kg"].max()
)


def generate_animal_sequence(
    min_weight: float = 0.2,
    max_weight: float = 1.0,
    n_samples: int = 100,
    line_length: int = 13,
    output_dir: str = "../experiments/location_on_line/data/experiment_files",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a sequence of animals whose normalized weights are between min_weight and max_weight.
    """
    np.random.seed(seed)

    # Filter animals by normalized weight
    eligible_animals = df_animals[
        (df_animals["normalized_weight"] >= min_weight)
        & (df_animals["normalized_weight"] <= max_weight)
    ].reset_index(drop=True)

    if len(eligible_animals) == 0:
        raise ValueError("No animals found in the specified normalized weight range.")

    # Sample with replacement if not enough animals
    sampled_animals = eligible_animals.sample(
        n=n_samples, replace=(n_samples > len(eligible_animals)), random_state=seed
    ).reset_index(drop=True)

    samples = []
    for i, row in sampled_animals.iterrows():
        samples.append(
            {
                "sample_id": f"sample_{i+1:03d}",
                "animal_name": row["Animal"],
                "average_weight_kg": row["AverageWeight_kg"],
                "normalized_weight": row["normalized_weight"],
            }
        )

    # Create DataFrame
    samples_df = pd.DataFrame(samples)

    # Save to CSV with informative filename
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{min_weight}_{max_weight}_{n_samples}samples.csv"
    filepath = os.path.join(output_dir, filename)
    samples_df.to_csv(filepath, index=False)

    print(f"Generated {len(samples)} animal weight samples")
    print(f"Normalized weight range: [{min_weight}, {max_weight}]")
    print(f"Saved to: {filepath}")
    return samples_df


if __name__ == "__main__":
    weight_ranges = [(0, 0.5), (0.3, 0.8), (0.5, 1.0)]
    # Example usage
    for min_weight, max_weight in weight_ranges:
        print(f"Generating samples for range: {min_weight} - {max_weight}")
        generate_animal_sequence(
            min_weight=min_weight,
            max_weight=max_weight,
            n_samples=100,
            line_length=13,
            output_dir=f"{parent_path}/data/experiment_files",
            seed=42,
        )

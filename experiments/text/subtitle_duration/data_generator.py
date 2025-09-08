import re
import os
import glob
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


# ---- AMI Corpus Parser and Preparation ----
def parse_single_file(file_path):
    import xml.etree.ElementTree as ET
    import os

    tree = ET.parse(file_path)
    root = tree.getroot()

    word_data = []
    for word_element in root.findall(".//w"):  # ← NO namespace
        start = word_element.attrib.get("starttime")
        end = word_element.attrib.get("endtime")
        text = word_element.text if word_element.text is not None else ""

        word_data.append(
            {
                "start": float(start),
                "end": float(end),
                "text": text,
                "file": os.path.basename(file_path),
            }
        )
    df = pd.DataFrame(word_data)
    return df


def combine_session_word_files(
    input_folder_path,
    session_prefix,
    save_to_csv=True,
    filename=None,
    output_folder_path="../experiments/time_estimation/data/AMI_Corpus/processed_files",
):
    """
    Combines all word XML files in a folder that start with a given session prefix (e.g. 'ES2008a.')
    """

    all_files = glob.glob(
        os.path.join(input_folder_path, f"{session_prefix}*.words.xml")
    )

    if not all_files:
        print("files not found")
        return pd.DataFrame()

    df_list = [parse_single_file(file) for file in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    if save_to_csv:
        if filename is None:
            filename = f"{session_prefix}_combined_words.csv"

        combined_df.to_csv(os.path.join(output_folder_path, filename), index=False)
    return combined_df.sort_values("start").reset_index(drop=True)


def combine_and_reindex_ami_files(
    processed_files_dir="../experiments/time_estimation/data/AMI_Corpus/processed_files",
    output_filename="combined_ami_corpus.csv",
    file_pattern="*.csv",
):
    """
    Combine multiple AMI CSV files, re-indexing start times to be continuous.

    Args:
        processed_files_dir: Directory containing processed CSV files
        output_filename: Name for the combined output file
        file_pattern: Pattern to match CSV files (e.g., "ES2008*.csv")

    Returns:
        pd.DataFrame: Combined and re-indexed DataFrame

    Example:
    # Usage example
        combined_ami = combine_and_reindex_ami_files()

        # Or combine specific files
        combined_ami = combine_and_reindex_ami_files(
            file_pattern="ES2008*.csv",  # Only ES2008 sessions
            output_filename="es2008_combined.csv"
        )
    """

    # Find all CSV files
    csv_files = glob.glob(os.path.join(processed_files_dir, file_pattern))

    if not csv_files:
        print(f"No CSV files found in {processed_files_dir}")
        return pd.DataFrame()

    print(f"Found {len(csv_files)} CSV files to combine")

    combined_data = []
    cumulative_offset = 0

    for i, file_path in enumerate(sorted(csv_files)):
        print(f"Processing {os.path.basename(file_path)}...")

        # Load the CSV
        df = pd.read_csv(file_path)

        # Calculate duration for each word
        df["duration"] = df["end"] - df["start"]

        # Re-index start times
        df["start"] = df["start"] - df["start"].min() + cumulative_offset
        df["end"] = df["start"] + df["duration"]

        # Add session identifier
        session_name = (
            os.path.basename(file_path)
            .replace("_combined_words.csv", "")
            .replace(".csv", "")
        )
        df["session"] = session_name
        df["session_order"] = i

        combined_data.append(df)

        # Update offset for next session (add small gap between sessions)
        cumulative_offset = df["end"].max() + 1.0  # 1 second gap between sessions

        print(f"  Session duration: {df['end'].max() - df['start'].min():.1f}s")

    # Combine all sessions
    combined_df = pd.concat(combined_data, ignore_index=True)

    # Sort by start time to ensure chronological order
    combined_df = combined_df.sort_values("start").reset_index(drop=True)

    # Clean up columns
    combined_df = combined_df.drop(
        "duration", axis=1
    )  # Remove temporary duration column

    # Save combined file
    output_path = os.path.join(processed_files_dir, output_filename)
    combined_df.to_csv(output_path, index=False)

    print(f"\n✅ Combined {len(csv_files)} sessions into {output_filename}")
    print(f"📊 Total words: {len(combined_df)}")
    print(f"⏱️  Total duration: {combined_df['end'].max():.1f} seconds")
    print(f"📚 Sessions: {list(combined_df['session'].unique())}")

    return combined_df


def sample_text_segments(df, duration=3.0, n=5, seed=None):
    if seed is not None:
        random.seed(seed)

    segments = []

    # Ensure the DataFrame is sorted by start time
    df = df.sort_values("start").reset_index(drop=True)

    # Get the maximum start point to allow room for a full segment
    max_start = df["start"].max() - duration

    for _ in range(n):
        # Randomly choose a valid start time
        t_start = random.uniform(df["start"].min(), max_start)
        t_end = t_start + duration

        # Get all words that fall within this time window
        mask = (df["start"] >= t_start) & (df["end"] <= t_end)
        words = df.loc[mask, "text"].tolist()
        span = " ".join(words)

        segments.append(
            {"start": round(t_start, 2), "end": round(t_end, 2), "text": span}
        )

    return pd.DataFrame(segments)


def generate_duration_samples(
    ami_df: pd.DataFrame,
    min_duration: float,
    max_duration: float,
    n_samples: int,
    output_dir: str = "../experiments/time_estimation/data/AMI_Corpus/experiment_files",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simple uniform sampling of text segments within duration range.

    Args:
        ami_df: Master AMI corpus DataFrame with 'start', 'end', 'text' columns
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        n_samples: Number of samples to generate
        output_dir: Directory to save CSV file
        seed: Random seed for reproducibility

    Returns:
        pd.DataFrame: Generated samples

    Example:
        samples = generate_duration_samples(ami_df, 5.0, 15.0, 20)
        # Saves to: ../data/experiment_samples/5.0_15.0_20samples.csv
    """

    np.random.seed(seed)

    # Uniformly sample target durations
    target_durations = np.random.uniform(min_duration, max_duration, n_samples)

    samples = []

    for i, target_duration in enumerate(target_durations):
        # Find a segment that matches target duration (±10% tolerance)
        tolerance = target_duration * 0.1

        # Try random starting points until we find a good match
        max_attempts = 100
        for attempt in range(max_attempts):
            # Random start point
            latest_possible_start = ami_df["start"].max() - max_duration
            random_start = np.random.uniform(
                ami_df["start"].min(), latest_possible_start
            )

            # Find words in the target window
            segment_end = random_start + target_duration
            mask = (ami_df["start"] >= random_start) & (ami_df["end"] <= segment_end)
            segment_words = ami_df[mask]

            if len(segment_words) > 0:
                # Calculate actual duration and text
                actual_start = segment_words["start"].min()
                actual_end = segment_words["end"].max()
                actual_duration = actual_end - actual_start
                text = " ".join(segment_words["text"].tolist())

                # Check if duration is within tolerance and has enough words
                if (
                    abs(actual_duration - target_duration) <= tolerance
                    and len(text.split()) >= 5
                ):

                    samples.append(
                        {
                            "sample_id": f"sample_{i+1:03d}",
                            "target_duration": round(target_duration, 2),
                            "actual_duration": round(actual_duration, 2),
                            "start_time": round(actual_start, 2),
                            "end_time": round(actual_end, 2),
                            "word_count": len(text.split()),
                            "text": text,
                        }
                    )
                    break
        else:
            print(f"Warning: Could not find suitable segment for sample {i+1}")

    # Create DataFrame
    samples_df = pd.DataFrame(samples)

    # Save to CSV with informative filename
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{min_duration}_{max_duration}_{n_samples}samples.csv"
    filepath = os.path.join(output_dir, filename)
    samples_df.to_csv(filepath, index=False)

    print(f"Generated {len(samples)} samples")
    print(f"Duration range: {min_duration}-{max_duration}s")
    print(f"Saved to: {filepath}")

    return samples_df

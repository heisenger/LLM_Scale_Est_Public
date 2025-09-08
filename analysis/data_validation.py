import pandas as pd
import numpy as np


def validate_data(data, min_samples=5):

    # Remove NaN/infinite values
    clean_data = data.dropna(subset=["correct", "response"])
    clean_data = clean_data[
        np.isfinite(clean_data["correct"]) & np.isfinite(clean_data["response"])
    ]

    if len(clean_data) < min_samples:
        return None, "Too many invalid values"

    # Check for variance
    if clean_data["correct"].std() < 1e-10 or clean_data["response"].std() < 1e-10:
        return None, "No variance in data"

    return clean_data, None

import geopandas as gpd  # pip install geopandas
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Dict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


# Generate a CSV file with the latitude and longitude of world capitals
# source  https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-populated-places/?utm_source=chatgpt.com
def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000  # Earth radius in metres
    φ1, φ2 = np.radians([lat1, lat2])
    dφ = np.radians(lat2 - lat1)
    dλ = np.radians(lon2 - lon1)
    a = np.sin(dφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(dλ / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a)) / 1000  # distance in km


def generate_world_capitals_latlon():
    """
    Generate a CSV file with the latitude and longitude of world capitals.
    The capitals are extracted from the Natural Earth populated places dataset.
    """

    # 1)  Download the Natural Earth populated places dataset
    # https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-populated-places/
    # (or use the provided file in the data/raw_files directory)

    # 2)  Read the shapefile using geopandas
    # point to ANY one of the component files (the .shp is enough)
    gdf = gpd.read_file(
        BASE_DIR / "data/raw_files/ne_10m_populated_places/ne_10m_populated_places.shp"
    )

    # 3)  Keep only national (admin‑0) capitals
    capitals = (
        gdf[gdf["ADM0CAP"] == 1]  # flag == national capital
        .loc[:, ["NAME", "ADM0NAME", "LATITUDE", "LONGITUDE"]]
        .rename(
            columns={
                "NAME": "city",
                "ADM0NAME": "country",
                "LATITUDE": "lat",
                "LONGITUDE": "lon",
            }
        )
        .reset_index(drop=True)
    )

    # 4)  Save for later use
    capitals.to_csv(
        BASE_DIR / "data/raw_files/world_capitals_latlon.csv",
        index=False,
    )

    print(f"Saved {len(capitals)} capitals to world_capitals_latlon.csv")
    print(capitals.head())


def load_world_capitals():
    """Load world capitals from CSV file."""
    try:
        capitals = pd.read_csv(BASE_DIR / "data/raw_files/world_capitals_latlon.csv")
        return capitals
    except FileNotFoundError:
        print("world_capitals_latlon.csv not found. Generating it first...")
        generate_world_capitals_latlon()
        return pd.read_csv(BASE_DIR / "data/raw_files/world_capitals_latlon.csv")


def find_capitals_in_distance_range(
    reference_capital: str,
    min_distance: float,
    max_distance: float,
    capitals_df: pd.DataFrame = None,
) -> List[Tuple[str, str, float]]:
    """
    Find all capitals within a specific distance range from a reference capital.

    Args:
        reference_capital: Name of the reference capital city
        min_distance: Minimum distance in km
        max_distance: Maximum distance in km
        capitals_df: DataFrame with capitals data (optional)

    Returns:
        List of tuples: (city_name, country_name, distance_km)
    """
    if capitals_df is None:
        capitals_df = load_world_capitals()

    # Find reference capital
    ref_row = capitals_df[
        capitals_df["city"].str.contains(reference_capital, case=False, na=False)
    ]
    if ref_row.empty:
        raise ValueError(f"Capital '{reference_capital}' not found in dataset")

    ref_lat, ref_lon = ref_row.iloc[0]["lat"], ref_row.iloc[0]["lon"]

    # Calculate distances to all other capitals
    distances = []
    for _, row in capitals_df.iterrows():
        if row["city"].lower() != reference_capital.lower():
            distance = haversine(ref_lat, ref_lon, row["lat"], row["lon"])
            if min_distance <= distance <= max_distance:
                distances.append((row["city"], row["country"], distance))

    # Sort by distance
    distances.sort(key=lambda x: x[2])
    return distances


def generate_multihop_city_routes(
    n_samples: int = 10,
    hop_count: int = 3,
    min_total_distance: float = 3000.0,
    max_total_distance: float = 10000.0,
    capitals_df: pd.DataFrame = None,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate multi-hop capital city routes that fall within a total distance range.

    Args:
        n_samples: Number of routes to generate
        hop_count: Number of cities in each route (e.g. 3 = two hops)
        min_total_distance: Minimum total distance in km
        max_total_distance: Maximum total distance in km
        capitals_df: Optional preloaded DataFrame of capitals
        seed: Random seed

    Returns:
        List of dicts with keys:
            - 'city_chain': list of cities
            - 'country_chain': list of countries
            - 'total_distance': total route distance (km)
    """
    import random

    if capitals_df is None:
        capitals_df = load_world_capitals()

    rng = np.random.default_rng(seed)
    results = []
    attempts = 0
    max_attempts = n_samples * 100

    while len(results) < n_samples and attempts < max_attempts:
        attempts += 1

        # Randomly sample one possible route of given length
        route_indices = rng.choice(capitals_df.index, size=hop_count, replace=False)
        route = capitals_df.loc[route_indices].reset_index(drop=True)

        # Compute total distance
        total_distance = 0.0
        valid = True
        for i in range(hop_count - 1):
            lat1, lon1 = route.loc[i, ["lat", "lon"]]
            lat2, lon2 = route.loc[i + 1, ["lat", "lon"]]
            d = haversine(lat1, lon1, lat2, lon2)
            if d == 0 or d > max_total_distance:  # filter degenerate or huge hops
                valid = False
                break
            total_distance += d

        if valid and min_total_distance <= total_distance <= max_total_distance:
            results.append(
                {
                    "city_chain": route["city"].tolist(),
                    "country_chain": route["country"].tolist(),
                    "total_distance": round(total_distance, 2),
                }
            )

    if len(results) < n_samples:
        print(f"Warning: only generated {len(results)} routes out of {n_samples}")

    return results


def generate_city_distance_samples(
    min_distance: float = 1000.0,
    max_distance: float = 5000.0,
    multi_hop: int = 0,  # >0 enables multi-hop routes with this many cities
    n_samples: int = 100,
    output_dir: str = BASE_DIR / "data/experiment_files",
    seed: int = 42,
    capitals_csv: str = BASE_DIR / "data/raw_files/world_capitals_latlon.csv",
) -> pd.DataFrame:
    """
    Generate samples of city distance estimation tasks.
    - Single-hop: Random city pairs within a distance range.
    - Multi-hop: Random routes with N cities where total distance falls within range.

    Args:
        min_distance: Minimum distance (or total distance for multi-hop) in km
        max_distance: Maximum distance (or total distance for multi-hop) in km
        multi_hop: Number of cities in route (>0 means multi-hop mode)
        n_samples: Number of samples to generate
        output_dir: Directory to save CSV file
        seed: Random seed
        capitals_csv: Path to capitals CSV file

    Returns:
        pd.DataFrame with:
            - sample_id
            - For single-hop: reference_city, target_city, actual_distance, etc.
            - For multi-hop: city_chain, country_chain, total_distance
    """
    np.random.seed(seed)
    capitals_df = pd.read_csv(capitals_csv)

    samples = []

    if multi_hop > 0:
        # Multi-hop mode
        routes = generate_multihop_city_routes(
            n_samples=n_samples,
            hop_count=multi_hop,
            min_total_distance=min_distance,
            max_total_distance=max_distance,
            capitals_df=capitals_df,
            seed=seed,
        )
        for i, route in enumerate(routes, 1):
            samples.append(
                {
                    "sample_id": f"route_{i:03d}",
                    "city_chain": ", ".join(route["city_chain"]),
                    "country_chain": ", ".join(route["country_chain"]),
                    "total_distance": route["total_distance"],
                    "distance_range": f"{min_distance}-{max_distance}km",
                    "min_distance": min_distance,
                    "max_distance": max_distance,
                    "hop_count": multi_hop,
                }
            )

        samples_df = pd.DataFrame(samples)
        filename = f"city_multihop_{multi_hop}hops_{min_distance}_{max_distance}_{len(samples)}samples.csv"

    else:
        # Single-hop (original pairwise logic)
        attempts = 0
        max_attempts = n_samples * 10
        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            ref_idx = np.random.randint(0, len(capitals_df))
            ref_capital = capitals_df.iloc[ref_idx]

            try:
                candidates = find_capitals_in_distance_range(
                    ref_capital["city"], min_distance, max_distance, capitals_df
                )
                if candidates:
                    target_city, target_country, distance = candidates[
                        np.random.randint(0, len(candidates))
                    ]
                    samples.append(
                        {
                            "sample_id": f"sample_{len(samples)+1:03d}",
                            "reference_city": ref_capital["city"],
                            "reference_country": ref_capital["country"],
                            "reference_lat": ref_capital["lat"],
                            "reference_lon": ref_capital["lon"],
                            "target_city": target_city,
                            "target_country": target_country,
                            "actual_distance": round(distance, 2),
                            "distance_range": f"{min_distance}-{max_distance}km",
                            "min_distance": min_distance,
                            "max_distance": max_distance,
                        }
                    )
            except ValueError:
                continue

        samples_df = pd.DataFrame(samples)
        filename = (
            f"city_distances_{min_distance}_{max_distance}_{len(samples)}samples.csv"
        )

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    samples_df.to_csv(filepath, index=False)

    print(
        f"Generated {len(samples)} {'multi-hop' if multi_hop > 0 else 'city pair'} samples"
    )
    print(f"Distance range: {min_distance} - {max_distance} km")
    avg_col = (
        "total_distance"
        if "total_distance" in samples_df.columns
        else "actual_distance"
    )
    print(f"Average distance: {samples_df[avg_col].mean():.1f} km")
    print(f"Distance std: {samples_df[avg_col].std():.1f} km")
    print(f"Saved to: {filepath}")

    return samples_df


if __name__ == "__main__":

    distance_range = [(1000.0, 5000.0), (3000.0, 8000.0), (5000.0, 10000.0)]
    distance_range_multi = [(10000.0, 50000.0), (30000.0, 80000.0), (50000.0, 100000.0)]
    nsamples = 100
    multi_hop = 8  # Set to 0 for single-hop

    for min_distance, max_distance in distance_range_multi:
        generate_city_distance_samples(
            min_distance=min_distance,
            max_distance=max_distance,
            multi_hop=multi_hop,  # Set to 0 for single-hop
            n_samples=nsamples,
            output_dir=BASE_DIR / f"data/experiment_files",
            seed=42,
        )

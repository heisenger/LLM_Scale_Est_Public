import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random
import math
import ast
from typing import List, Tuple
from PIL import Image, ImageFilter, ImageDraw
from pdf2image import convert_from_path
import cairosvg
import io

file_path = Path(__file__)  # or Path("your/file/path.py")
parent_path = file_path.parent


def open_image_any_format(img_path):
    img_path = str(img_path)
    if img_path.lower().endswith(".pdf"):
        # Convert first page of PDF to PIL Image
        pil_imgs = convert_from_path(img_path, first_page=1, last_page=1)
        img = pil_imgs[0].convert("RGBA")
    elif img_path.lower().endswith(".png"):
        img = Image.open(img_path).convert("RGBA")
        png_bytes = cairosvg.svg2png(url="myplot.svg")
        img = Image.open(io.BytesIO(png_bytes))
    else:
        img = Image.open(img_path).convert("RGBA")
    return img


def get_experiment_files_root(df_path):
    df_path = Path(df_path)
    parts = df_path.parts
    if "experiment_files" in parts:
        idx = parts.index("experiment_files")
        return Path(*parts[: idx + 1])
    else:
        raise ValueError("experiment_files not found in path")


def gaussian_blur_img(df_path, radius=5, suffix="blurred", save_format="png"):
    """
    Apply a selective Gaussian blur to an image and save the result in a 'blurred' subfolder.

    Args:
        img_path (str or Path): Path to the input image.
        radius (int): Blur radius.
        suffix (str): Suffix for the output filename.
    """

    stimulus_df = pd.read_csv(df_path)
    folder_path = get_experiment_files_root(df_path)
    for i in stimulus_df.index:

        img_path = stimulus_df.loc[i, "image_path"]

        coord_end = ast.literal_eval(stimulus_df.maze_path[i])[-1]
        x_end, y_end = coord_end[0], coord_end[1]

        img_path = Path(folder_path / img_path)
        # import pdb

        # pdb.set_trace()
        img = open_image_any_format(img_path)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))

        margin_est = 0.75
        x_end = (x_end + 1 + margin_est) / (margin_est + 1 + 15) * img.width
        y_end = (
            img.height - (y_end + 1 + margin_est) / (margin_est + 1 + 15) * img.height
        )

        # Create a mask: white (255) where you want blur, black (0) elsewhere
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        center = (
            x_end,
            y_end,
        )  # x, y coordinates of blur center
        blur_radius = 40  # radius of blur region
        draw.ellipse(
            [
                center[0] - blur_radius,
                center[1] - blur_radius,
                center[0] + blur_radius,
                center[1] + blur_radius,
            ],
            fill=255,
        )

        # Composite: blur only where mask is white
        result = Image.composite(blurred, img, mask)
        if radius != 2:
            output_dir = img_path.parent / f"blurred_{radius}"
        else:
            output_dir = img_path.parent / "blurred"
        output_dir.mkdir(
            parents=True, exist_ok=True
        )  # Create the folder if it doesn't exist

        result.save(output_dir / f"{img_path.stem}_{suffix}.{save_format}")


def generate_spiral_path(
    grid_size=15,
    target_euclidean=10,
    step_size_factor=0.25,
    size_adj=1,
    save_path: str = "line_dot.png",
):
    """
    Drop-in replacement: generate a self-avoiding, axis-aligned path whose
    prefix is as close as possible (in Euclidean distance) to `target_euclidean`.
    Signature, return values, and plotting match the original.

    Returns:
        best_path (list[(x,y) floats]): path vertices in the same coordinate system
                                        (multiples of step_size_factor).
        best_dist (float): Euclidean distance from start to end of best_path.
    """
    import math, random

    # Lattice scale: operate on integer lattice, then rescale by step_size_factor
    scale = 1.0 / step_size_factor
    W = int(grid_size * scale)
    cx = int((grid_size // 2) * scale)
    cy = int((grid_size // 2) * scale)
    start_lat = (cx, cy)

    # Target distance in lattice units
    target_lat = target_euclidean * scale

    # Heuristics: how far we let each attempt explore, and how many attempts
    MAX_STEPS = min(int(target_lat * 3) + 20, int(0.9 * W))
    ATTEMPTS = 200
    CLOSE_TOL = 0.5 * step_size_factor  # "good enough" tolerance in original units

    best_diff = float("inf")
    best_path_lat = [start_lat]

    def to_float_path(lat_path):
        return [(x * step_size_factor, y * step_size_factor) for x, y in lat_path]

    for _ in range(ATTEMPTS):
        path = [start_lat]
        used = {start_lat}

        for _step in range(MAX_STEPS):
            x, y = path[-1]
            candidates = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                dx *= size_adj
                dy *= size_adj
                nx, ny = x + dx, y + dy
                # stay in bounds and avoid revisiting vertices (prevents self-crossing)
                if 0 <= nx < W and 0 <= ny < W and (nx, ny) not in used:
                    # bias toward moves whose *resulting* Euclidean distance
                    # is closer to the target (in original units)
                    rx = nx - start_lat[0]
                    ry = ny - start_lat[1]
                    r_next = step_size_factor * math.hypot(rx, ry)
                    score = -abs(r_next - target_euclidean)
                    candidates.append(((nx, ny), score))

            if not candidates:
                # dead end; stop this attempt
                break

            # Softmax sampling to avoid greedy traps (temperature ~ 0.5)
            scores = [s for _, s in candidates]
            m = max(scores)
            weights = [math.exp(2.0 * (s - m)) for s in scores]
            r = random.random() * sum(weights)
            acc = 0.0
            chosen = None
            for (p, s), w in zip(candidates, weights):
                acc += w
                if r <= acc:
                    chosen = p
                    break

            path.append(chosen)
            used.add(chosen)

            # Track best prefix so far (like your spiral version)
            rx = chosen[0] - start_lat[0]
            ry = chosen[1] - start_lat[1]
            r_here = step_size_factor * math.hypot(rx, ry)
            diff = abs(r_here - target_euclidean)
            if diff < best_diff:
                best_diff = diff
                best_path_lat = list(path)
                if best_diff <= CLOSE_TOL:
                    break  # good enough for this attempt

        if best_diff <= CLOSE_TOL:
            break  # early exit overall

    # Convert to original float coordinates and compute final distance
    best_path = to_float_path(best_path_lat)
    fx = (best_path_lat[-1][0] - start_lat[0]) * step_size_factor
    fy = (best_path_lat[-1][1] - start_lat[1]) * step_size_factor
    best_dist = math.hypot(fx, fy)

    # # --- save graphs ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1, grid_size)
    ax.set_ylim(-1, grid_size)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(True)

    xs, ys = zip(*best_path)
    ax.plot(xs, ys, "-k")
    ax.plot(xs[0], ys[0], "go", markersize=12, label="Start")
    ax.plot(xs[-1], ys[-1], "rx", markersize=12, label="End")
    ax.legend()
    # plt.show()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return best_path, best_dist


def generate_path_samples(
    min_distance: float = 1,
    max_distance: float = 5,
    n_samples: int = 100,
    output_dir: str = None,
    seed: int = 42,
    save_format: str = "png",
    output_path_append: str = "",
) -> pd.DataFrame:
    """ """
    rounding_decimals = 4
    np.random.seed(seed)

    # Uniformly sample target lengths
    path_distances = np.random.uniform(min_distance, max_distance, n_samples)

    samples = []
    seen_paths = set()
    os.makedirs(output_dir, exist_ok=True)
    for i, path_distance in enumerate(path_distances):
        # import pdb

        # pdb.set_trace()

        # Calculate line A length based on target length
        image_path = (
            f"sample_{i:03d}_{round(path_distance, rounding_decimals)}.{save_format}"
        )

        # Generate the image
        maze_path, path_length = generate_spiral_path(
            target_euclidean=path_distance,
            save_path=os.path.join(output_dir, image_path),
        )
        # Check for duplicate path
        path_str = str(maze_path)
        if path_str in seen_paths:
            continue
        seen_paths.add(path_str)

        # Generate the meta data dataframe
        samples.append(
            {
                "sample_id": f"sample_{i+1:03d}",
                "actual_length": round(path_length, rounding_decimals),
                "maze_path": maze_path,
                "image_path": f"{min_distance}_{max_distance}/{output_path_append}{image_path}",
            }
        )

    # Create DataFrame
    samples_df = pd.DataFrame(samples)

    # Save to CSV with informative filename
    filename = f"maze_distance_{min_distance}_{max_distance}_{n_samples}samples.csv"
    filepath = os.path.join(output_dir, filename)
    samples_df.to_csv(filepath, index=False)

    print(f"Generated {len(samples)} maze path samples")
    print(f"Length range: {min_distance} - {max_distance}")
    print(f"Saved to: {filepath}")

    return samples_df


def path_to_text_directions(path, step_size_factor=0.25):
    """
    Convert a path (list of (x,y) coordinates) into a compressed textual description.
    Groups consecutive steps in the same direction.

    Example output:
    "Take 1 units east, then Take 2 units north, then Take 3 units west."
    """
    if len(path) < 2:
        return "Start only, no moves."

    # Direction mapping
    dir_name = {
        (step_size_factor, 0): "east",
        (-step_size_factor, 0): "west",
        (0, step_size_factor): "north",
        (0, -step_size_factor): "south",
    }
    instructions = []

    px, py = path[0]
    dx_prev, dy_prev = None, None
    run_len = 0

    for x, y in path[1:]:
        dx, dy = x - px, y - py
        # Safety: check that we’re moving only 1 unit orthogonally
        # if abs(dx) + abs(dy) != 1:
        #     raise ValueError(f"Non-unit step from {(px,py)} to {(x,y)}")

        if (dx, dy) == (dx_prev, dy_prev):
            run_len += step_size_factor
        else:
            if dx_prev is not None:
                instructions.append(
                    f"Take {run_len} units {dir_name[(dx_prev,dy_prev)]}"
                )
            dx_prev, dy_prev = dx, dy
            run_len = step_size_factor
        px, py = x, y

    # Append the last run
    if dx_prev is not None:
        instructions.append(f"Take {run_len} units {dir_name[(dx_prev,dy_prev)]}")

    return ", then ".join(instructions) + "."


if __name__ == "__main__":
    # Example usage
    generate_new_data = True
    generate_thesis_output = True

    if generate_thesis_output:
        output_path_append = "thesis_output/"
        n_samples = 10
        save_format = "pdf"
    else:
        output_path_append = ""
        n_samples = 100
        save_format = "png"

    # Generate new Data
    if generate_new_data:
        len_ranges = [(1.0, 5.0), (3.0, 7.0), (5.0, 9.0)]
        for min_distance, max_distance in len_ranges:
            generate_path_samples(
                min_distance=min_distance,
                max_distance=max_distance,
                n_samples=n_samples,
                output_dir=f"{parent_path}/data/experiment_files/{min_distance}_{max_distance}/{output_path_append}",
                seed=42,
                save_format=save_format,
                output_path_append=output_path_append,
            )
            sample_df = pd.read_csv(
                f"{parent_path}/data/experiment_files/{min_distance}_{max_distance}/{output_path_append}maze_distance_{min_distance}_{max_distance}_{n_samples}samples.csv"
            )

            # Generate base path text
            path_text_descriptions = []
            for i in range(len(sample_df)):
                maze_path = ast.literal_eval(sample_df.iloc[i]["maze_path"])
                text_description = path_to_text_directions(path=maze_path)
                path_text_descriptions.append(text_description)
            sample_df["path_text_description"] = path_text_descriptions
            sample_df.to_csv(
                f"{parent_path}/data/experiment_files/{min_distance}_{max_distance}/{output_path_append}maze_distance_{min_distance}_{max_distance}_{n_samples}samples.csv",
                index=False,
            )
    else:
        for file in [
            f"1.0_5.0/{output_path_append}maze_distance_1.0_5.0_{n_samples}samples.csv",
            f"3.0_7.0/{output_path_append}maze_distance_3.0_7.0_{n_samples}samples.csv",
            f"5.0_9.0/{output_path_append}maze_distance_5.0_9.0_{n_samples}samples.csv",
        ]:

            for blur_radius in [1, 2, 5]:
                gaussian_blur_img(
                    f"{parent_path}/data/experiment_files/{file}",
                    radius=blur_radius,
                    save_format=save_format,
                )
                df = pd.read_csv(f"{parent_path}/data/experiment_files/{file}")
                if blur_radius == 1:
                    df["blurred_low_image_path"] = df["image_path"].apply(
                        lambda x: f"{Path(x).parent}/blurred_{blur_radius}/{Path(x).stem}_blurred.{save_format}"
                    )
                if blur_radius == 2:
                    df["blurred_image_path"] = df["image_path"].apply(
                        lambda x: f"{Path(x).parent}/blurred/{Path(x).stem}_blurred.{save_format}"
                    )
                if blur_radius == 5:
                    df["blurred_high_image_path"] = df["image_path"].apply(
                        lambda x: f"{Path(x).parent}/blurred_{blur_radius}/{Path(x).stem}_blurred.{save_format}"
                    )
                df.to_csv(f"{parent_path}/data/experiment_files/{file}", index=False)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random
from PIL import Image, ImageFilter, ImageDraw

file_path = Path(__file__)  # or Path("your/file/path.py")
parent_path = file_path.parent


def generate_line_dot_image(
    dot_position: float,
    width: int = 400,
    height: int = 100,
    line_y: int = 50,
    save_path: str = "line_dot.png",
    dot_size: int = 10,
    blur: bool = False,
):
    """
    Generate an image of a horizontal line with a dot at a given relative position.

    Args:
        dot_position: Float between 0 and 1 representing the relative position along the line.
        width: Width of the image in pixels.
        height: Height of the image in pixels.
        line_y: Vertical position of the horizontal line.
        save_path: File path to save the image.
        dot_size: Size of the dot in pixels.
        blur: Whether to blur the dot slightly (simulate noisy visual input).
    """
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    # Draw line
    ax.plot([0, width], [line_y, line_y], color="black", linewidth=2)

    # Compute dot position
    dot_x = dot_position * width

    # Draw dot
    ax.scatter(dot_x, line_y, s=dot_size**2, color="red", zorder=5)

    # Optionally blur (simulate noise by jitter)
    if blur:
        jitter_x = np.random.normal(dot_x, width * 0.02, size=50)
        jitter_y = np.random.normal(line_y, height * 0.02, size=50)
        ax.scatter(jitter_x, jitter_y, s=(dot_size / 2) ** 2, color="red", alpha=0.2)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def generate_line_dot_image_samples(
    min_fraction: float = 10.0,
    max_fraction: float = 90.0,
    n_samples: int = 100,
    output_dir: str = None,
    seed: int = 42,
) -> pd.DataFrame:
    """ """

    np.random.seed(seed)
    rounding_decimals = 5

    # Uniformly sample target lengths
    dot_locations = np.random.uniform(min_fraction, max_fraction, n_samples)

    samples = []
    seen = set()
    os.makedirs(output_dir, exist_ok=True)
    for i, dot_location in enumerate(dot_locations):
        # Calculate line A length based on target length

        rounded_loc = round(dot_location, rounding_decimals)
        if rounded_loc in seen:
            continue
        seen.add(rounded_loc)
        image_path = f"sample_{i:03d}_{round(dot_location, rounding_decimals)}.png"

        # Generate the meta data dataframe
        samples.append(
            {
                "sample_id": f"sample_{i+1:03d}",
                "actual_length": round(dot_location, rounding_decimals),
                "image_path": f"{min_fraction}_{max_fraction}/{image_path}",
            }
        )
        # Generate the image
        generate_line_dot_image(
            dot_position=dot_location,
            width=400,
            height=100,
            line_y=50,
            save_path=os.path.join(output_dir, image_path),
            dot_size=10,
            blur=True,
        )

    # Create DataFrame
    samples_df = pd.DataFrame(samples)

    # Save to CSV with informative filename
    filename = f"marker_loc_{min_fraction}_{max_fraction}_{n_samples}samples.csv"
    filepath = os.path.join(output_dir, filename)
    samples_df.to_csv(filepath, index=False)

    print(f"Generated {len(samples)} line length samples")
    print(f"Length range: {min_fraction} - {max_fraction}")
    print(f"Saved to: {filepath}")

    return samples_df


def generate_noisy_ascii_line_dot(
    dot_position: float, line_length: int = 40, noise_level: int = 2
) -> str:
    """
    Generate a textual (ASCII) representation of a horizontal line with a dot,
    with optional noise introduced as random extra or missing dashes.

    Args:
        dot_position: Float between 0 and 1 representing the relative position along the line.
        line_length: Total number of characters for the line (excluding borders).
        noise_level: Approximate number of noisy characters to introduce.

    Returns:
        str: ASCII representation of the line with the dot and noise.
    """
    # Clamp dot_position
    dot_position = max(0.0, min(1.0, dot_position))

    # Compute index for the dot
    dot_index = int(dot_position * (line_length - 1))

    # Build base line
    line_chars = ["-"] * line_length
    line_chars[dot_index] = "O"

    # Inject noise
    for _ in range(noise_level):
        idx = random.randint(0, line_length - 1)
        if line_chars[idx] == "-":
            line_chars[idx] = random.choice([" ", "=", ".", "~"])
        elif line_chars[idx] == "O":
            # Occasionally jitter the dot
            jitter = random.choice([-1, 1])
            new_idx = min(max(0, idx + jitter), line_length - 1)
            line_chars[new_idx] = "O"
            line_chars[idx] = "-"

    ascii_line = "|" + "".join(line_chars) + "|"
    return ascii_line


def gaussian_blur_img(df_path, radius=5, suffix="blurred"):
    """
    Apply a selective Gaussian blur to an image and save the result in a 'blurred' subfolder.

    Args:
        img_path (str or Path): Path to the input image.
        radius (int): Blur radius.
        suffix (str): Suffix for the output filename.
    """

    stimulus_df = pd.read_csv(df_path)
    folder_path = Path(df_path).parent.parent
    for i in stimulus_df.index:

        img_path = stimulus_df.loc[i, "image_path"]
        dot_location = stimulus_df.loc[i, "actual_length"]

        img_path = Path(folder_path / img_path)
        img = Image.open(img_path).convert("RGBA")
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # Create a mask: white (255) where you want blur, black (0) elsewhere
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        center = (
            dot_location * img.size[0],
            0.5 * img.size[1],
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
        if suffix != 5:
            output_dir = img_path.parent / f"blurred_{radius}"
        else:
            output_dir = img_path.parent / "blurred"
        output_dir.mkdir(
            parents=True, exist_ok=True
        )  # Create the folder if it doesn't exist

        result.save(output_dir / f"{img_path.stem}_{suffix}.png")


if __name__ == "__main__":
    # Example usage
    # len_ranges = [(0.1, 0.5), (0.3, 0.8), (0.5, 0.9)]
    # for min_frac, max_frac in len_ranges:
    #     generate_line_dot_image_samples(
    #         min_fraction=min_frac,
    #         max_fraction=max_frac,
    #         n_samples=100,
    #         output_dir=f"{parent_path}/data/experiment_files/{min_frac}_{max_frac}/",
    #         seed=42,
    #     )

    #     sample_df = pd.read_csv(
    #         f"{parent_path}/data/experiment_files/{min_frac}_{max_frac}/marker_loc_{min_frac}_{max_frac}_100samples.csv"
    #     )

    #     # Generate base ASCII lines
    #     ascii_lines = []
    #     for i in range(len(sample_df)):
    #         ascii_line = generate_noisy_ascii_line_dot(
    #             dot_position=sample_df["actual_length"].iloc[i],
    #             line_length=40,
    #             noise_level=5,
    #         )
    #         ascii_lines.append(ascii_line)
    #     sample_df["ascii_line"] = ascii_lines

    #     # Generate shifted ASCII lines
    #     shifted_ascii_lines = []
    #     for i in range(len(sample_df)):
    #         ascii_line = generate_noisy_ascii_line_dot(
    #             dot_position=sample_df["actual_length"].iloc[i] + 0.1,
    #             line_length=40,
    #             noise_level=5,
    #         )
    #         shifted_ascii_lines.append(ascii_line)
    #     sample_df["shifted_ascii_line"] = shifted_ascii_lines

    #     # Generate high noise ASCII lines
    #     high_noise_ascii_lines = []
    #     for i in range(len(sample_df)):
    #         ascii_line = generate_noisy_ascii_line_dot(
    #             dot_position=sample_df["actual_length"].iloc[i],
    #             line_length=40,
    #             noise_level=10,
    #         )
    #         high_noise_ascii_lines.append(ascii_line)
    #     sample_df["high_noise_ascii_line"] = high_noise_ascii_lines

    #     sample_df.to_csv(
    #         f"{parent_path}/data/experiment_files/{min_frac}_{max_frac}/marker_loc_{min_frac}_{max_frac}_100samples.csv",
    #         index=False,
    #     )

    # Add blurred images (base blurriness)
    # for file in [
    #     "0.1_0.5/marker_loc_0.1_0.5_100samples.csv",
    #     "0.3_0.8/marker_loc_0.3_0.8_100samples.csv",
    #     "0.5_0.9/marker_loc_0.5_0.9_100samples.csv",
    # ]:
    #     # import pdb

    #     # pdb.set_trace()
    #     gaussian_blur_img(f"{parent_path}/data/experiment_files/{file}")
    #     df = pd.read_csv(f"{parent_path}/data/experiment_files/{file}")
    #     df["blurred_image_path"] = df["image_path"].apply(
    #         lambda x: f"{Path(x).parent}/blurred/{Path(x).stem}_blurred.png"
    #     )
    #     df.to_csv(f"{parent_path}/data/experiment_files/{file}", index=False)

    # Add blurred images (lower blurriness)
    for file in [
        "0.1_0.5/marker_loc_0.1_0.5_100samples.csv",
        "0.3_0.8/marker_loc_0.3_0.8_100samples.csv",
        "0.5_0.9/marker_loc_0.5_0.9_100samples.csv",
    ]:
        # import pdb

        # pdb.set_trace()
        # blur_radius = 2
        # gaussian_blur_img(
        #     f"{parent_path}/data/experiment_files/{file}", radius=blur_radius
        # )
        # df = pd.read_csv(f"{parent_path}/data/experiment_files/{file}")
        # df["blurred_low_image_path"] = df["image_path"].apply(
        #     lambda x: f"{Path(x).parent}/blurred_{blur_radius}/{Path(x).stem}_blurred.png"
        # )
        # df.to_csv(f"{parent_path}/data/experiment_files/{file}", index=False)

        blur_radius = 20
        gaussian_blur_img(
            f"{parent_path}/data/experiment_files/{file}", radius=blur_radius
        )
        df = pd.read_csv(f"{parent_path}/data/experiment_files/{file}")
        df["blurred_high_image_path"] = df["image_path"].apply(
            lambda x: f"{Path(x).parent}/blurred_{blur_radius}/{Path(x).stem}_blurred.png"
        )
        df.to_csv(f"{parent_path}/data/experiment_files/{file}", index=False)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random
from PIL import Image, ImageFilter, ImageDraw
import ast

file_path = Path(__file__)  # or Path("your/file/path.py")
parent_path = file_path.parent


def generate_two_lines_image(
    line1_length: float,
    line2_length: float,
    width: int = 400,
    height: int = 120,
    line1_y: int = 40,
    line2_y: int = 80,
    save_path: str = "two_lines.png",
    line_width: int = 4,
    color1: str = "black",
    color2: str = "black",
):
    """
    Generate an image with two horizontal lines of specified lengths.

    Args:
        line1_length: Fraction (0-1) of the width for the first line.
        line2_length: Fraction (0-1) of the width for the second line.
        width: Width of the image in pixels.
        height: Height of the image in pixels.
        line1_y: Vertical position of the first line.
        line2_y: Vertical position of the second line.
        save_path: File path to save the image.
        line_width: Thickness of the lines.
        color1: Color of the first line.
        color2: Color of the second line.
    """
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    # Draw first line
    ax.plot(
        [0, line1_length * width],
        [line1_y, line1_y],
        color=color1,
        linewidth=line_width,
        zorder=2,
    )

    # Draw second line
    ax.plot(
        [0, line2_length * width],
        [line2_y, line2_y],
        color=color2,
        linewidth=line_width,
        zorder=2,
    )

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def generate_line_samples(
    min_fraction: float = 0.1,
    max_fraction: float = 0.3,
    n_samples: int = 100,
    output_dir: str = None,
    seed: int = 42,
) -> pd.DataFrame:
    """ """
    round_decimals = 4
    np.random.seed(seed)

    # Uniformly sample target lengths
    line_ratios = np.random.uniform(min_fraction, max_fraction, n_samples)

    samples = []
    os.makedirs(output_dir, exist_ok=True)
    for i, line_ratio in enumerate(line_ratios):
        # Calculate line A length based on target length
        image_path = f"sample_{i:03d}_{round(line_ratio, round_decimals)}.png"

        # Generate the meta data dataframe
        samples.append(
            {
                "sample_id": f"sample_{i+1:03d}",
                "actual_length": round(line_ratio, round_decimals),
                "image_path": f"{min_fraction}_{max_fraction}/{image_path}",
            }
        )
        # Generate the image
        generate_two_lines_image(
            line1_length=line_ratio,
            line2_length=1.0,
            save_path=os.path.join(output_dir, image_path),
        )

    # Create DataFrame
    samples_df = pd.DataFrame(samples)

    # Save to CSV with informative filename
    filename = f"line_len_ratio_{min_fraction}_{max_fraction}_{n_samples}samples.csv"
    filepath = os.path.join(output_dir, filename)
    samples_df.to_csv(filepath, index=False)

    print(f"Generated {len(samples)} line length samples")
    print(f"Length range: {min_fraction} - {max_fraction}")
    print(f"Saved to: {filepath}")

    return samples_df


def generate_noisy_ascii_two_lines_ratio(
    ratio: float,
    line_length: int = 40,
    noise_level1: int = 2,
    noise_level2: int = 2,
) -> str:
    """
    Generate two noisy ASCII lines, the first with length = ratio * line_length,
    the second with length = line_length. No dots.

    Args:
        ratio: Fraction (0-1) for the first line's length.
        line_length: Number of characters for the full line.
        noise_level1: Noise level for the first line.
        noise_level2: Noise level for the second line.

    Returns:
        str: Two-line ASCII representation.
    """
    import random

    def build_noisy_line(effective_length, noise_level):
        line_chars = ["-"] * effective_length
        for _ in range(noise_level):
            idx = random.randint(0, effective_length - 1)
            if line_chars[idx] == "-":
                line_chars[idx] = random.choice([" ", "=", ".", "~"])
        return "|" + "".join(line_chars).ljust(line_length) + "|"

    len1 = max(1, int(ratio * line_length))
    len2 = line_length

    line1 = build_noisy_line(len1, noise_level1)
    line2 = build_noisy_line(len2, noise_level2)
    return line1, line2


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

        coord_end = dot_location = stimulus_df.loc[i, "actual_length"]

        img_path = Path(folder_path / img_path)
        img = Image.open(img_path).convert("RGBA")
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # Create a mask: white (255) where you want blur, black (0) elsewhere
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        center = (
            dot_location * img.size[0],
            0.65 * img.size[1],
        )  # x, y coordinates of blur center
        blur_radius = 30  # radius of blur region
        draw.ellipse(
            [
                center[0] - blur_radius,
                center[1] - blur_radius * 0.5,
                center[0] + blur_radius,
                center[1] + blur_radius * 0.5,
            ],
            fill=255,
        )

        # Composite: blur only where mask is white
        result = Image.composite(blurred, img, mask)
        if radius != 5:
            output_dir = img_path.parent / f"blurred_{radius}"
        else:
            output_dir = img_path.parent / "blurred"
        output_dir.mkdir(
            parents=True, exist_ok=True
        )  # Create the folder if it doesn't exist

        result.save(output_dir / f"{img_path.stem}_{suffix}.png")


if __name__ == "__main__":
    # Example usage

    # Example usage
    generate_new_data = False
    # Generate new Data
    if generate_new_data:

        len_ranges = [(0.1, 0.5), (0.3, 0.8), (0.5, 0.9)]
        for min_frac, max_frac in len_ranges:
            generate_line_samples(
                min_fraction=min_frac,
                max_fraction=max_frac,
                n_samples=100,
                output_dir=f"{parent_path}/data/experiment_files/{min_frac}_{max_frac}/",
                seed=42,
            )

            sample_df = pd.read_csv(
                f"{parent_path}/data/experiment_files/{min_frac}_{max_frac}/line_len_ratio_{min_frac}_{max_frac}_100samples.csv"
            )

            # Generate base ASCII lines
            ascii_lines = []
            for i in range(len(sample_df)):
                ascii_line = generate_noisy_ascii_two_lines_ratio(
                    ratio=sample_df["actual_length"].iloc[i],
                    line_length=40,
                )
                ascii_lines.append(ascii_line)
            sample_df["ascii_line"] = ascii_lines

            # # Generate high noise ASCII lines
            # high_noise_ascii_lines = []
            # for i in range(len(sample_df)):
            #     ascii_line = generate_noisy_ascii_line_dot(
            #         dot_position=sample_df["actual_length"].iloc[i],
            #         line_length=40,
            #         noise_level=10,
            #     )
            #     high_noise_ascii_lines.append(ascii_line)
            # sample_df["high_noise_ascii_line"] = high_noise_ascii_lines

            sample_df.to_csv(
                f"{parent_path}/data/experiment_files/{min_frac}_{max_frac}/line_len_ratio_{min_frac}_{max_frac}_100samples.csv",
                index=False,
            )

    else:
        for file in [
            "0.1_0.5/line_len_ratio_0.1_0.5_100samples.csv",
            "0.3_0.8/line_len_ratio_0.3_0.8_100samples.csv",
            "0.5_0.9/line_len_ratio_0.5_0.9_100samples.csv",
        ]:

            for blur_radius in [2, 5, 9]:
                gaussian_blur_img(
                    f"{parent_path}/data/experiment_files/{file}", radius=blur_radius
                )
                df = pd.read_csv(f"{parent_path}/data/experiment_files/{file}")
                if blur_radius == 2:
                    df["blurred_low_image_path"] = df["image_path"].apply(
                        lambda x: f"{Path(x).parent}/blurred_{blur_radius}/{Path(x).stem}_blurred.png"
                    )
                if blur_radius == 5:
                    df["blurred_image_path"] = df["image_path"].apply(
                        lambda x: f"{Path(x).parent}/blurred/{Path(x).stem}_blurred.png"
                    )
                if blur_radius == 9:
                    df["blurred_high_image_path"] = df["image_path"].apply(
                        lambda x: f"{Path(x).parent}/blurred_{blur_radius}/{Path(x).stem}_blurred.png"
                    )
                df.to_csv(f"{parent_path}/data/experiment_files/{file}", index=False)

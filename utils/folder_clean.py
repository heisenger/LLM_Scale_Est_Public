import os
import shutil


def clean_and_move_results(folder_path):
    # Skip 'configs' folder
    for entry in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, entry)
        if entry == "configs" or not os.path.isdir(subfolder_path):
            continue
        # Now process each subfolder (e.g., openai_gpt-4o-2024-08-06)
        aw_dir = os.path.join(subfolder_path, "raw")
        os.makedirs(aw_dir, exist_ok=True)
        for fname in os.listdir(subfolder_path):
            fpath = os.path.join(subfolder_path, fname)
            if os.path.isfile(fpath):
                if fname.startswith("results_"):
                    shutil.move(fpath, os.path.join(aw_dir, fname))
                else:
                    os.remove(fpath)


# Example usage:
clean_and_move_results("experiments/text_image/marker_location/runs/100825_4_image")

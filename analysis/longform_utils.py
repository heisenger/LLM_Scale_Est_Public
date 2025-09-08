from analysis.model_fit import w_p_conversion
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from pathlib import Path

base_dir = Path(__file__).parent


import os
import pandas as pd
from datetime import datetime
from .nrmse_summary import save_nrmse_observed_csv
from .factor_summary import save_factor_csvs

# =========================
# Create one summary long form df for model response
# =========================


def collect_experimental_data(
    runs_root: str,
    output_path: str,
    csv_name: str = "experimental_data.csv",
) -> pd.DataFrame:
    """
    Scan runs/<exp_id>/<llm_model>/derived/experimental_data.csv (excluding exp_id=='prior'),
    stack them into one long-form DataFrame, and save to Parquet/CSV.
    """
    matches = []
    for dirpath, _, filenames in os.walk(runs_root):
        if csv_name in filenames and os.path.basename(dirpath) == "derived":
            # rel path: <exp_id>/<llm_model>/derived
            rel = os.path.relpath(dirpath, runs_root)
            parts = rel.split(os.sep)
            exp_id = parts[0] if parts else ""
            if exp_id.lower() == "prior":
                continue
            matches.append(os.path.join(dirpath, csv_name))

    if not matches:
        raise RuntimeError(f"No '{csv_name}' found under: {runs_root}")

    frames = []
    for fpath in sorted(matches):
        rel = os.path.relpath(fpath, runs_root)
        parts = rel.split(os.sep)
        exp_id = parts[0] if len(parts) >= 1 else None
        llm_model = parts[1] if len(parts) >= 3 else None

        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"[skip] {fpath}: {e}")
            continue

        # Ensure expected columns exist; coerce numerics safely
        expected = [
            "correct",
            "response",
            "input_value",
            "range_category",
            "trial",
            "stimulus_id",
        ]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            print(f"[skip] {fpath}: missing columns {missing}")
            continue

        for c in ["correct", "response", "input_value"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in ["trial", "stimulus_id"]:
            # trials/stim IDs are often ints, but keep as numeric with coercion
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Add metadata columns
        df["exp_id"] = exp_id
        df["llm_model"] = llm_model
        df["csv_path"] = fpath
        df["ingest_ts"] = datetime.now().isoformat(timespec="seconds")

        # Helpful composite keys for later merges/group-bys
        df["run_llm_key"] = f"{exp_id}/{llm_model}"

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith(".parquet"):
        out.to_parquet(output_path, index=False)
    elif output_path.endswith(".csv"):
        out.to_csv(output_path, index=False)
    else:
        raise ValueError("output_path must end with .parquet or .csv")

    print(f"[ok] {len(out)} rows from {len(frames)} CSVs → {output_path}")
    return out


def summarize_experimental_data_stats(
    input_csv: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Group by range_category, stimulus_id, exp_id, llm_model and compute:
      - mean of input_value, correct, response
      - variance of response and correct
      - squared bias (mean response - mean correct)^2
    Saves to output_path (.csv or .parquet).
    """
    df = pd.read_csv(input_csv)
    grouped = df.groupby(["range_category", "stimulus_id", "exp_id", "llm_model"])

    stats = grouped.agg(
        mean_input_value=("input_value", "mean"),
        mean_correct=("correct", "mean"),
        mean_response=("response", "mean"),
        var_response=("response", "var"),
        var_correct=("correct", "var"),
    ).reset_index()

    stats["squared_bias"] = (stats["mean_response"] - stats["mean_correct"]) ** 2

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith(".parquet"):
        stats.to_parquet(output_path, index=False)
    elif output_path.endswith(".csv"):
        stats.to_csv(output_path, index=False)
    else:
        raise ValueError("output_path must end with .parquet or .csv")

    print(f"[ok] Saved {len(stats)} grouped rows to {output_path}")
    return stats


# =========================
# Create one summary long form df for psychophysics response
# =========================


import os
import pandas as pd


def collect_psychophys_summary(
    runs_root: str,
    out_csv: str = None,
    out_parquet: str = None,
) -> pd.DataFrame:
    """
    Walks runs_root/**/derived/psychophysics_summary.csv and concatenates them.

    Assumes directory layout:
      runs/{exp_id}/{llm_model}/derived/psychophysics_summary.csv

    Skips any path segment named 'prior'.
    """
    records = []
    expected_cols = [
        "range_category",
        "regression_angle",
        "scalar_variability",
        "short_vs_medium_t",
        "medium_vs_long_t",
        "response_autocorr",
        "stimulus_autocorr",
    ]

    for dirpath, _, filenames in os.walk(runs_root):
        # only look inside "derived" directories
        if os.path.basename(dirpath) != "derived":
            continue
        if "psychophysics_summary.csv" not in filenames:
            continue

        # skip any path that includes a 'prior' segment
        rel = os.path.relpath(dirpath, runs_root)
        parts = rel.split(os.sep)
        if "prior" in [p.lower() for p in parts]:
            continue

        # parse exp_id and llm_model from path: {exp_id}/{llm_model}/derived
        if len(parts) < 2:
            # not the expected layout; ignore
            continue
        exp_id = parts[0]
        llm_model = parts[1]

        fpath = os.path.join(dirpath, "psychophysics_summary.csv")
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"[warn] Failed to read {fpath}: {e}")
            continue

        # keep only columns we expect (if present)
        cols_present = [c for c in expected_cols if c in df.columns]
        df = df[cols_present].copy()

        # attach context
        df["exp_id"] = exp_id
        df["llm_model"] = llm_model

        # coerce numeric columns (leaves range_category as-is)
        for c in df.columns:
            if c not in ("range_category", "exp_id", "llm_model"):
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # consistent column order
        order = ["exp_id", "llm_model", "range_category"] + [
            c for c in expected_cols if c != "range_category" and c in df.columns
        ]
        df = df[order]

        records.append(df)

    if not records:
        raise RuntimeError(f"No 'psychophysics_summary.csv' found under: {runs_root}")

    out = pd.concat(records, ignore_index=True)

    # optional writes
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        out.to_csv(out_csv, index=False)
    if out_parquet:
        os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
        out.to_parquet(out_parquet, index=False)

    # quick preview
    print(f"[ok] Collected {len(out)} rows from psychophysics summaries.")
    print(out.head(6).to_string(index=False))
    return out


# =========================
# Create one summary long form df for behavioral model fit
# =========================


def _to_list_from_str_array(s):
    """Parse pretty-printed numpy-like string '[1 2 3]' to list of floats."""
    if isinstance(s, (list, np.ndarray)):
        return list(s)
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    arr = np.fromstring(s.replace("\n", " "), sep=" ")
    return arr.tolist()


def _split_model_key(key: str):
    """'StaticGain_basic' -> ('StaticGain','basic'); 'Linear_weber' -> ('Linear','weber')"""
    if "_" in key:
        name, family = key.split("_", 1)
    else:
        name, family = key, None
    return name, family


def _flatten_params(row_dict: dict, params: dict):
    """
    Add params as 'param_<name>' columns.
    If 'r_div_q' is present, convert it to 'param_w_p' using _derive_wp.
    """
    if not isinstance(params, dict):
        return row_dict

    for k, v in params.items():
        if k == "r_div_q":
            try:
                row_dict["param_w_p"] = w_p_conversion(v)
            except Exception:
                row_dict["param_w_p"] = None
        else:
            row_dict[f"param_{k}"] = v

    return row_dict


def nested_results_to_long_df(results: dict) -> pd.DataFrame:
    rows = []

    # per-range
    for range_id, models in results.get("per_range", {}).items():
        for key, block in models.items():
            name, family = _split_model_key(key)
            base = {
                "scope": "per_range",
                "range_id": range_id,
                "model_key": key,
                "model_name": name,
                "family": family,
                "k": block.get("k"),
                "aic": block.get("aic"),
                "nll": block.get("nll"),
                "mse": block.get("mse"),
                "r2": block.get("r2"),
                "pred": _to_list_from_str_array(block.get("pred")),
            }
            params = block.get("params", {})
            rows.append(_flatten_params(base, params))

    # overall
    for key, block in results.get("overall", {}).items():
        name, family = _split_model_key(key)
        base = {
            "scope": "overall",
            "range_id": None,
            "model_key": key,
            "model_name": name,
            "family": family,
            "k": block.get("k"),
            "aic": block.get("aic"),
            "nll": block.get("nll"),
            "mse": block.get("mse"),
            "r2": block.get("r2"),
            "pred": _to_list_from_str_array(block.get("pred")),
        }
        params = block.get("params", {})
        rows.append(_flatten_params(base, params))

    df = pd.DataFrame(rows)
    return df


# =========================
# go through folders to extract results json
# =========================
import os
import glob
import json
import pandas as pd
from typing import Tuple, List, Dict, Any


def _safe_load_messages(cell: Any) -> List[Dict[str, Any]]:
    """
    Best-effort loader: CSV stores a JSON-encoded list of {'role','content'}.
    Returns [] if it can’t parse.
    """
    if isinstance(cell, list):
        return cell
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    try:
        return json.loads(s)
    except Exception:
        # Sometimes extra quotes/escapes sneak in; try a gentle cleanup
        try:
            s2 = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
            return json.loads(s2)
        except Exception:
            return []


def _summarize_messages(msgs: List[Dict[str, Any]], sep=" | ") -> Dict[str, Any]:
    roles, contents = [], []
    users, assists, systems = [], [], []
    for m in msgs:
        role = str(m.get("role", "")).strip()
        content = m.get("content", "")
        # flatten content if it’s non-string (e.g., list of dicts)
        if not isinstance(content, str):
            try:
                content = json.dumps(content, ensure_ascii=False)
            except Exception:
                content = str(content)
        roles.append(role)
        contents.append(content)

        if role == "user":
            users.append(content)
        elif role == "assistant":
            assists.append(content)
        elif role == "system":
            systems.append(content)

    last_assistant = assists[-1] if assists else ""
    return {
        "roles_csv": ",".join(roles),
        "prompt_contents_joined": sep.join(contents),
        "user_contents_joined": sep.join(users),
        "assistant_contents_joined": sep.join(assists),
        "system_contents_joined": sep.join(systems),
        "response_raw": last_assistant,  # last assistant = final answer for the sample
        "n_turns": len(msgs),
        "n_user_turns": len(users),
        "n_assistant_turns": len(assists),
        "n_system_turns": len(systems),
    }


def _explode_turns(
    msgs: List[Dict[str, Any]],
    *,
    experiment_id: str,
    model_name: str,
    result_file: str,
    sample_id: str,
) -> List[Dict[str, Any]]:
    rows = []
    for i, m in enumerate(msgs):
        role = str(m.get("role", "")).strip()
        content = m.get("content", "")
        if not isinstance(content, str):
            try:
                content = json.dumps(content, ensure_ascii=False)
            except Exception:
                content = str(content)
        rows.append(
            {
                "experiment_id": experiment_id,
                "model_name": model_name,
                "result_file": result_file,
                "sample_id": sample_id,
                "turn_idx": i,
                "role": role,
                "content_text": content,
            }
        )
    return rows


def _read_head_csv(path: str, n: int) -> pd.DataFrame:
    # Robust CSV read for potentially large files; restrict to head n.
    # We read normally and then head(n). (Chunked read would complicate dtype.)
    df = pd.read_csv(path)
    return df.head(n)


def collect_llm_raw_interactions_for_first_model_all_results(
    runs_root: str,
    out_samples_parquet: str,
    out_turns_parquet: str,
    head_n: int = 15,
    results_glob: str = "results_*.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk all experiments under runs_root.
    For each experiment folder, pick the FIRST model (alphabetical),
    read every raw/{results_*.csv}, take only the first `head_n` rows,
    parse prompt_sent into sample-level summary + turn-level expansion,
    and save to parquet.

    Returns (samples_df, turns_df).
    """
    if not os.path.isdir(runs_root):
        raise RuntimeError(f"runs_root does not exist: {runs_root}")

    all_samples = []
    all_turns = []

    # experiments are direct children of runs_root
    for exp_dir in sorted(
        [d for d in glob.glob(os.path.join(runs_root, "*")) if os.path.isdir(d)]
    ):
        experiment_id = os.path.basename(exp_dir)

        # find model subfolders
        model_dirs = sorted(
            [d for d in glob.glob(os.path.join(exp_dir, "*")) if os.path.isdir(d)]
        )
        if not model_dirs:
            continue

        # pick the first model only
        model_dir = model_dirs[0]
        model_name = os.path.basename(model_dir)

        raw_dir = os.path.join(model_dir, "raw")
        if not os.path.isdir(raw_dir):
            # some runs may not have 'raw'
            continue

        result_paths = sorted(glob.glob(os.path.join(raw_dir, results_glob)))
        if not result_paths:
            continue

        for rp in result_paths:
            result_file = os.path.basename(rp)
            try:
                df = _read_head_csv(rp, head_n)
            except Exception as e:
                print(f"[warn] Failed to read {rp}: {e}")
                continue

            # expected columns (but keep robust)
            # model,sample_id,text,ground_truth,input_values,prediction,prompt_sent,response
            # parse messages
            msgs_col = "prompt_sent"
            if msgs_col not in df.columns:
                # skip if no structured prompts
                continue

            for _, row in df.iterrows():
                sample_id = str(row.get("sample_id", ""))
                msgs = _safe_load_messages(row.get(msgs_col, ""))

                # sample-level
                summ = _summarize_messages(msgs)
                sample_row = {
                    "experiment_id": experiment_id,
                    "model_name": model_name,
                    "result_file": result_file,
                    "sample_id": sample_id,
                    "model_field": row.get("model", ""),
                    "text": row.get("text", ""),
                    "ground_truth": row.get("ground_truth", None),
                    "input_values": row.get("input_values", None),
                    "prediction": row.get("prediction", None),
                    "response_col": row.get("response", ""),
                    **summ,
                }
                all_samples.append(sample_row)

                # turn-level
                all_turns.extend(
                    _explode_turns(
                        msgs,
                        experiment_id=experiment_id,
                        model_name=model_name,
                        result_file=result_file,
                        sample_id=sample_id,
                    )
                )

    if not all_samples:
        raise RuntimeError(f"No results found under: {runs_root}")

    samples_df = pd.DataFrame(all_samples)
    turns_df = (
        pd.DataFrame(all_turns)
        if all_turns
        else pd.DataFrame(
            columns=[
                "experiment_id",
                "model_name",
                "result_file",
                "sample_id",
                "turn_idx",
                "role",
                "content_text",
            ]
        )
    )

    # Ensure artifacts folder exists
    for out_path in [out_samples_parquet, out_turns_parquet]:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Parquet-friendly: coerce obvious numeric columns; leave text columns as-is
    for c in ("ground_truth", "input_values", "prediction"):
        if c in samples_df.columns:
            try:
                samples_df[c] = pd.to_numeric(samples_df[c])
            except Exception:
                pass

    # --- Make Parquet happy: force all text-ish columns to strings ---
    text_like_cols = [
        "model_field",
        "text",
        "response_col",
        "response_raw",
        "roles_csv",
        "prompt_contents_joined",
        "user_contents_joined",
        "assistant_contents_joined",
        "system_contents_joined",
        "experiment_id",
        "model_name",
        "result_file",
        "sample_id",
    ]

    for c in text_like_cols:
        if c in samples_df.columns:
            samples_df[c] = samples_df[c].fillna("").astype(str)

    # also for the turns frame
    turn_text_cols = [
        "experiment_id",
        "model_name",
        "result_file",
        "sample_id",
        "role",
        "content_text",
    ]
    for c in turn_text_cols:
        if c in turns_df.columns:
            turns_df[c] = turns_df[c].fillna("").astype(str)

    samples_df.to_parquet(out_samples_parquet, index=False)
    turns_df.to_parquet(out_turns_parquet, index=False)

    print(f"[ok] samples_df: {len(samples_df)} rows → {out_samples_parquet}")
    print(f"[ok] turns_df:   {len(turns_df)} rows → {out_turns_parquet}")
    return samples_df, turns_df


# =========================
# go through folders to extract model comparison json
# =========================


def collect_combo_longform(
    runs_root: str,
    output_path: str,
    json_name: str = "model_comparison_aggregate.json",
) -> pd.DataFrame:
    """
    sweep through the directories to find JSON files for model comparisons, ignoring 'prior' runs
    """
    import os, json
    import pandas as pd
    from datetime import datetime

    matches = []
    for dirpath, _, filenames in os.walk(runs_root):
        if json_name in filenames and os.path.basename(dirpath) == "derived":
            # rel path like: <exp_id>/<llm_model>/derived/file.json
            rel = os.path.relpath(dirpath, runs_root)
            parts = rel.split(os.sep)
            exp_id = parts[0] if parts else ""
            if exp_id.lower() == "prior":
                continue  # <-- skip the 'prior' run
            matches.append(os.path.join(dirpath, json_name))

    if not matches:
        raise RuntimeError(
            f"No '{json_name}' (excluding 'prior') found under: {runs_root}"
        )

    frames = []
    for fpath in sorted(matches):
        rel = os.path.relpath(fpath, runs_root)
        parts = rel.split(os.sep)
        exp_id = parts[0] if len(parts) >= 1 else None
        llm_model = parts[1] if len(parts) >= 3 else None

        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            df = nested_results_to_long_df(data)  # your flattener
        except Exception as e:
            print(f"[skip] {fpath}: {e}")
            continue

        df["exp_id"] = exp_id
        df["llm_model"] = llm_model
        df["json_path"] = fpath
        df["ingest_ts"] = datetime.now().isoformat(timespec="seconds")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith(".parquet"):
        out.to_parquet(output_path, index=False)
    elif output_path.endswith(".csv"):
        out.to_csv(output_path, index=False)
    else:
        raise ValueError("output_path must end with .parquet or .csv")

    print(
        f"[ok] {len(out)} rows from {len(frames)} JSONs (excluding 'prior') → {output_path}"
    )
    return out


# =========================
# go through folders to extract cue combination results
# =========================

import os
import json
import pandas as pd
from pathlib import Path


def collect_runset_results(runsets_root: str, out_parquet: str) -> dict:
    """
    Collect long-form DataFrames for runset-level files:
    - weights_bayes_per_stimulus.csv
    - weights_bayes_per_range.csv
    - weights_global.json
    - overall.csv

    Saves them as parquet in `out_parquet` and returns them as dict of DataFrames.
    """
    weights_stimulus = []
    weights_range = []
    weights_global = []
    overall = []

    for root, _, files in os.walk(runsets_root):
        files = set(files)
        if not any(f.startswith("weights") or f == "overall.csv" for f in files):
            continue

        parts = Path(root).parts
        try:
            # runset = parts[-3]  # e.g. "040825"
            base = parts[-2]  # e.g. "base"
            model = parts[-1]  # e.g. "anthropic_claude-3.7-sonnet"
        except IndexError:
            continue

        if "weights_global.json" in files:
            with open(os.path.join(root, "weights_global.json")) as f:
                j = json.load(f)
            # flatten dict: prefix each param with its top-level key
            flat = {}
            for top_key, params in j.items():
                if isinstance(params, dict):
                    for k, v in params.items():
                        flat[f"{top_key}_{k}"] = v
                else:
                    flat[top_key] = params
            flat["base"], flat["LLM_model"] = base, model
            weights_global.append(flat)

        if "overall.csv" in files:
            df = pd.read_csv(os.path.join(root, "overall.csv"))
            df["base"], df["LLM_model"] = base, model
            overall.append(df)

    # Concatenate
    weights_stimulus_df = (
        pd.concat(weights_stimulus, ignore_index=True)
        if weights_stimulus
        else pd.DataFrame()
    )
    weights_range_df = (
        pd.concat(weights_range, ignore_index=True) if weights_range else pd.DataFrame()
    )
    weights_global_df = (
        pd.DataFrame(weights_global) if weights_global else pd.DataFrame()
    )
    overall_df = pd.concat(overall, ignore_index=True) if overall else pd.DataFrame()

    # Save all into a single parquet (multi-table not supported → save dict as parquet with suffix)
    out_dir = Path(out_parquet).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # import pdb

    # pdb.set_trace()
    weights_global_df.to_parquet(out_dir / "weights_global.parquet", index=False)
    overall_df.to_parquet(out_dir / "overall_cue_combo.parquet", index=False)

    return {
        "weights_stimulus": weights_stimulus_df,
        "weights_range": weights_range_df,
        "weights_global": weights_global_df,
        "overall": overall_df,
    }


if __name__ == "__main__":
    # Example usage
    experiment_list = [
        "text_image/line_length_ratio",
        "text_image/marker_location",
        "text_image/maze_distance",
        # "text/subtitle_duration",
    ]

    for experiment in experiment_list:
        collect_combo_longform(
            runs_root=f"{base_dir}/../experiments/{experiment}/runs",
            output_path=f"{base_dir}/../experiments/{experiment}/artifacts/behavioral_models.parquet",
        )
        collect_experimental_data(
            runs_root=f"{base_dir}/../experiments/{experiment}/runs",
            output_path=f"{base_dir}/../experiments/{experiment}/artifacts/experimental_data.csv",
        )

        summarize_experimental_data_stats(
            input_csv=f"{base_dir}/../experiments/{experiment}/artifacts/experimental_data.csv",
            output_path=f"{base_dir}/../experiments/{experiment}/artifacts/experimental_data_summary.csv",
        )
        collect_psychophys_summary(
            runs_root=f"{base_dir}/../experiments/{experiment}/runs",
            out_csv=f"{base_dir}/../experiments/{experiment}/artifacts/psychophysics.csv",
        )

        samples_df, turns_df = collect_llm_raw_interactions_for_first_model_all_results(
            runs_root=f"{base_dir}/../experiments/{experiment}/runs",
            out_samples_parquet=f"{base_dir}/../experiments/{experiment}/artifacts/llm_prompt_preview.samples.parquet",
            out_turns_parquet=f"{base_dir}/../experiments/{experiment}/artifacts/llm_prompt_preview.turns.parquet",
            head_n=15,  # only first 15 rows per results_*.csv
            results_glob="results_*.csv",
        )

        save_nrmse_observed_csv(
            f"{base_dir}/../experiments/{experiment}/artifacts/experimental_data.csv",
            f"{base_dir}/../experiments/{experiment}/artifacts",
            experiment.split("/", 1)[1],
            baseline="experiment_mean",
        )

        save_factor_csvs(
            f"{base_dir}/../experiments/{experiment}/artifacts/behavioral_models.parquet",
            f"{base_dir}/../experiments/{experiment}/artifacts",
            experiment.split("/", 1)[1],
        )

        (
            weights_stimulus_df,
            weights_range_df,
            weights_global_df,
            overall_df,
        ) = collect_runset_results(
            runsets_root=f"{base_dir}/../experiments/{experiment}/runsets",  # f"{base_dir}/../experiments/{experiment}/runs/prior/170825/runsets",
            out_parquet=f"{base_dir}/../experiments/{experiment}/artifacts/runsets_output.parquet",  # f"{base_dir}/../experiments/{experiment}/runs/prior/170825//artifacts/runsets_output.parquet"
        )

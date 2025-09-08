import pandas as pd
import os
from pathlib import Path
from analysis.data_processing import clean_experiment_df
from analysis.model_fit import compare_models, summarize_results_df
from analysis.visualizations import plot_psychophysics_overview
from analysis.psychophysics_laws import (
    analyze_psychophysics_laws,
    summarize_psychophysics_laws,
)
from models.interface import LLM_Interface
import importlib
from utils.mappings import config_map, model_folder_map
from utils.misc import get_explicit_class_vars
import json


def run_suite_experiment(
    config: type,
    models: list = None,
    run_experiment: bool = True,
):
    """
    Run experiment with a simple config dict.

    Args:
        config: Experiment configuration dict
        models: List of model names to test
        run_experiment: Whether to run experiments or load existing results

    Returns:
        dict: Results for each model
    """

    # Import experiment_run.py
    module = importlib.import_module(config.experiment_module)
    # Import function: run_experiment
    experiment_function = getattr(module, config.experiment_function)

    # Ensure experiment path exists
    experiment_path = Path(config.experiment_path)
    experiment_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for model in models:
        print(f"Running experiment for model: {model}")
        result_list = []
        model_folder_name = model_folder_map[model]

        # Ensure relevant directories exist
        output_dir = experiment_path / model_folder_name
        os.makedirs(output_dir, exist_ok=True)
        results_dir = output_dir / "raw"
        os.makedirs(results_dir, exist_ok=True)

        # Run or load experiment results
        for experiment_file in config.experiment_files:
            result_file = results_dir / f"results_{Path(experiment_file).name}"

            if run_experiment:
                if result_file.exists():
                    print(
                        f"Result file {result_file} already exists. Skipping to avoid overwrite."
                    )
                    results_data = pd.read_csv(result_file)
                    result_list.append(results_data)
                else:
                    # Prepare experiment parameters
                    experiment_params = {
                        "models": [model],
                        "llm_object": LLM_Interface,
                        "experiment_file": experiment_file,
                        "config": config,
                    }

                    # Add any extra parameters
                    experiment_params.update(config.extra_params)

                    # Run the experiment
                    results_data = experiment_function(**experiment_params)

                    # Save results
                    results_data.to_csv(result_file, index=False)
                    result_list.append(results_data)
            else:
                # Load existing results
                results_data = pd.read_csv(result_file)
                if config.input_column not in results_data.columns:
                    results_data[config.input_column] = results_data[config.true_column]
                result_list.append(results_data)

        if run_experiment:
            # Save config when we run an experiment
            used_configs = [
                vars(config),
                get_explicit_class_vars(type(config)),
            ]
            config_dir = Path(config.experiment_path) / "configs"
            os.makedirs(config_dir, exist_ok=True)
            with open(
                f"{config_dir}/used_experiment_configs.json",
                "w",
            ) as f:
                json.dump(used_configs, f, indent=2, default=str)

        # Run analysis on experimental data
        derived_dir = output_dir / "derived"
        os.makedirs(derived_dir, exist_ok=True)

        # Clean experimental data
        experimental_data = clean_experiment_df(
            result_list,
            true_column=config.true_column,
            pred_column=config.pred_column,
            input_column=config.input_column,
            block_size=config.block_size,
        )
        experimental_data.to_csv(
            derived_dir / f"experimental_data.csv",
            index=False,
        )
        # import pdb; pdb.set_trace()

        # Fit behavioral models
        aggregate_comparison = compare_models(
            experimental_data,
            save_results=True,
            results_filename=str(derived_dir / "model_comparison_aggregate.json"),
        )
        summarize_results_df(
            aggregate_comparison,
            filepath=str(derived_dir / "model_comparison_summary.csv"),
        )

        # # Psychophysics analysis
        # psychophysics_analysis = analyze_psychophysics_laws(
        #     experimental_data,
        #     save_path=str(derived_dir / "psychophysics_analysis_aggregate.json"),
        # )
        # summarize_psychophysics_laws(
        #     psychophysics_analysis,
        #     save_path=str(derived_dir / "psychophysics_summary.csv"),
        # )

        results[model] = experimental_data

    return results

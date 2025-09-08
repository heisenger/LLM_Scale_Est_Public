import os
import json
import argparse
import yaml

# from script.run_plots import run_plots
from utils.mappings import config_map, model_folder_map
from experiment_runners.run_experiments import run_suite_experiment
from configs.experiments.config import (
    CommonConfig,
)

if __name__ == "__main__":
    # Run all experiments in the list

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",  # Allow multiple config files
        default=["configs/experiments/exp_run.yaml"],
        help="Path(s) to experiment config YAML file(s)",
    )

    args = parser.parse_args()

    for config_file in args.config:
        used_configs = []
        experiment_list = []

        with open(config_file) as f:
            batch = yaml.safe_load(f)

        # Run each experiment in the batch
        for exp in batch["experiments"]:
            config_class = config_map[exp["class"]]
            experiment_config = config_class(**exp.get("params", {}))
            experiment_list.append(experiment_config)

            # Run experiments
            run_suite_experiment(
                config=experiment_config,
                models=experiment_config.DEFAULT_MODELS,
                run_experiment=True,
            )

        LLM_names = [
            (
                model_folder_map[model_name]
                if model_name in model_folder_map
                else model_name
            )
            for model_name in CommonConfig.DEFAULT_MODELS
        ]
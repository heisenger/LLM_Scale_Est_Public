# analysis/thesis/graph_codes/plot_heatmaps.py
from __future__ import annotations
import pandas as pd
from .config import COMBO_MODELS
from .plotters import plot_triptych


def triptych_by_modality(
    tidy: pd.DataFrame, value_col: str, *, title: str, save_as=None
):
    panels = []
    for mod in ["text", "image", "text_image"]:
        panels.append((tidy[tidy["modality"] == mod], value_col, mod.replace("_", "+")))
    return plot_triptych(
        panels,
        colorbar_label=value_col.replace("_", " ").title(),
        title=title,
        save_as=save_as,
    )


def plot_all_from_csvs(
    summary_dir: str, exp_tag: str, out_dir: str, multi_modal: bool = True
):
    # Factors
    for fac in ["bayesian", "weber", "sequential"]:
        fac_df = pd.read_csv(f"{summary_dir}/{exp_tag}_summary_factor_{fac}.csv")
        # import pdb

        # pdb.set_trace()
        triptych_by_modality(
            fac_df[
                [
                    "llm_model",
                    "model_pretty",
                    "ablation",
                    "ablation_pretty",
                    "modality",
                    "prob",
                ]
            ],
            value_col="prob",
            title=f"{fac.capitalize()}-ness (factor probability)",
            save_as=f"{out_dir}/{exp_tag}_grid_factor_{fac}.pdf",
        )
    nrmse_df = pd.read_csv(f"{summary_dir}/{exp_tag}_summary_nrmse_observed.csv")
    triptych_by_modality(
        nrmse_df[
            [
                "llm_model",
                "model_pretty",
                "ablation",
                "ablation_pretty",
                "modality",
                "nrmse",
            ]
        ],
        value_col="nrmse",
        title="Observed NRMSE (normalized to experiment mean)",
        save_as=f"{out_dir}/{exp_tag}_grid_nrmse_observed.pdf",
    )
    # Cue combination
    if multi_modal:
        cue = pd.read_csv(f"{summary_dir}/{exp_tag}_summary_cue_combo_long.csv")
        # import pdb

        # pdb.set_trace()
        for combo in COMBO_MODELS:
            sub = cue[cue["combo_model"] == combo]
            plot_triptych(
                [
                    (
                        sub[
                            [
                                "llm_model",
                                "model_pretty",
                                "ablation",
                                "ablation_pretty",
                                "prob",
                            ]
                        ],
                        "prob",
                        "Probability",
                    ),
                    (
                        sub[
                            [
                                "llm_model",
                                "model_pretty",
                                "ablation",
                                "ablation_pretty",
                                "nrmse",
                            ]
                        ],
                        "nrmse",
                        "NRMSE",
                    ),
                    (
                        sub[
                            [
                                "llm_model",
                                "model_pretty",
                                "ablation",
                                "ablation_pretty",
                                "w_img",
                            ]
                        ],
                        "w_img",
                        r"$w_{\mathrm{img}}$",
                    ),
                ],
                title=f"Cue combination — {combo}",
                colorbar_label="Value",
                save_as=f"{out_dir}/{exp_tag}_cue_triptych_{combo.replace('(','_').replace(')','').replace(' ','_')}.pdf",
            )

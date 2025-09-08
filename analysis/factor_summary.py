# analysis/thesis/graph_codes/factor_summary.py
from __future__ import annotations
import pandas as pd
from analysis.thesis_graphs.config import ABLATION_LABEL
from analysis.thesis_graphs.util import (
    add_ablation_modality,
    duplicate_text_base,
    pretty_model,
    ensure_dir,
)

# import your existing function:
from analysis.summary_notebook import factor_evidence


def build_factor_summary(
    beh_df: pd.DataFrame, factor="Bayesian", scope="overall", model_col="model_key"
) -> pd.DataFrame:
    rows = []
    # import pdb

    # pdb.set_trace()
    d = add_ablation_modality(beh_df)
    d = d[d["scope"] == scope]
    for (exp_id, llm), g in d.groupby(["exp_id", "llm_model"], sort=False):
        fe = factor_evidence(g, model_col=model_col)
        sel = (
            (fe["scope"] == scope) & (fe["factor"] == factor) & (fe["level"] == "True")
        )
        p = float(fe.loc[sel, "prob"].iloc[0]) if sel.any() else float("nan")
        r0 = g.iloc[0]
        rows.append(
            {
                "llm_model": llm,
                "ablation": r0["ablation"],
                "modality": r0["modality"],
                "prob": p,
            }
        )
    out = pd.DataFrame(rows)
    out = duplicate_text_base(out, ["prob"])
    out["model_pretty"] = pretty_model(out["llm_model"])
    out["ablation_pretty"] = out["ablation"].map(ABLATION_LABEL).fillna(out["ablation"])
    return out


def save_factor_csvs(beh_path: str, out_dir: str, exp_tag: str):
    beh = (
        pd.read_parquet(beh_path)
        if beh_path.endswith(".parquet")
        else pd.read_csv(beh_path)
    )
    ensure_dir(out_dir)
    for fac in ["Bayesian", "Weber", "Sequential"]:
        df = build_factor_summary(beh, factor=fac)
        df.insert(0, "factor", fac)
        df.to_csv(f"{out_dir}/{exp_tag}_summary_factor_{fac.lower()}.csv", index=False)

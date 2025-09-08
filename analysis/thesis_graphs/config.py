# analysis/thesis/graph_codes/config.py
from __future__ import annotations

# Pretty names
MODEL_NAME = {
    "anthropic_claude-3.7-sonnet": "Claude 3.7 Sonnet",
    "openai_gpt-4o-2024-08-06": "GPT 4o",
    # "openai_gpt-4.1-mini": "GPT 4.1 mini",
    "meta-llama_llama-4-maverick": "Llama 4 Maverick",
    "google_gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "qwen_qwen2.5-vl-32b-instruct": "Qwen2.5 VL 32B",
    "mistralai_mistral-small-3.2-24b-instruct": "Mistral 24B",
    "gemma-3-4b-it": "Gemma 3 4B it",
    "microsoft_phi-4-multimodal-instruct": "Phi 4 Multimodal",
    "GPT-5 Mini": "GPT-5 Mini",
}

ABLATION_ORDER = [
    "base",
    "base_1",
    "base_2",
    "base_3",
    "base_4",
    "base_5",
    "base_6",
    "base_7",
    "base_8",
]
ABLATION_LABEL = {
    "base": "Base",
    "base_1": "Verbal cue",
    "base_2": "Numerical cue",
    "base_3": "Constant noise",
    "base_4": "Shorter context",
    "base_5": "Gradual noise",
    "base_6": "Compressed Numerical cue",
    "base_7": "Longer context",
    "base_8": "Reverse order",
}

MODALITY_ORDER = ["text", "image", "text_image"]

# Cue-combo models you actually plot
COMBO_MODELS = ["EmpiricalLinear", "Bayes(stimulus)", "Bayes(range)", "Equal"]

# Plot style
MPL_STYLE = "thesis.mplstyle"


# Add near the bottom of config.py
MODEL_PLOT_ORDER = [
    "GPT-5 Mini",
    "Claude 3.7 Sonnet",
    "GPT 4o",
    "Qwen2.5 VL 32B",
    "Gemini 2.5 Flash Lite",
    "Llama 4 Maverick",
    "Mistral 24B",
    "Gemma 3 4B it",
    "Phi 4 Multimodal",
    # "GPT 4.1 mini",
]

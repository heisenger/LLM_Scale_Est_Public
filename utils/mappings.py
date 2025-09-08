import sys

sys.path.append("../")

from configs.experiments.config import (
    CommonConfig,
    # CharacterPositionConfig,
    # CityDistancesConfig,
    # MarkerLocationConfig,
    # NumLineIntervalConfig,
    # NumberPairDeltaConfig,
    # NumberSimilarityConfig,
    # StringLengthRatioConfig,
    # LineLengthConfig,
    SubtitleDuration,
    ImageMarkerLocation,
    ImageLineLengthRatio,
    # AnimalWeightsConfig,
    ImageMazeDistance,
    AudioSpeechDuration,
)

config_map = {
    "ImageMazeDistance": ImageMazeDistance,
    "ImageLineLengthRatio": ImageLineLengthRatio,
    "ImageMarkerLocation": ImageMarkerLocation,
    "AudioSpeechDuration": AudioSpeechDuration,
    "SubtitleDuration": SubtitleDuration,
}

model_name_map = {
    "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B",
    "meta-llama/llama-4-maverick": "Llama 4 Maverick",
    "google/gemini-3-4b-it": "Gemini 3 4B IT",
    "qwen/qwen3-32b": "Qwen3 32B",
    "openai/gpt-3.5-turbo": "GPT-3.5 Turbo",
    "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
    "anthropic/claude-3.7-sonnet": "Claude 3.7 Sonnet",
    "meta-llama/llama-4-maverick": "Llama 4 Maverick",
    "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "qwen/qwen-vl-plus": "Qwen VL Plus",
    "mistralai/mistral-small-3.2-24b-instruct": "Mistral Small 3.2",
    "openai/gpt-4.1-mini": "GPT-4.1 Mini",
    "openai/gpt-4o-2024-08-06": "GPT-4o",
    "anthropic/claude-3.7-sonnet": "Claude 3.7 Sonnet",
    "openai/gpt-5-mini": "GPT-5 Mini",
    "openai/gpt-oss-20b": "GPT-OSS 20B",
    "openai/gpt-5": "GPT-5",
}

model_folder_map = {
    "google/gemini-2.5-flash-lite": "google_gemini-2.5-flash-lite",
    "openai/gpt-4.1-mini": "openai_gpt-4.1-mini",
    "google/gemma-3-4b-it": "gemma-3-4b-it",
    "moonshotai/kimi-vl-a3b-thinking": "moonshotai_kimi_vl_a3b_thinking",
    "openai/gpt-oss-20b": "gpt_oss_20b",
    "meta-llama/llama-4-maverick": "meta-llama_llama-4-maverick",
    "meta-llama/llama-3.2-11b-vision-instruct:free": "llama-3.2-11b-vision-instruct",
    "mistralai/mistral-small-3.2-24b-instruct": "mistralai_mistral-small-3.2-24b-instruct",
    "qwen/qwen-vl-plus": "qwen_qwen-vl-plus",
    "qwen/qwen2.5-vl-32b-instruct": "qwen_qwen2.5-vl-32b-instruct",
    "microsoft/phi-4-multimodal-instruct": "microsoft_phi-4-multimodal-instruct",
    "openai/gpt-5-nano": "openai_gpt-5-nano",
    "openai/gpt-4o-2024-08-06": "openai_gpt-4o-2024-08-06",
    "anthropic/claude-3.7-sonnet": "anthropic_claude-3.7-sonnet",
    "meta-llama/llama-3.1-8b-instruct": "meta-llama_llama-3.1-8b-instruct",
    "qwen/qwen3-32b": "qwen_qwen3-32b",
    "qwen/qwen2.5-vl-72b-instruct:free": "qwen_qwen2.5-vl-72b-instruct",
    "google/gemini-2.0-flash-exp:free": "google_gemini-2.0-flash-exp",
    "openai/gpt-3.5-turbo": "openai_gpt-3.5-turbo",
    "meta-llama/llama-3.3-70b-instruct": "meta-llama_llama-3.3-70b-instruct",
    "llama-3.1-8b-instant": "llama-3.1-8b-instant",
    "mistral-saba-24b": "mistral-saba-24b",
    "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "meta-llama_llama-4-maverick-17b-128e-instruct",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "openai/gpt-5": "GPT-5",
    "openai/gpt-5-mini": "GPT-5 Mini",
}

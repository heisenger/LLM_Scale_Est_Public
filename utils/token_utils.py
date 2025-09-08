from transformers import AutoTokenizer

# Map Groq model IDs to Hugging Face tokenizer names
MODEL_TOKENIZER_MAP = {
    "llama-4-scout-17b-16e": "meta-llama/Meta-Llama-4-17B",
    "qwen3-32b-instruct": "Qwen/Qwen2-32B-Instruct",
    "mistral-saba-24b": "mistralai/Mixtral-8x22B",  # Approximate
    "llama-3.3-70b-versatile": "meta-llama/Meta-Llama-3-70B",
    "llama-3.1-8b-instant": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gemma-2-9b": "google/gemma-2b-it",
}


# Load tokenizers for all models
def load_tokenizers(model_map):
    tokenizer_dict = {}
    for groq_id, hf_model_id in model_map.items():
        tokenizer_dict[groq_id] = AutoTokenizer.from_pretrained(hf_model_id)
    return tokenizer_dict


# Count tokens in a message list (chat format)
def count_tokens(tokenizer, messages):
    return sum(len(tokenizer.encode(msg["content"])) for msg in messages)


# Utility function to get token count
def get_token_count_for_model(model_id, messages, tokenizer_dict):
    tokenizer = tokenizer_dict.get(model_id)
    if tokenizer is None:
        raise ValueError(f"No tokenizer found for model ID: {model_id}")
    return count_tokens(tokenizer, messages)


# Example usage
if __name__ == "__main__":
    tokenizers = load_tokenizers(MODEL_TOKENIZER_MAP)

    test_messages = [
        {"role": "system", "content": "Do not reason. Only return one number."},
        {
            "role": "user",
            "content": "Estimate how many seconds it takes to say: 'forty two elephants'.",
        },
    ]

    for model_id in MODEL_TOKENIZER_MAP.keys():
        count = get_token_count_for_model(model_id, test_messages, tokenizers)
        print(f"{model_id}: {count} tokens")

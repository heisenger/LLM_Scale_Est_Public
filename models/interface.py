import os
import pip
import requests
import numpy as np
import time
from dotenv import load_dotenv
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
import re
import json
import textwrap

# from unsloth import FastLanguageModel
from transformers import pipeline


class LLM_Interface:
    def __init__(
        self,
        provider="groq",  # "unsloth"
        model="llama3-8b-8192",
        task="angle_estimation",
    ):
        self.provider = provider
        self.model = model
        self.task = task
        self.history = ""  # for plain text history accumulation
        self.plaintext_mode = True if self.provider.lower() == "unsloth" else False

        # uncomment this if unsloth is available (problem installing it on mac)
        # if provider.lower() == "unsloth":
        #     # Load Unsloth model
        #     model, tokenizer = FastLanguageModel.from_pretrained(
        #         model_name=model,  # e.g. "marcelbinz/Llama-3.1-Minitaur-8B-adapter"
        #         max_seq_length=32768,
        #         dtype=None,
        #         load_in_4bit=True,
        #     )
        #     FastLanguageModel.for_inference(model)

        #     self.pipe = pipeline(
        #         "text-generation",
        #         model=model,
        #         tokenizer=tokenizer,
        #         trust_remote_code=True,
        #         pad_token_id=0,
        # do_sample=True,
        # temperature=0.3,
        # max_new_tokens=1,
        #     )
        #     self.pipe = None
        load_dotenv("models/env/api.env")
        if provider.lower() == "groq":
            self.api_key = os.getenv("GROQ_API_KEY")
            self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        if provider.lower() == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.endpoint = "https://api.openai.com/v1/chat/completions"
        if provider.lower() == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def ask(
        self,
        messages,
        max_tokens=150,
        reasoning_max_tokens=2000,  # but for reasoning models we need to budget more
        temperature=0.7,
        retry=True,
        max_retries=10,
        max_price=None,  # 0.2,
        backoff_factor=2,
        timeout=60,
    ):
        """
        Make LLM request with optional retry mechanism

        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            retry: Whether to retry on failures (default: True)
            max_retries: Maximum number of retry attempts (default: 3)
            backoff_factor: Exponential backoff multiplier (default: 2)
            timeout: Request timeout in seconds (default: 60)
        """
        # ---- Logic for plain text calls (such as for Centaur model)

        if self.provider.lower() == "unsloth":
            prompt = (
                self.flatten_messages(messages)
                if isinstance(messages, list)
                else messages
            )
            response_text = self._call_unsloth_API(
                prompt, max_tokens, temperature, timeout
            )
            return prompt, response_text, None

        # ---- Logic for API calls
        if not retry:
            # Original behavior - no retry
            return self._make_single_request(
                messages,
                max_tokens,
                reasoning_max_tokens,
                temperature,
                timeout,
                max_price,
            )

        # Retry logic
        last_error = None

        for attempt in range(max_retries):
            try:
                result = self._make_single_request(
                    messages,
                    max_tokens,
                    reasoning_max_tokens,
                    temperature,
                    timeout,
                    max_price,
                )
                messages_sent, response_text, usage, reasoning = result

                # Check if response is valid (not an error)
                if (
                    response_text
                    and not response_text.startswith("Error")
                    and len(response_text.strip()) > 0
                ):
                    return result

                # Handle rate limit errors specifically
                if "rate_limit_exceeded" in response_text:
                    wait_time = self._parse_rate_limit_error(response_text)

                    if wait_time and attempt < max_retries - 1:
                        print(
                            f"Rate limit hit. Waiting {wait_time:.2f}s as suggested by API..."
                        )
                        time.sleep(wait_time + 0.1)  # Add small buffer
                        continue
                    else:
                        # Use default backoff if we can't parse wait time
                        wait_time = backoff_factor**attempt
                        print(f"Rate limit hit. Using default backoff: {wait_time}s")
                        time.sleep(wait_time)
                        continue

                # Handle context length errors
                elif "Context length exceeded" in response_text:
                    if attempt < max_retries - 1:
                        # Truncate messages and retry
                        print(f"Context too long. Truncating messages and retrying...")
                        messages = self._truncate_messages(
                            messages, reduction_factor=0.75
                        )
                        continue
                    else:
                        last_error = f"Context length exceeded after truncation attempts: {response_text}"
                        break

                # If it's an error response, treat as failed attempt
                last_error = f"API returned error response: {response_text}"

            except (
                ChunkedEncodingError,
                ConnectionError,
                Timeout,
                requests.exceptions.RequestException,
                Exception,
            ) as e:
                last_error = f"Error: {str(e)}"

            # Wait before retry (exponential backoff)
            if attempt < max_retries - 1:
                wait_time = backoff_factor**attempt
                print(f"Attempt {attempt + 1} failed: {last_error}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        # All retries failed
        print(f"All {max_retries} attempts failed. Last error: {last_error}")
        return messages, f"Error: Max retries exceeded. Last error: {last_error}", None

    def _make_single_request(
        self,
        messages,
        max_tokens,
        reasoning_max_tokens,
        temperature,
        timeout,
        max_price=None,
    ):
        """Make a single request without retry logic"""

        if self.provider.lower() in ["groq", "openai", "openrouter"]:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if (self.model.lower() == "openai/gpt-5") or (
                self.model.lower() == "openai/gpt-5-mini"
            ):
                payload["reasoning"] = {
                    "effort": "low",
                    "exclude": False,
                }
                payload["max_tokens"] = (
                    reasoning_max_tokens  # this is because we need to leave some room for the reasoning tokens
                )

            if max_price is not None:
                provider_prefs = {}
                provider_prefs["max_price"] = {"completion": max_price}
                payload["providers"] = provider_prefs
            # Add timeout to the request
            response = requests.post(
                self.endpoint, headers=headers, json=payload, timeout=timeout
            )

            if response.status_code == 200:
                data = response.json()
                # import pdb

                # pdb.set_trace()
                if "error" in data:
                    error_info = data["error"]
                    error_msg = f"Error {error_info.get('code', '')}: {error_info.get('message', '')}"
                    return messages, error_msg, None, None
                return (
                    messages,
                    data["choices"][0]["message"]["content"],
                    data.get("usage"),
                    data["choices"][0]["message"].get("reasoning", None),
                )
            else:
                return (
                    messages,
                    f"Error {response.status_code}: {response.text}",
                    None,
                    None,
                )
        else:
            raise NotImplementedError(f"Provider {self.provider} not supported yet.")

    def _parse_rate_limit_error(self, response_text):
        """Extract wait time from rate limit error message"""
        try:
            # Extract JSON from error message
            if "Error 429:" in response_text:
                json_start = response_text.find("{")
                if json_start != -1:
                    json_str = response_text[json_start:]
                    error_data = json.loads(json_str)
                    message = error_data.get("error", {}).get("message", "")

                    # Extract wait time (e.g., "907.7ms", "2.5s")
                    wait_match = re.search(r"try again in ([\d.]+)(ms|s)", message)
                    if wait_match:
                        time_val = float(wait_match.group(1))
                        unit = wait_match.group(2)

                        if unit == "ms":
                            return time_val / 1000  # Convert to seconds
                        else:
                            return time_val
        except:
            pass

        return None

    def _truncate_messages(self, messages, reduction_factor=0.75):
        """
        Truncate messages by removing user/assistant pairs from conversation history
        Keeps the system message and removes pairs from the middle of the conversation
        """
        if len(messages) <= 1:
            return messages

        # Always keep the first message (system prompt)
        truncated_messages = [messages[0]]

        # Calculate how many pairs to remove
        remaining_messages = messages[1:]  # Skip system message

        # Messages should come in user/assistant pairs
        num_pairs = len(remaining_messages) // 2
        pairs_to_keep = max(
            1, int(num_pairs * reduction_factor)
        )  # Keep at least 1 pair
        pairs_to_remove = num_pairs - pairs_to_keep

        print(
            f"🔄 Truncating conversation: keeping {pairs_to_keep} pairs, removing {pairs_to_remove} pairs"
        )

        if pairs_to_remove > 0:
            # Remove pairs from the beginning of conversation (keep recent context)
            start_idx = 1 + (pairs_to_remove * 2)  # Skip system + removed pairs
            truncated_messages.extend(remaining_messages[start_idx:])
        else:
            # Keep all remaining messages
            truncated_messages.extend(remaining_messages)

        return truncated_messages

    def flatten_messages(self, messages):
        lines = []
        for m in messages:
            role = m["role"].capitalize()
            lines.append(f"{role}: {m['content']}")
        return "\n".join(lines) + "\nAssistant:"

    def _call_unsloth_API(self, messages, max_tokens, temperature, timeout):
        if isinstance(messages, list):
            prompt = self.flatten_messages(messages)
        else:
            prompt = messages  # already a string

        output = self.pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
        )

        generated = output[0]["generated_text"]
        if generated.startswith(prompt):
            generated = generated[len(prompt) :]
        return generated.strip()

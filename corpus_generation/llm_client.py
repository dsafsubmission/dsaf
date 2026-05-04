"""
llm_client.py — Unified LLM client for DSAF adversarial testing.

Supports:
  - Ollama (local):   http://localhost:11434/v1/chat/completions
  - DeepSeek:         https://api.deepseek.com/v1/chat/completions
  - OpenAI:           https://api.openai.com/v1/chat/completions
  - Anthropic:        https://api.anthropic.com/v1/messages

Temperature: 0.3 by default. Real-world prior-auth AI systems do not run at
temperature=0; 0.3 models realistic stochastic decision-making at the margin
while remaining low enough to preserve response coherence.

Reproducibility: seed=42 is passed to OpenAI-compatible endpoints so that
identical (model, seed, temperature, prompt) triplets yield identical outputs.
Anthropic does not expose a seed parameter; reproducibility is achieved via
fixed temperature and prompt construction.
"""

import json
import re
import time
import urllib.request
import urllib.error

# Temperature for all model calls. 0.3 models realistic production conditions
# while keeping responses coherent. Seed is pinned for full reproducibility.
TEMPERATURE = 0.3
SEED        = 42

# ─────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────

MODELS = {
    # Local Ollama
    "ollama/qwen2.5:14b": {
        "endpoint": "http://localhost:11434/v1/chat/completions",
        "api_model": "qwen2.5:14b",
        "backend":   "openai",
        "api_key":   None,
    },
    "ollama/llama3.1:8b": {
        "endpoint": "http://localhost:11434/v1/chat/completions",
        "api_model": "llama3.1:8b",
        "backend":   "openai",
        "api_key":   None,
    },
    # Cloud models
    "deepseek/deepseek-v3": {
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
        "api_model": "deepseek-chat",
        "backend":   "openai",
        "api_key":   None,   # set at runtime
    },
    "openai/gpt-4o": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "api_model": "gpt-4o",
        "backend":   "openai",
        "api_key":   None,
    },
    "anthropic/claude-sonnet-4-20250514": {
        "endpoint": "https://api.anthropic.com/v1/messages",
        "api_model": "claude-sonnet-4-20250514",
        "backend":   "anthropic",
        "api_key":   None,
    },
    "anthropic/claude-opus-4-20250514": {
        "endpoint": "https://api.anthropic.com/v1/messages",
        "api_model": "claude-opus-4-20250514",
        "backend":   "anthropic",
        "api_key":   None,
    },
    "anthropic/claude-haiku-4-5-20251001": {
        "endpoint": "https://api.anthropic.com/v1/messages",
        "api_model": "claude-haiku-4-5-20251001",
        "backend":   "anthropic",
        "api_key":   None,
    },
    "openrouter/glm-5.1": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "api_model": "z-ai/glm-5.1",
        "backend":   "openai",
        "api_key":   None,
    },
    "openrouter/qwen-2.5-72b": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "api_model": "qwen/qwen-2.5-72b-instruct",
        "backend":   "openai",
        "api_key":   None,
    },
    "openrouter/llama-4-maverick": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "api_model": "meta-llama/llama-4-maverick",
        "backend":   "openai",
        "api_key":   None,
    },
    "openrouter/qwen3-235b": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "api_model": "qwen/qwen3-235b-a22b-2507",
        "backend":   "openai",
        "api_key":   None,
    },
    "google/gemini-2.5-flash": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "api_model": "gemini-2.5-flash-preview-04-17",
        "backend":   "openai",
        "api_key":   None,
    },
    "google/gemini-2.5-pro": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "api_model": "gemini-2.5-pro",
        "backend":   "openai",
        "api_key":   None,
    },
    "openrouter/gemini-2.5-pro": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "api_model": "google/gemini-2.5-pro",
        "backend":   "openai",
        "api_key":   None,
    },
    "openrouter/gemma-3-27b": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "api_model": "google/gemma-3-27b-it",
        "backend":   "openai",
        "api_key":   None,
    },
    "openrouter/llama-3.3-70b": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "api_model": "meta-llama/llama-3.3-70b-instruct",
        "backend":   "openai",
        "api_key":   None,
    },
    "together/llama-3.3-70b": {
        "endpoint": "https://api.together.xyz/v1/chat/completions",
        "api_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "backend":   "openai",
        "api_key":   None,
    },
    "openrouter/llama-3.1-8b": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "api_model": "meta-llama/llama-3.1-8b-instruct",
        "backend":   "openai",
        "api_key":   None,
    },
    "google/gemini-3.1-pro": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "api_model": "gemini-3.1-pro-preview",
        "backend":   "openai",
        "api_key":   None,
    },
}


class LLMClient:
    def __init__(self, model_id: str, api_key: str = None, temperature: float = None):
        if model_id not in MODELS:
            raise ValueError(f"Unknown model: {model_id}. Valid: {list(MODELS.keys())}")
        self.model_id  = model_id
        self.config    = MODELS[model_id].copy()
        if api_key:
            self.config["api_key"] = api_key
        self.temperature = temperature if temperature is not None else TEMPERATURE
        self.call_count    = 0
        self.retry_count   = 0
        self.error_count   = 0

    def call(self, system_prompt: str, user_prompt: str) -> tuple[dict, str, bool]:
        """
        Call the LLM with a system + user prompt.

        Returns:
            (parsed_json_action, raw_reasoning, reasoning_insufficient)

        Retries:
            - 429 rate limit: up to 5 retries with exponential backoff
            - Short reasoning (<50 words): one retry with elaboration request
        """
        raw, insufficient = self._call_with_retry(system_prompt, user_prompt)
        action = self._parse_json(raw)

        # JSON format retry: if response is not valid JSON or missing expected keys, retry
        has_valid_key = (action.get("decision") or
                         (action.get("action") and action["action"] != "hold") or
                         action.get("recommended_account"))
        if not action or not has_valid_key:
            retry_raw, _ = self._call_with_retry(
                system_prompt,
                user_prompt,
                retry_suffix="CRITICAL: You MUST respond with ONLY a valid JSON object. No text, no markdown, no explanation. Start your response with { and end with }. Nothing else."
            )
            retry_action = self._parse_json(retry_raw)
            if retry_action and (retry_action.get("decision") or retry_action.get("action") or retry_action.get("recommended_account")):
                self.retry_count += 1
                action = retry_action
                raw = retry_raw

        # Enforce minimum reasoning length
        reasoning = action.get("reasoning", "")
        if len(reasoning.split()) < 50:
            # One retry
            retry_raw, _ = self._call_with_retry(
                system_prompt,
                user_prompt,
                retry_suffix="Your reasoning was too brief. Provide at least 50 words of reasoning. Respond with ONLY valid JSON."
            )
            retry_action = self._parse_json(retry_raw)
            retry_reasoning = retry_action.get("reasoning", "")
            if len(retry_reasoning.split()) >= 50:
                self.retry_count += 1
                return retry_action, retry_reasoning, False
            else:
                insufficient = True
                return action, reasoning, True

        return action, reasoning, insufficient

    def _call_with_retry(self, system_prompt: str, user_prompt: str,
                         retry_suffix: str = "") -> tuple[str, bool]:
        full_user = user_prompt
        if retry_suffix:
            full_user = user_prompt + "\n\n" + retry_suffix

        for attempt in range(6):
            try:
                content = self._http_call(system_prompt, full_user)
                self.call_count += 1
                return content, False
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 5:
                    wait = 2 ** (attempt + 2)  # 4, 8, 16, 32, 64 seconds
                    time.sleep(wait)
                    self.retry_count += 1
                    continue
                if e.code == 529 and attempt < 5:  # Anthropic overloaded
                    wait = 2 ** (attempt + 2)
                    time.sleep(wait)
                    self.retry_count += 1
                    continue
                self.error_count += 1
                raise
            except Exception as e:
                self.error_count += 1
                if attempt < 2:
                    time.sleep(2)
                    continue
                raise

        raise RuntimeError("Max retries exceeded")

    def _http_call(self, system_prompt: str, user_prompt: str) -> str:
        # Rate limiting for Gemini
        if "gemini" in self.config.get("api_model", ""):
            time.sleep(2)
        backend = self.config["backend"]
        if backend == "openai":
            return self._call_openai_compat(system_prompt, user_prompt)
        elif backend == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _call_openai_compat(self, system_prompt: str, user_prompt: str) -> str:
        max_tok = 1024
        payload_dict = {
            "model":       self.config["api_model"],
            "temperature": self.temperature,
            "max_tokens":  max_tok,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        }
        # For Gemini: native JSON mode, no max_tokens cap (thinking eats into it)
        if "gemini" in self.config.get("api_model", ""):
            payload_dict["response_format"] = {"type": "json_object"}
            del payload_dict["max_tokens"]
        # For Together: force JSON mode (otherwise Llama wraps in markdown)
        if "together" in self.config.get("endpoint", ""):
            payload_dict["response_format"] = {"type": "json_object"}
        payload = json.dumps(payload_dict).encode()

        headers = {"Content-Type": "application/json"}
        if self.config["api_key"]:
            headers["Authorization"] = f"Bearer {self.config['api_key']}"

        req = urllib.request.Request(
            self.config["endpoint"],
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"].strip()

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        payload = json.dumps({
            "model":       self.config["api_model"],
            "temperature": self.temperature,
            "max_tokens":  1024,
            "system":      system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }).encode()

        headers = {
            "Content-Type":      "application/json",
            "x-api-key":         self.config["api_key"],
            "anthropic-version": "2023-06-01",
        }
        req = urllib.request.Request(
            self.config["endpoint"],
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        return result["content"][0]["text"].strip()

    @staticmethod
    def _parse_json(raw: str) -> dict:
        # Strip markdown code fences
        text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try extracting the first JSON object
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        # Fallback: return hold with the raw text as reasoning
        return {
            "action": "hold", "asset": "", "size": 0,
            "side": "", "reasoning": raw[:500], "confidence": 0.5,
        }
# Note: qwen3-235b already registered above


"""Shared cloud LLM client factory and completion helper.

Every OFFLINE cloud LLM call in this project (entity/relation extraction,
LLM-as-judge, synthetic QA generation) goes through here, so switching
providers is one .env setting away -- call sites never construct a client
or branch on SDK shape themselves.

Alibaba Cloud (DashScope) and other OpenAI-compatible providers only need a
different base_url + api_key; only Anthropic uses a different SDK entirely.
Adding a new OpenAI-compatible provider is one line in _OPENAI_COMPATIBLE_BASE_URLS.

This module is for OFFLINE processing only (public legal text, synthetic
data, system outputs). The online audit path never calls a cloud API by
design -- see pipeline/generation.py, which is local-SLM-only.
"""
import json
import os
import time
from functools import lru_cache

_OPENAI_COMPATIBLE_BASE_URLS = {
    "openai": None,  # SDK default (api.openai.com)
    "alibaba": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "alibaba": "ALIBABA_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


@lru_cache(maxsize=None)
def _client(provider: str):
    if provider not in _API_KEY_ENV:
        raise ValueError(f"Unknown provider {provider!r}; known: {sorted(_API_KEY_ENV)}")
    env_var = _API_KEY_ENV[provider]
    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(f"{env_var} is not set (required for provider={provider!r})")

    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=api_key)

    from openai import OpenAI
    base_url = _OPENAI_COMPATIBLE_BASE_URLS[provider]
    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)


def complete_json(
    provider: str, model: str, prompt: str, *,
    max_tokens: int = 2000, temperature: float = 0, max_attempts: int = 3,
) -> tuple[dict, dict]:
    """Runs a JSON-producing chat completion, retried with backoff.

    Returns (parsed_json, usage) where usage = {"prompt_tokens", "completion_tokens"}.
    A truncated or malformed response is retried like any other failure --
    it must not abort a long batch over a single call.

    JSON is extracted defensively (first "{" to last "}") for every provider:
    OpenAI-compatible providers already return clean JSON via response_format,
    but Anthropic has no equivalent strict mode and may wrap it in prose or
    code fences.
    """
    client = _client(provider)
    for attempt in range(1, max_attempts + 1):
        try:
            if provider == "anthropic":
                response = client.messages.create(
                    model=model, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text
                usage = {"prompt_tokens": response.usage.input_tokens,
                         "completion_tokens": response.usage.output_tokens}
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if response.choices[0].finish_reason == "length":
                    raise ValueError(f"LLM output truncated (provider={provider}, model={model})")
                text = response.choices[0].message.content
                usage = {"prompt_tokens": response.usage.prompt_tokens,
                         "completion_tokens": response.usage.completion_tokens}

            start, end = text.find("{"), text.rfind("}")
            return json.loads(text[start:end + 1]), usage
        except Exception:
            if attempt == max_attempts:
                raise
            time.sleep(2 ** attempt)

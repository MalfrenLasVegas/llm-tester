from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any


def normalize_base_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    cleaned = re.sub(r"/v1/v1$", "/v1", cleaned)
    return cleaned


def mask_api_key(api_key: str | None) -> str:
    if not api_key:
        return "(sin API key)"
    if len(api_key) <= 8:
        return "****"
    prefix = api_key[:3] if api_key.startswith("sk-") else api_key[:2]
    return f"{prefix}****{api_key[-4:]}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def compact_error(value: Any, max_len: int = 1200) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False)
        except TypeError:
            value = str(value)
    value = value.strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def extract_chat_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    text = choices[0].get("text")
    return text if isinstance(text, str) else ""


def extract_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    choices = response.get("choices") or []
    if not choices:
        return []
    message = choices[0].get("message") or {}
    calls = message.get("tool_calls") or message.get("function_call")
    if isinstance(calls, list):
        return [c for c in calls if isinstance(c, dict)]
    if isinstance(calls, dict):
        return [calls]
    return []


def parse_json_from_text(text: str) -> tuple[bool, Any | None, str | None]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return True, json.loads(stripped), None
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match:
            try:
                return True, json.loads(match.group(0)), None
            except json.JSONDecodeError:
                pass
        return False, None, str(exc)


def looks_uppercase(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return all(not c.islower() for c in letters)


def infer_embedding_model(models: list[str]) -> str | None:
    lowered = [(m, m.lower()) for m in models]
    preferred_words = ("embedding", "embed", "bge", "e5", "text-embedding")
    for original, low in lowered:
        if any(word in low for word in preferred_words):
            return original
    return None


def detect_provider(
    base_url: str,
    headers: dict[str, str] | None = None,
    error_text: str | None = None,
    model_ids: list[str] | None = None,
) -> str:
    haystack = " ".join(
        [
            base_url.lower(),
            " ".join(f"{k}:{v}" for k, v in (headers or {}).items()).lower(),
            (error_text or "").lower(),
            " ".join(model_ids or []).lower(),
        ]
    )
    checks = [
        ("OpenAI oficial", ["api.openai.com", "openai-organization", "openai-processing-ms"]),
        ("Ollama OpenAI-compatible", ["localhost:11434", "ollama"]),
        ("llama.cpp server", ["llama.cpp", "llamacpp", "slot_id"]),
        ("LM Studio", ["localhost:1234", "lmstudio", "lm studio"]),
        ("vLLM", ["vllm"]),
        ("OpenRouter", ["openrouter.ai", "x-ratelimit", "openrouter"]),
        ("LiteLLM", ["litellm"]),
        ("Groq", ["api.groq.com", "groq"]),
        ("Together", ["api.together.xyz", "together"]),
        ("DeepInfra", ["deepinfra.com", "deepinfra"]),
    ]
    for provider, needles in checks:
        if any(needle in haystack for needle in needles):
            return provider
    return "otro compatible"


def approx_token_text(tokens: int) -> str:
    # Texto deliberadamente aburrido: una palabra corta suele ser cerca de un token.
    chunk = "dato "
    return chunk * max(1, tokens)

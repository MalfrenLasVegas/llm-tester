from __future__ import annotations

import time
from typing import Any

from llm_tester.client import LLMAPIError, LLMClient
from llm_tester.models import TestResult


def _delta_text(event: dict[str, Any]) -> str:
    choices = event.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if isinstance(content, str):
        return content
    text = choices[0].get("text")
    return text if isinstance(text, str) else ""


def run_streaming_test(client: LLMClient, model: str) -> TestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Cuenta del 1 al 3, separado por comas."}],
        "temperature": 0,
        "max_tokens": 32,
    }
    started = time.perf_counter()
    first_token_ms: float | None = None
    chunks = 0
    content_parts: list[str] = []
    try:
        for event in client.chat_completion_stream(payload):
            chunks += 1
            piece = _delta_text(event)
            if piece and first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
            if piece:
                content_parts.append(piece)
        total_ms = (time.perf_counter() - started) * 1000
        supported = chunks > 0 and (first_token_ms is not None or total_ms > 0)
        return TestResult(
            name="Streaming",
            supported=supported,
            status="ok" if supported else "partial",
            latency_ms=total_ms,
            details={
                "chunks": chunks,
                "time_to_first_token_ms": first_token_ms,
                "total_ms": total_ms,
                "response_preview": "".join(content_parts)[:300],
            },
        )
    except LLMAPIError as exc:
        return TestResult(
            name="Streaming",
            supported=False,
            status="fail",
            details={"time_to_first_token_ms": first_token_ms},
            raw_error=exc.readable(),
        )

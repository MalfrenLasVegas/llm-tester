from __future__ import annotations

import re

from llm_tester.client import LLMAPIError, LLMClient
from llm_tester.models import TestResult
from llm_tester.utils import extract_chat_text


def _mentions_240(text: str) -> bool:
    return bool(re.search(r"\b240\b", text))


def run_system_prompt_test(client: LLMClient, model: str) -> TestResult:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Responde siempre en mayúsculas"},
            {"role": "user", "content": "hola"},
        ],
        "temperature": 0,
        "max_tokens": 32,
    }
    try:
        response = client.chat_completion(payload)
        text = extract_chat_text(response.data)
        letters = [c for c in text if c.isalpha()]
        ok = bool(letters) and all(not c.islower() for c in letters)
        return TestResult(
            name="System prompt",
            supported=ok,
            status="ok" if ok else "partial",
            latency_ms=response.latency_ms,
            details={"response": text, "expected_behavior": "mayúsculas"},
            usage=response.data.get("usage") if isinstance(response.data, dict) else None,
        )
    except LLMAPIError as exc:
        return TestResult(
            name="System prompt",
            supported=False,
            status="fail",
            raw_error=exc.readable(),
        )


def run_reasoning_test(client: LLMClient, model: str) -> TestResult:
    base_messages = [
        {
            "role": "user",
            "content": (
                "Resuelve este problema paso a paso de forma breve: si tengo 3 servidores "
                "y cada uno procesa 120 peticiones por segundo, pero uno cae, "
                "¿cuántas peticiones por segundo quedan? Termina con 'Respuesta final: N'."
            ),
        }
    ]
    payload = {
        "model": model,
        "messages": base_messages,
        "reasoning_effort": "low",
        "max_completion_tokens": 256,
        "temperature": 0,
    }
    reasoning_effort_accepted: bool | None = None
    raw_error: str | None = None
    response_latency: float | None = None
    text = ""

    try:
        response = client.chat_completion(payload)
        reasoning_effort_accepted = True
        response_latency = response.latency_ms
        text = extract_chat_text(response.data)
        usage = response.data.get("usage") if isinstance(response.data, dict) else None
    except LLMAPIError as exc:
        raw_error = exc.readable()
        lower = raw_error.lower()
        if "reasoning_effort" in lower or "max_completion_tokens" in lower:
            reasoning_effort_accepted = False
            fallback = {
                "model": model,
                "messages": base_messages,
                "max_tokens": 256,
                "temperature": 0,
            }
            try:
                response = client.chat_completion(fallback)
                response_latency = response.latency_ms
                text = extract_chat_text(response.data)
                usage = response.data.get("usage") if isinstance(response.data, dict) else None
            except LLMAPIError as fallback_exc:
                return TestResult(
                    name="Reasoning / thinking",
                    supported=False,
                    status="fail",
                    details={"reasoning_effort_accepted": reasoning_effort_accepted},
                    raw_error=f"{raw_error}\nFallback: {fallback_exc.readable()}",
                )
        else:
            return TestResult(
                name="Reasoning / thinking",
                supported=None,
                status="fail",
                details={"reasoning_effort_accepted": reasoning_effort_accepted},
                raw_error=raw_error,
            )

    correct = _mentions_240(text)
    return TestResult(
        name="Reasoning / thinking",
        supported=correct,
        status="ok" if correct else "partial",
        latency_ms=response_latency,
        details={
            "reasoning_effort_accepted": reasoning_effort_accepted,
            "appears_correct": correct,
            "expected_final_answer": 240,
            "response": text[:800],
            "private_chain_of_thought": "no solicitada ni extraída",
        },
        raw_error=raw_error,
        usage=usage,
    )

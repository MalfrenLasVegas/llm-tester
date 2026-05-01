from __future__ import annotations

from typing import Any

from llm_tester.client import LLMAPIError, LLMClient
from llm_tester.models import TestResult
from llm_tester.utils import compact_error, extract_chat_text


def run_connectivity_test(client: LLMClient) -> TestResult:
    try:
        response = client.list_models()
        data = response.data
        models = []
        if isinstance(data, dict):
            for item in data.get("data", []):
                if isinstance(item, dict) and item.get("id"):
                    models.append(str(item["id"]))
        return TestResult(
            name="Conectividad y /models",
            supported=True,
            status="ok",
            latency_ms=response.latency_ms,
            details={
                "models_endpoint": "available",
                "model_count": len(models),
                "models": models[:50],
                "status_code": response.status_code,
            },
        )
    except LLMAPIError as exc:
        status = "partial" if exc.status_code in {400, 401, 403, 404, 405} else "fail"
        return TestResult(
            name="Conectividad y /models",
            supported=None,
            status=status,
            details={
                "models_endpoint": "unavailable",
                "status_code": exc.status_code,
                "hint": "Puede seguir funcionando si se indica --model manualmente.",
            },
            raw_error=exc.readable(),
        )


def run_basic_chat_test(client: LLMClient, model: str) -> TestResult:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Responde únicamente con OK"}],
        "temperature": 0,
        "max_tokens": 16,
    }
    try:
        response = client.chat_completion(payload)
        text = extract_chat_text(response.data)
        ok = text.strip().upper().strip(".! ") == "OK"
        return TestResult(
            name="Chat básico",
            supported=ok,
            status="ok" if ok else "partial",
            latency_ms=response.latency_ms,
            details={"response": text, "expected": "OK"},
            usage=response.data.get("usage") if isinstance(response.data, dict) else None,
        )
    except LLMAPIError as exc:
        return TestResult(
            name="Chat básico",
            supported=False,
            status="fail",
            details={"model": model},
            raw_error=exc.readable(),
        )


def available_models_from_result(result: TestResult) -> list[str]:
    models = result.details.get("models")
    if isinstance(models, list):
        return [str(m) for m in models]
    return []


def summarize_error_for_detection(results: list[TestResult]) -> str:
    return "\n".join(compact_error(r.raw_error) for r in results if r.raw_error)

from __future__ import annotations

from llm_tester.client import LLMAPIError, LLMClient
from llm_tester.models import TestResult
from llm_tester.utils import approx_token_text, extract_chat_text


def run_context_test(client: LLMClient, model: str, sizes: list[int]) -> TestResult:
    passed: list[int] = []
    failures: list[dict[str, object]] = []
    total_latency = 0.0
    attempts = 0
    for size in sizes:
        prompt = (
            f"Lee el siguiente bloque de unas {size} tokens aproximados. "
            "Responde solo con OK-CONTEXT.\n\n"
            f"{approx_token_text(size)}"
        )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 32,
        }
        attempts += 1
        try:
            response = client.chat_completion(payload)
            total_latency += response.latency_ms
            text = extract_chat_text(response.data)
            if "OK" in text.upper():
                passed.append(size)
            else:
                failures.append({"size": size, "error": "Respuesta inesperada", "response": text[:200]})
                break
        except LLMAPIError as exc:
            failures.append({"size": size, "error": exc.readable()})
            break
    max_passed = max(passed) if passed else None
    supported = bool(passed)
    return TestResult(
        name="Contexto aproximado",
        supported=supported,
        status="ok" if len(passed) == len(sizes) else ("partial" if passed else "fail"),
        latency_ms=(total_latency / attempts) if attempts else None,
        details={
            "tested_sizes_approx_tokens": sizes,
            "passed_sizes_approx_tokens": passed,
            "max_passed_approx_tokens": max_passed,
            "failures": failures,
        },
    )

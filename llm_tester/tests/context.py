from __future__ import annotations

from collections.abc import Callable

from llm_tester.client import LLMAPIError, LLMClient
from llm_tester.models import TestResult
from llm_tester.utils import approx_token_text, extract_chat_text


ContextProgress = Callable[[dict[str, object]], None]


def run_context_test(
    client: LLMClient,
    model: str,
    sizes: list[int],
    *,
    progress: ContextProgress | None = None,
) -> TestResult:
    passed: list[int] = []
    failures: list[dict[str, object]] = []
    total_latency = 0.0
    responses_with_latency = 0
    total_sizes = len(sizes)
    for index, size in enumerate(sizes, start=1):
        if progress:
            progress({"event": "start", "size": size, "index": index, "total": total_sizes})
        sentinel = f"CTX-SENTINEL-{size}-END"
        prompt = (
            f"Lee el siguiente bloque de unas {size} tokens aproximados. "
            "Al final del bloque hay una marca. Responde unicamente con esa marca final, "
            "sin texto extra.\n\n"
            f"{approx_token_text(size)}\n\n"
            f"{sentinel}"
        )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 32,
        }
        try:
            response = client.chat_completion(payload)
            total_latency += response.latency_ms
            responses_with_latency += 1
            text = extract_chat_text(response.data)
            if sentinel in text.upper():
                passed.append(size)
                if progress:
                    progress({"event": "pass", "size": size, "latency_ms": response.latency_ms})
            else:
                failure = {
                    "size": size,
                    "error": "No recupero la marca final esperada",
                    "expected": sentinel,
                    "response": text[:200],
                }
                failures.append(failure)
                if progress:
                    progress(
                        {
                            "event": "fail",
                            "size": size,
                            "latency_ms": response.latency_ms,
                            "error": failure["error"],
                        }
                    )
                break
        except LLMAPIError as exc:
            error = exc.readable()
            failures.append({"size": size, "error": error})
            if progress:
                progress({"event": "fail", "size": size, "error": error})
            break
    max_passed = max(passed) if passed else None
    supported = bool(passed)
    return TestResult(
        name="Contexto aproximado",
        supported=supported,
        status="ok" if len(passed) == len(sizes) else ("partial" if passed else "fail"),
        latency_ms=(total_latency / responses_with_latency) if responses_with_latency else None,
        details={
            "tested_sizes_approx_tokens": sizes,
            "passed_sizes_approx_tokens": passed,
            "max_passed_approx_tokens": max_passed,
            "failures": failures,
        },
    )

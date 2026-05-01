from __future__ import annotations

from llm_tester.client import LLMAPIError, LLMClient
from llm_tester.models import TestResult


def run_embeddings_test(client: LLMClient, embedding_model: str | None) -> TestResult:
    if not embedding_model:
        return TestResult(
            name="Embeddings",
            supported=None,
            status="skipped",
            details={"reason": "No se proporcionó ni detectó modelo de embeddings."},
        )
    payload = {"model": embedding_model, "input": "Texto simple para probar embeddings."}
    try:
        response = client.embeddings(payload)
        data = response.data.get("data", []) if isinstance(response.data, dict) else []
        vector = data[0].get("embedding") if data and isinstance(data[0], dict) else None
        dimension = len(vector) if isinstance(vector, list) else None
        ok = isinstance(vector, list) and bool(vector)
        return TestResult(
            name="Embeddings",
            supported=ok,
            status="ok" if ok else "partial",
            latency_ms=response.latency_ms,
            details={"model": embedding_model, "dimension": dimension},
            usage=response.data.get("usage") if isinstance(response.data, dict) else None,
        )
    except LLMAPIError as exc:
        return TestResult(
            name="Embeddings",
            supported=False,
            status="fail",
            details={"model": embedding_model},
            raw_error=exc.readable(),
        )

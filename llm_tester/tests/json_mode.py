from __future__ import annotations

from jsonschema import Draft202012Validator

from llm_tester.client import LLMAPIError, LLMClient
from llm_tester.models import TestResult
from llm_tester.utils import extract_chat_text, parse_json_from_text


def run_json_mode_test(client: LLMClient, model: str) -> TestResult:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Eres un generador estricto de JSON. Devuelve solo un objeto JSON válido.",
            },
            {
                "role": "user",
                "content": (
                    'Devuelve exactamente este tipo de objeto: {"status":"ok","model":"tested"}. '
                    "Cambia el valor de model por el nombre del modelo si lo conoces. Sin texto extra."
                ),
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": 128,
    }
    try:
        response = client.chat_completion(payload)
        text = extract_chat_text(response.data)
        ok, parsed, error = parse_json_from_text(text)
        has_keys = isinstance(parsed, dict) and {"status", "model"}.issubset(parsed)
        return TestResult(
            name="JSON mode",
            supported=ok and has_keys,
            status="ok" if ok and has_keys else "partial",
            latency_ms=response.latency_ms,
            details={"response": text[:500], "parsed": parsed, "parse_error": error},
            usage=response.data.get("usage") if isinstance(response.data, dict) else None,
        )
    except LLMAPIError as exc:
        return TestResult(
            name="JSON mode",
            supported=False,
            status="fail",
            details={"response_format": {"type": "json_object"}},
            raw_error=exc.readable(),
        )


def run_structured_outputs_test(client: LLMClient, model: str) -> TestResult:
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "score": {"type": "number"},
            "capabilities": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["status", "score", "capabilities"],
        "additionalProperties": False,
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Evalúa esta API ficticia y devuelve el objeto solicitado.",
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "capability_report", "strict": True, "schema": schema},
        },
        "temperature": 0,
        "max_tokens": 256,
    }
    try:
        response = client.chat_completion(payload)
        text = extract_chat_text(response.data)
        ok, parsed, error = parse_json_from_text(text)
        validation_errors: list[str] = []
        if ok:
            validation_errors = [e.message for e in Draft202012Validator(schema).iter_errors(parsed)]
        valid = ok and not validation_errors
        return TestResult(
            name="Structured outputs",
            supported=valid,
            status="ok" if valid else "partial",
            latency_ms=response.latency_ms,
            details={
                "response": text[:500],
                "parsed": parsed,
                "parse_error": error,
                "schema_errors": validation_errors,
            },
            usage=response.data.get("usage") if isinstance(response.data, dict) else None,
        )
    except LLMAPIError as exc:
        return TestResult(
            name="Structured outputs",
            supported=False,
            status="fail",
            details={"response_format": "json_schema"},
            raw_error=exc.readable(),
        )

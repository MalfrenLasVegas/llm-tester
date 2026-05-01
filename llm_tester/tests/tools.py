from __future__ import annotations

from llm_tester.client import LLMAPIError, LLMClient
from llm_tester.models import TestResult
from llm_tester.utils import extract_chat_text, extract_tool_calls


def run_tools_test(client: LLMClient, model: str) -> TestResult:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Usa la herramienta get_server_status para comprobar el servidor "
                    "web-01. No respondas de memoria."
                ),
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_server_status",
                    "description": "Devuelve el estado ficticio de un servidor.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "server_name": {"type": "string"},
                            "include_metrics": {"type": "boolean"},
                        },
                        "required": ["server_name"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 128,
    }
    try:
        response = client.chat_completion(payload)
        calls = extract_tool_calls(response.data)
        text = extract_chat_text(response.data)
        supported = bool(calls)
        doubtful = not calls and "get_server_status" in text
        return TestResult(
            name="Tools/function calling",
            supported=True if supported else (None if doubtful else False),
            status="ok" if supported else ("partial" if doubtful else "fail"),
            latency_ms=response.latency_ms,
            details={"tool_calls": calls, "response": text[:500]},
            usage=response.data.get("usage") if isinstance(response.data, dict) else None,
        )
    except LLMAPIError as exc:
        raw = exc.readable()
        unsupported_markers = [
            "tools",
            "tool_choice",
            "function",
            "unsupported",
            "extra_forbidden",
            "unrecognized",
        ]
        likely_unsupported = any(marker in raw.lower() for marker in unsupported_markers)
        return TestResult(
            name="Tools/function calling",
            supported=False if likely_unsupported else None,
            status="fail",
            details={"classification": "no soportado" if likely_unsupported else "dudoso"},
            raw_error=raw,
        )

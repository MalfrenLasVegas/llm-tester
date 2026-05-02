from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from llm_tester.client import LLMAPIError, LLMClient
from llm_tester.models import TestResult
from llm_tester.utils import extract_chat_text


FeatureProgress = Callable[[dict[str, object]], None]


BASE_PROBES: list[dict[str, Any]] = [
    {
        "id": "max_tokens",
        "category": "generation_limit",
        "description": "OpenAI-style max_tokens output cap.",
        "payload": {"max_tokens": 8},
    },
    {
        "id": "max_completion_tokens",
        "category": "generation_limit",
        "description": "OpenAI reasoning-model output cap.",
        "payload": {"max_completion_tokens": 8},
    },
    {
        "id": "temperature_zero",
        "category": "sampling",
        "description": "temperature=0 accepted.",
        "payload": {"temperature": 0, "max_tokens": 8},
    },
    {
        "id": "top_p",
        "category": "sampling",
        "description": "top_p accepted.",
        "payload": {"top_p": 1, "max_tokens": 8},
    },
    {
        "id": "stop",
        "category": "generation_control",
        "description": "stop sequence accepted.",
        "payload": {"stop": ["."], "max_tokens": 8},
    },
    {
        "id": "seed",
        "category": "sampling",
        "description": "seed accepted.",
        "payload": {"seed": 1234, "max_tokens": 8},
    },
    {
        "id": "reasoning_effort_low",
        "category": "reasoning",
        "description": "reasoning_effort=low accepted.",
        "payload": {"reasoning_effort": "low", "max_completion_tokens": 32},
    },
    {
        "id": "reasoning_effort_medium",
        "category": "reasoning",
        "description": "reasoning_effort=medium accepted.",
        "payload": {"reasoning_effort": "medium", "max_completion_tokens": 32},
    },
    {
        "id": "reasoning_effort_high",
        "category": "reasoning",
        "description": "reasoning_effort=high accepted.",
        "payload": {"reasoning_effort": "high", "max_completion_tokens": 32},
    },
    {
        "id": "reasoning_object_effort_low",
        "category": "reasoning",
        "description": "OpenRouter-style reasoning.effort accepted.",
        "payload": {"reasoning": {"effort": "low"}, "max_tokens": 32},
    },
    {
        "id": "include_reasoning",
        "category": "reasoning",
        "description": "include_reasoning accepted.",
        "payload": {"include_reasoning": True, "max_tokens": 32},
    },
    {
        "id": "reasoning_exclude_false",
        "category": "reasoning",
        "description": "reasoning.exclude accepted.",
        "payload": {"reasoning": {"exclude": False}, "max_tokens": 32},
    },
    {
        "id": "thinking_disabled",
        "category": "thinking",
        "description": "thinking.type=disabled accepted.",
        "payload": {"thinking": {"type": "disabled"}, "max_tokens": 32},
    },
    {
        "id": "thinking_enabled_budget_32",
        "category": "thinking",
        "description": "thinking.type=enabled with small budget accepted.",
        "payload": {"thinking": {"type": "enabled", "budget_tokens": 32}, "max_tokens": 64},
    },
    {
        "id": "enable_thinking_true",
        "category": "thinking",
        "description": "enable_thinking=true accepted.",
        "payload": {"enable_thinking": True, "max_tokens": 32},
    },
    {
        "id": "enable_thinking_false",
        "category": "thinking",
        "description": "enable_thinking=false accepted.",
        "payload": {"enable_thinking": False, "max_tokens": 32},
    },
    {
        "id": "chat_template_kwargs_enable_thinking",
        "category": "thinking",
        "description": "chat_template_kwargs.enable_thinking accepted.",
        "payload": {"chat_template_kwargs": {"enable_thinking": True}, "max_tokens": 32},
    },
    {
        "id": "thinking_budget",
        "category": "thinking",
        "description": "thinking_budget accepted.",
        "payload": {"thinking_budget": 32, "max_tokens": 32},
    },
    {
        "id": "max_reasoning_tokens",
        "category": "reasoning",
        "description": "max_reasoning_tokens accepted.",
        "payload": {"max_reasoning_tokens": 32, "max_tokens": 32},
    },
]


DEEP_PROBES: list[dict[str, Any]] = [
    {
        "id": "anthropic_thinking_budget_1024",
        "category": "thinking",
        "description": "Anthropic-style extended thinking budget.",
        "payload": {"thinking": {"type": "enabled", "budget_tokens": 1024}, "max_tokens": 1100},
    },
    {
        "id": "reasoning_object_max_tokens_1024",
        "category": "reasoning",
        "description": "OpenRouter-style reasoning.max_tokens budget.",
        "payload": {"reasoning": {"max_tokens": 1024}, "max_tokens": 32},
    },
    {
        "id": "reasoning_object_effort_high",
        "category": "reasoning",
        "description": "OpenRouter-style reasoning.effort=high.",
        "payload": {"reasoning": {"effort": "high"}, "max_tokens": 32},
    },
]


def run_api_feature_detection(
    client: LLMClient,
    model: str,
    *,
    output_token_caps: list[int] | None = None,
    deep: bool = False,
    progress: FeatureProgress | None = None,
) -> TestResult:
    probes = [*BASE_PROBES]
    if output_token_caps:
        probes.extend(_output_cap_probes(output_token_caps))
    if deep:
        probes.extend(DEEP_PROBES)

    probe_results: list[dict[str, Any]] = []
    total_latency = 0.0
    responses_with_latency = 0
    total = len(probes)

    for index, probe in enumerate(probes, start=1):
        if progress:
            progress({"event": "start", "probe": probe["id"], "index": index, "total": total})
        payload = _build_payload(model, probe["payload"])
        try:
            response = client.chat_completion(payload)
            total_latency += response.latency_ms
            responses_with_latency += 1
            text = extract_chat_text(response.data) if isinstance(response.data, dict) else ""
            usage = response.data.get("usage") if isinstance(response.data, dict) else None
            message_keys = _message_keys(response.data)
            result = {
                "id": probe["id"],
                "category": probe["category"],
                "description": probe["description"],
                "status": "accepted",
                "latency_ms": response.latency_ms,
                "response_preview": text[:160],
                "usage": usage,
                "message_keys": message_keys,
                "response_signals": _response_signals(response.data),
            }
            probe_results.append(result)
            if progress:
                progress({"event": "accepted", "probe": probe["id"], "latency_ms": response.latency_ms})
        except LLMAPIError as exc:
            error = exc.readable()
            result = {
                "id": probe["id"],
                "category": probe["category"],
                "description": probe["description"],
                "status": "rejected",
                "error": error,
                "status_code": exc.status_code,
                "likely_parameter_error": _looks_like_parameter_error(error, probe["payload"]),
            }
            probe_results.append(result)
            if progress:
                progress({"event": "rejected", "probe": probe["id"], "error": error})

    accepted = [item["id"] for item in probe_results if item["status"] == "accepted"]
    rejected = [item["id"] for item in probe_results if item["status"] == "rejected"]
    categories = _summarize_categories(probe_results)
    supported = bool(accepted)

    return TestResult(
        name="API feature detection",
        supported=supported,
        status="ok" if accepted and not rejected else ("partial" if accepted else "fail"),
        latency_ms=(total_latency / responses_with_latency) if responses_with_latency else None,
        details={
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "accepted_features": accepted,
            "rejected_features": rejected,
            "categories": categories,
            "probes": probe_results,
            "note": (
                "HTTP acceptance means the endpoint accepted the payload. Some OpenAI-compatible "
                "gateways silently ignore unknown parameters, so semantic support may need manual review."
            ),
        },
        raw_error=None if accepted else _first_rejection_error(probe_results),
    )


def _build_payload(model: str, extra: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Responde solo OK."}],
    }
    payload.update(deepcopy(extra))
    return payload


def _output_cap_probes(caps: list[int]) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for cap in sorted(set(caps)):
        probes.append(
            {
                "id": f"max_tokens_cap_{cap}",
                "category": "generation_limit",
                "description": f"Request max_tokens={cap}.",
                "payload": {"max_tokens": cap},
            }
        )
        probes.append(
            {
                "id": f"max_completion_tokens_cap_{cap}",
                "category": "generation_limit",
                "description": f"Request max_completion_tokens={cap}.",
                "payload": {"max_completion_tokens": cap},
            }
        )
    return probes


def _message_keys(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return []
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return []
    message = choices[0].get("message") or {}
    return sorted(message.keys()) if isinstance(message, dict) else []


def _response_signals(data: Any) -> dict[str, bool]:
    text = repr(data).lower()
    return {
        "has_reasoning_key": "reasoning" in text,
        "has_thinking_key": "thinking" in text,
        "has_reasoning_tokens": "reasoning_tokens" in text,
        "has_reasoning_content": "reasoning_content" in text,
    }


def _looks_like_parameter_error(error: str, payload: dict[str, Any]) -> bool:
    lowered = error.lower()
    if any(word in lowered for word in ("unsupported", "unrecognized", "unknown", "extra_forbidden")):
        return True
    return any(str(key).lower() in lowered for key in payload)


def _summarize_categories(probe_results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for result in probe_results:
        category = str(result["category"])
        status = str(result["status"])
        if category not in summary:
            summary[category] = {"accepted": 0, "rejected": 0}
        summary[category][status] = summary[category].get(status, 0) + 1
    return summary


def _first_rejection_error(probe_results: list[dict[str, Any]]) -> str | None:
    for result in probe_results:
        if result.get("status") == "rejected" and result.get("error"):
            return str(result["error"])
    return None

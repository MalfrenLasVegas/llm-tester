from __future__ import annotations

import hashlib
import json
from typing import Any

from llm_tester.client import LLMAPIError, LLMClient
from llm_tester.models import TestResult
from llm_tester.tests.context import ContextProgress
from llm_tester.utils import approx_token_text, compact_error, extract_chat_text

WEIGHTS = {
    "retrieval": 0.50,
    "instruction": 0.25,
    "reasoning": 0.20,
    "json_parse": 0.05,
}
POSITION_KEYS = ["K05", "K25", "K50", "K75", "K95"]
POSITION_MARKERS = {
    5: "K05",
    15: "BASE_VALUE",
    25: "K25",
    50: "K50",
    55: "MULTIPLIER",
    75: "K75",
    85: "OFFSET",
    95: "K95",
}
DISCLAIMER = (
    "No se puede confirmar la cuantización del KV Cache desde esta API. Estos resultados solo miden "
    "calidad efectiva del contexto y síntomas compatibles con degradación, truncado o configuración "
    "agresiva del backend."
)


def run_context_quality_test(
    client: LLMClient,
    model: str,
    sizes: list[int],
    *,
    runs: int = 3,
    progress: ContextProgress | None = None,
) -> TestResult:
    size_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_latency = 0.0
    latency_count = 0
    total_work = len(sizes) * runs
    work_index = 0

    for size in sizes:
        run_results: list[dict[str, Any]] = []
        for run_number in range(1, runs + 1):
            work_index += 1
            if progress:
                progress(
                    {
                        "event": "start",
                        "size": size,
                        "run": run_number,
                        "runs": runs,
                        "index": work_index,
                        "total": total_work,
                    }
                )
            expected = _expected_values(size, run_number)
            prompt = _build_prompt(size, run_number, expected)
            try:
                response = _chat_completion_json_fallback(client, model, prompt)
                latency_ms = response.latency_ms
                total_latency += latency_ms
                latency_count += 1
                text = extract_chat_text(response.data)
                run_result = _score_response(run_number, expected, text, latency_ms)
            except LLMAPIError as exc:
                run_result = _failed_run(run_number, expected, exc.readable())
            run_results.append(run_result)
            if run_result.get("error"):
                failures.append({"size": size, "run": run_number, "error": run_result["error"]})

        size_result = _summarize_size(size, run_results)
        size_results.append(size_result)
        if progress:
            progress({"event": "size_complete", **size_result})

    score_sizes = [item for item in size_results if isinstance(item.get("avg_total_score"), int | float)]
    any_usable = any(item["avg_total_score"] >= 0.70 for item in score_sizes)
    all_ok = bool(score_sizes) and all(item["avg_total_score"] >= 0.90 for item in score_sizes)
    all_below_usable = bool(score_sizes) and all(item["avg_total_score"] < 0.70 for item in score_sizes)
    status = "ok" if all_ok else ("fail" if all_below_usable else "partial")

    details = {
        "tested_sizes_approx_tokens": sizes,
        "runs_per_size": runs,
        "score_weights": WEIGHTS,
        "sizes": size_results,
        **_degradation_points(size_results),
        "hypotheses": _hypotheses(size_results),
        "disclaimer": DISCLAIMER,
        "failures": failures,
    }
    raw_error = None
    if failures and not any_usable:
        raw_error = failures[0]["error"]
    return TestResult(
        name="Calidad de contexto largo",
        supported=any_usable if score_sizes else None,
        status=status if score_sizes else "fail",
        latency_ms=(total_latency / latency_count) if latency_count else None,
        details=details,
        raw_error=raw_error,
    )


def _chat_completion_json_fallback(client: LLMClient, model: str, prompt: str):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }
    try:
        return client.chat_completion(payload)
    except LLMAPIError as exc:
        if not _looks_like_unsupported_response_format(exc):
            raise
        payload.pop("response_format", None)
        return client.chat_completion(payload)


def _looks_like_unsupported_response_format(exc: LLMAPIError) -> bool:
    text = f"{exc} {compact_error(exc.body)}".lower()
    return "response_format" in text and any(
        marker in text
        for marker in (
            "unsupported",
            "not support",
            "unknown",
            "unrecognized",
            "extra",
            "forbidden",
            "invalid parameter",
            "json_object",
        )
    )


def _expected_values(size: int, run: int) -> dict[str, Any]:
    words = {
        "K05": "alpha",
        "K25": "bravo",
        "K50": "charlie",
        "K75": "delta",
        "K95": "echo",
    }
    needles = {key: f"{word}-{_stable_number(size, run, key):05d}" for key, word in words.items()}
    color = f"azul-cobalto-{_stable_number(size, run, 'box') % 90 + 10}"
    base = _stable_number(size, run, "base") % 23 + 7
    multiplier = _stable_number(size, run, "multiplier") % 7 + 2
    offset = _stable_number(size, run, "offset") % 11 + 1
    return {
        "needles": needles,
        "box_color": color,
        "base_value": base,
        "multiplier": multiplier,
        "offset": offset,
        "calculation_result": base * multiplier + offset,
    }


def _stable_number(size: int, run: int, label: str) -> int:
    digest = hashlib.sha256(f"context-quality:{size}:{run}:{label}".encode()).hexdigest()
    return int(digest[:10], 16) % 100_000


def _build_prompt(size: int, run: int, expected: dict[str, Any]) -> str:
    header = (
        "INSTRUCCIONES:\n"
        "Vas a leer un contexto largo con ruido controlado. Dentro hay datos importantes.\n"
        "Al final debes responder SOLO con JSON válido, sin markdown y sin explicación.\n\n"
        f"REGLA_CRITICA_RUN_{run}: Si se pregunta por el color de la caja, responde exactamente "
        f'"{expected["box_color"]}".\n\n'
    )
    final_task = (
        "\n\nTAREA FINAL:\n"
        "Recupera las claves, recuerda la regla crítica del color de la caja y calcula "
        "BASE_VALUE * MULTIPLIER + OFFSET.\n"
        "Devuelve exclusivamente este JSON:\n"
        "{\n"
        '  "needles": {\n'
        '    "K05": "...",\n'
        '    "K25": "...",\n'
        '    "K50": "...",\n'
        '    "K75": "...",\n'
        '    "K95": "..."\n'
        "  },\n"
        '  "box_color": "...",\n'
        '  "calculation_result": 0\n'
        "}\n"
    )
    marker_blocks = _marker_blocks(expected)
    overhead_tokens = max(1, len((header + final_task + "\n".join(marker_blocks.values())).split()))
    filler_tokens = max(1, size - overhead_tokens)
    points = [0, *sorted(POSITION_MARKERS), 100]
    parts = [header]
    previous = 0
    for point in points[1:-1]:
        segment_tokens = max(1, round(filler_tokens * (point - previous) / 100))
        parts.append(approx_token_text(segment_tokens))
        parts.append("\n" + marker_blocks[point] + "\n")
        previous = point
    tail_tokens = max(1, round(filler_tokens * (100 - previous) / 100))
    parts.append(approx_token_text(tail_tokens))
    parts.append(final_task)
    return "".join(parts)


def _marker_blocks(expected: dict[str, Any]) -> dict[int, str]:
    needles = expected["needles"]
    return {
        5: f'DATO_IMPORTANTE K05 = "{needles["K05"]}"',
        15: f'PIEZA_DE_CALCULO BASE_VALUE = {expected["base_value"]}',
        25: f'DATO_IMPORTANTE K25 = "{needles["K25"]}"',
        50: f'DATO_IMPORTANTE K50 = "{needles["K50"]}"',
        55: f'PIEZA_DE_CALCULO MULTIPLIER = {expected["multiplier"]}',
        75: f'DATO_IMPORTANTE K75 = "{needles["K75"]}"',
        85: f'PIEZA_DE_CALCULO OFFSET = {expected["offset"]}',
        95: f'DATO_IMPORTANTE K95 = "{needles["K95"]}"',
    }


def _score_response(run: int, expected: dict[str, Any], raw_text: str, latency_ms: float) -> dict[str, Any]:
    parsed_ok, parsed, parse_error = _parse_json_object(raw_text)
    preview = raw_text.strip().replace("\n", " ")[:500]
    if not parsed_ok or not isinstance(parsed, dict):
        return _failed_run(
            run,
            expected,
            f"No se pudo parsear JSON: {parse_error or 'respuesta no es objeto JSON'}",
            latency_ms=latency_ms,
            raw_response_preview=preview,
            json_parse_score=0.0,
        )

    actual_needles = parsed.get("needles") if isinstance(parsed.get("needles"), dict) else {}
    position_scores = {
        key: 1 if str(actual_needles.get(key, "")).strip() == expected["needles"][key] else 0
        for key in POSITION_KEYS
    }
    retrieval_score = sum(position_scores.values()) / len(POSITION_KEYS)
    instruction_score = 1.0 if str(parsed.get("box_color", "")).strip() == expected["box_color"] else 0.0
    reasoning_score = 1.0 if _coerce_int(parsed.get("calculation_result")) == expected["calculation_result"] else 0.0
    json_parse_score = 1.0
    total_score = _total_score(retrieval_score, instruction_score, reasoning_score, json_parse_score)
    return {
        "run": run,
        "total_score": total_score,
        "retrieval_score": retrieval_score,
        "instruction_score": instruction_score,
        "reasoning_score": reasoning_score,
        "json_parse_score": json_parse_score,
        "latency_ms": latency_ms,
        "expected": _public_expected(expected),
        "actual": parsed,
        "position_scores": position_scores,
        "raw_response_preview": preview,
    }


def _failed_run(
    run: int,
    expected: dict[str, Any],
    error: str,
    *,
    latency_ms: float | None = None,
    raw_response_preview: str = "",
    json_parse_score: float = 0.0,
) -> dict[str, Any]:
    return {
        "run": run,
        "total_score": _total_score(0.0, 0.0, 0.0, json_parse_score),
        "retrieval_score": 0.0,
        "instruction_score": 0.0,
        "reasoning_score": 0.0,
        "json_parse_score": json_parse_score,
        "latency_ms": latency_ms,
        "expected": _public_expected(expected),
        "actual": None,
        "position_scores": {key: 0 for key in POSITION_KEYS},
        "raw_response_preview": raw_response_preview,
        "error": error,
    }


def _public_expected(expected: dict[str, Any]) -> dict[str, Any]:
    return {
        "needles": expected["needles"],
        "box_color": expected["box_color"],
        "calculation_result": expected["calculation_result"],
    }


def _parse_json_object(text: str) -> tuple[bool, Any | None, str | None]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return True, json.loads(stripped), None
    except json.JSONDecodeError as exc:
        first_error = str(exc)
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[index:])
            return True, parsed, None
        except json.JSONDecodeError:
            continue
    return False, None, first_error


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _total_score(retrieval: float, instruction: float, reasoning: float, json_parse: float) -> float:
    return round(
        retrieval * WEIGHTS["retrieval"]
        + instruction * WEIGHTS["instruction"]
        + reasoning * WEIGHTS["reasoning"]
        + json_parse * WEIGHTS["json_parse"],
        4,
    )


def _summarize_size(size: int, runs: list[dict[str, Any]]) -> dict[str, Any]:
    averages = {
        "avg_total_score": _avg([run["total_score"] for run in runs]),
        "avg_retrieval_score": _avg([run["retrieval_score"] for run in runs]),
        "avg_instruction_score": _avg([run["instruction_score"] for run in runs]),
        "avg_reasoning_score": _avg([run["reasoning_score"] for run in runs]),
        "avg_json_parse_score": _avg([run["json_parse_score"] for run in runs]),
        "avg_latency_ms": _avg([run["latency_ms"] for run in runs if run.get("latency_ms") is not None]),
    }
    position_score_avg = {
        key: _avg([run["position_scores"][key] for run in runs if "position_scores" in run]) for key in POSITION_KEYS
    }
    return {
        "size": size,
        "classification": _classification(averages["avg_total_score"]),
        **averages,
        "position_score_avg": position_score_avg,
        "successful_runs": sum(1 for run in runs if not run.get("error")),
        "failed_runs": sum(1 for run in runs if run.get("error")),
        "runs": runs,
    }


def _avg(values: list[float | int]) -> float | None:
    if not values:
        return None
    return round(sum(float(value) for value in values) / len(values), 4)


def _classification(score: float | None) -> str:
    if score is None:
        return "failed"
    if score >= 0.90:
        return "ok"
    if score >= 0.70:
        return "warning"
    if score >= 0.40:
        return "degraded"
    return "failed"


def _degradation_points(size_results: list[dict[str, Any]]) -> dict[str, int | None]:
    ordered = sorted(size_results, key=lambda item: item["size"])
    return {
        "first_warning_size": _first_below(ordered, 0.90),
        "first_degraded_size": _first_below(ordered, 0.70),
        "first_failed_size": _first_below(ordered, 0.40),
        "useful_context_estimate": _max_at_least(ordered, 0.90),
        "max_usable_size": _max_at_least(ordered, 0.70),
        "max_tested_size": max((item["size"] for item in ordered), default=None),
    }


def _first_below(items: list[dict[str, Any]], threshold: float) -> int | None:
    for item in items:
        score = item.get("avg_total_score")
        if isinstance(score, int | float) and score < threshold:
            return int(item["size"])
    return None


def _max_at_least(items: list[dict[str, Any]], threshold: float) -> int | None:
    values = [int(item["size"]) for item in items if isinstance(item.get("avg_total_score"), int | float) and item["avg_total_score"] >= threshold]
    return max(values) if values else None


def _hypotheses(size_results: list[dict[str, Any]]) -> list[str]:
    if not size_results:
        return []
    ordered = sorted(size_results, key=lambda item: item["size"])
    hypotheses: list[str] = []
    first = ordered[0]
    last = ordered[-1]
    first_score = float(first.get("avg_total_score") or 0.0)
    last_score = float(last.get("avg_total_score") or 0.0)

    degraded_items = [item for item in ordered if (item.get("avg_total_score") or 0.0) < 0.70]
    if degraded_items:
        for item in degraded_items:
            pos = item.get("position_score_avg", {})
            early = _avg([pos.get("K05", 0.0), pos.get("K25", 0.0)]) or 0.0
            late = _avg([pos.get("K75", 0.0), pos.get("K95", 0.0)]) or 0.0
            if early + 0.30 < late:
                hypotheses.append("possible_sliding_window_or_early_context_loss")
                hypotheses.append("early_tokens_less_reliable_than_recent_tokens")
                break

    if any((item.get("avg_retrieval_score") or 0.0) >= 0.80 and (item.get("avg_reasoning_score") or 0.0) < 0.70 for item in ordered):
        hypotheses.append("reasoning_degrades_before_retrieval")

    for previous, current in zip(ordered, ordered[1:]):
        prev_score = float(previous.get("avg_total_score") or 0.0)
        curr_score = float(current.get("avg_total_score") or 0.0)
        if prev_score >= 0.85 and curr_score < 0.40:
            hypotheses.append("possible_hard_context_limit_or_backend_truncation")
            break

    if any((item.get("avg_json_parse_score") or 0.0) < 0.85 for item in ordered[-max(1, len(ordered) // 3) :]):
        hypotheses.append("format_following_degrades_with_long_context")

    if first_score - last_score >= 0.20 and not any(
        float(previous.get("avg_total_score") or 0.0) - float(current.get("avg_total_score") or 0.0) >= 0.45
        for previous, current in zip(ordered, ordered[1:])
    ):
        hypotheses.append("progressive_attention_degradation")

    if all((item.get("avg_total_score") or 0.0) >= 0.90 for item in ordered):
        hypotheses.append("no_significant_degradation_detected")

    latencies = [item.get("avg_latency_ms") for item in ordered if item.get("avg_latency_ms") is not None]
    if len(latencies) >= 2 and max(latencies) >= max(1.0, min(latencies) * 3):
        hypotheses.append("latency_spike_at_long_context")

    return list(dict.fromkeys(hypotheses))

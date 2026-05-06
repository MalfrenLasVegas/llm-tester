from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import Report, TestResult


def _status_style(result: TestResult) -> str:
    if result.status == "ok":
        return "green"
    if result.status == "partial":
        return "yellow"
    if result.status == "skipped":
        return "cyan"
    return "red"


def _supported_label(value: bool | None) -> str:
    if value is True:
        return "sí"
    if value is False:
        return "no"
    return "dudoso"


def print_report(report: Report, console: Console | None = None) -> None:
    console = console or Console()
    title = Text("LLM Tester - Informe final", style="bold white")
    subtitle = (
        f"Endpoint: {report.endpoint}\n"
        f"Modelo: {report.tested_model or '(no indicado)'}\n"
        f"Fecha/hora UTC: {report.created_at.isoformat()}\n"
        f"Compatibilidad estimada: {report.compatibility_guess}"
    )
    console.print(Panel(subtitle, title=title, border_style="blue"))

    table = Table(show_header=True, header_style="bold")
    table.add_column("Prueba")
    table.add_column("Estado")
    table.add_column("Soportado")
    table.add_column("Latencia")
    table.add_column("Detalles")
    for result in report.results:
        details = _details_summary(result)
        latency = f"{result.latency_ms:.0f} ms" if result.latency_ms is not None else "-"
        table.add_row(
            result.name,
            f"[{_status_style(result)}]{result.status}[/]",
            _supported_label(result.supported),
            latency,
            details,
        )
    console.print(table)

    metrics = Table(show_header=False)
    metrics.add_column("Métrica", style="bold")
    metrics.add_column("Valor")
    avg = report.average_latency_ms
    metrics.add_row("Latencia media", f"{avg:.0f} ms" if avg is not None else "-")
    ttft = _find_detail(report, "Streaming", "time_to_first_token_ms")
    metrics.add_row("Tiempo hasta primer token", f"{ttft:.0f} ms" if isinstance(ttft, float) else "-")
    tokens = _usage_summary(report)
    metrics.add_row("Tokens usados", tokens or "No devueltos")
    console.print(metrics)

    if report.errors:
        console.print(Panel("\n".join(report.errors), title="Errores detectados", border_style="red"))
    if report.recommendations:
        console.print(
            Panel("\n".join(f"- {item}" for item in report.recommendations), title="Recomendaciones")
        )


def save_json_report(report: Report, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return output


def save_markdown_report(report: Report, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LLM Tester - Informe final",
        "",
        f"- Endpoint: `{report.endpoint}`",
        f"- Modelo probado: `{report.tested_model or '(no indicado)'}`",
        f"- Fecha/hora UTC: `{report.created_at.isoformat()}`",
        f"- Compatibilidad estimada: **{report.compatibility_guess}**",
        "",
        "| Prueba | Estado | Soportado | Latencia | Detalles |",
        "|---|---:|---:|---:|---|",
    ]
    for result in report.results:
        latency = f"{result.latency_ms:.0f} ms" if result.latency_ms is not None else "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_md(result.name),
                    result.status,
                    _supported_label(result.supported),
                    latency,
                    _escape_md(_details_summary(result, plain=True)),
                ]
            )
            + " |"
        )
    for result in report.results:
        if result.name == "Calidad de contexto largo":
            lines.extend(_context_quality_markdown(result))

    lines.extend(["", "## Errores detectados", ""])
    if report.errors:
        lines.extend(f"- `{_escape_md(error)}`" for error in report.errors)
    else:
        lines.append("- Ninguno destacado.")
    lines.extend(["", "## Recomendaciones", ""])
    if report.recommendations:
        lines.extend(f"- {item}" for item in report.recommendations)
    else:
        lines.append("- Sin recomendaciones adicionales.")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _context_quality_markdown(result: TestResult) -> list[str]:
    details = result.details
    lines = [
        "",
        "## Context quality / degradation test",
        "",
        details.get("disclaimer", ""),
        "",
        f"- Useful context estimate: `{details.get('useful_context_estimate') or '-'}`",
        f"- Max usable size: `{details.get('max_usable_size') or '-'}`",
        f"- Max tested size: `{details.get('max_tested_size') or '-'}`",
        "",
        "| Tamaño aprox. | Clasificación | Total | Retrieval | Instruction | Reasoning | JSON parse | Latencia media |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in details.get("sizes", []):
        latency = item.get("avg_latency_ms")
        latency_text = f"{latency:.0f} ms" if isinstance(latency, int | float) else "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("size", "-")),
                    _escape_md(str(item.get("classification", "-"))),
                    _format_score(item.get("avg_total_score")),
                    _format_score(item.get("avg_retrieval_score")),
                    _format_score(item.get("avg_instruction_score")),
                    _format_score(item.get("avg_reasoning_score")),
                    _format_score(item.get("avg_json_parse_score")),
                    latency_text,
                ]
            )
            + " |"
        )
    hypotheses = details.get("hypotheses") or []
    lines.extend(["", "### Hipótesis", ""])
    if hypotheses:
        lines.extend(f"- `{_escape_md(str(item))}`" for item in hypotheses)
    else:
        lines.append("- Sin hipótesis destacadas.")
    return lines


def _format_score(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{value:.2f}"
    return "-"


def build_recommendations(report: Report) -> list[str]:
    recs: list[str] = []
    by_name = {r.name: r for r in report.results}
    if by_name.get("Conectividad y /models", TestResult(name="", supported=None, status="skipped")).status != "ok":
        recs.append("Si /models no está disponible, usa --model explícitamente y revisa permisos de la API key.")
    for name in ["Streaming", "Tools/function calling", "JSON mode", "Structured outputs", "Vision"]:
        result = by_name.get(name)
        if result and result.supported is False:
            recs.append(f"{name}: no lo declares como capacidad estable para este endpoint/modelo.")
    context_quality = by_name.get("Calidad de contexto largo")
    if context_quality:
        useful = context_quality.details.get("useful_context_estimate")
        usable = context_quality.details.get("max_usable_size")
        tested = context_quality.details.get("max_tested_size")
        if usable and tested and usable < tested:
            recs.append(
                f"Calidad de contexto: el modelo acepta/procesa hasta ~{tested} tokens en esta prueba, "
                f"pero el contexto usable estimado cae hacia ~{usable}; revisa details.sizes e hipótesis."
            )
        elif useful:
            recs.append(f"Calidad de contexto: no se detectó degradación significativa hasta ~{useful} tokens útiles.")
    context = by_name.get("Contexto aproximado")
    if context and context.details.get("max_passed_approx_tokens"):
        recs.append(
            "Contexto: usa margen por debajo del máximo aproximado detectado; la prueba no sustituye "
            "la ventana oficial del proveedor."
        )
    features = by_name.get("API feature detection")
    if features and features.details.get("rejected_count"):
        recs.append(
            "API features: revisa accepted_features/rejected_features en el JSON; un 200 HTTP puede indicar "
            "aceptacion del gateway, no necesariamente uso semantico del parametro."
        )
    if not recs:
        recs.append("La API parece funcional para los tests ejecutados. Repite con --verbose si necesitas auditar payloads.")
    return recs


def collect_errors(results: list[TestResult]) -> list[str]:
    return [f"{r.name}: {r.raw_error}" for r in results if r.raw_error]


def _details_summary(result: TestResult, *, plain: bool = False) -> str:
    details = result.details
    interesting_keys = [
        "model_count",
        "response",
        "chunks",
        "time_to_first_token_ms",
        "reasoning_effort_accepted",
        "accepted_count",
        "rejected_count",
        "accepted_features",
        "appears_correct",
        "dimension",
        "max_passed_approx_tokens",
        "classification",
        "useful_context_estimate",
        "max_usable_size",
        "max_tested_size",
        "hypotheses",
        "reason",
    ]
    parts: list[str] = []
    for key in interesting_keys:
        if key in details and details[key] is not None:
            value = details[key]
            if isinstance(value, float):
                value = f"{value:.0f}"
            value_text = str(value).replace("\n", " ")
            if len(value_text) > 120:
                value_text = value_text[:117] + "..."
            parts.append(f"{key}={value_text}")
    if not parts and result.raw_error:
        error = result.raw_error.replace("\n", " ")
        parts.append(error[:140])
    summary = "; ".join(parts) if parts else "-"
    return summary if plain else summary


def _find_detail(report: Report, test_name: str, key: str) -> Any:
    for result in report.results:
        if result.name == test_name:
            return result.details.get(key)
    return None


def _usage_summary(report: Report) -> str | None:
    totals: dict[str, int] = {}
    for result in report.results:
        if not result.usage:
            continue
        for key, value in result.usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    if not totals:
        return None
    return json.dumps(totals, ensure_ascii=False)


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")

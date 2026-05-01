from __future__ import annotations

import getpass
import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .client import LLMAPIError, LLMClient
from .models import Report, TestResult
from .reporting import (
    build_recommendations,
    collect_errors,
    print_report,
    save_json_report,
    save_markdown_report,
)
from .tests.chat import (
    available_models_from_result,
    run_basic_chat_test,
    run_connectivity_test,
    summarize_error_for_detection,
)
from .tests.context import run_context_test
from .tests.embeddings import run_embeddings_test
from .tests.json_mode import run_json_mode_test, run_structured_outputs_test
from .tests.reasoning import run_reasoning_test, run_system_prompt_test
from .tests.streaming import run_streaming_test
from .tests.tools import run_tools_test
from .tests.vision import run_vision_test
from .utils import detect_provider, infer_embedding_model, mask_api_key, normalize_base_url, utc_now

app = typer.Typer(
    name="llm-tester",
    help="Testea modelos/API LLM compatibles con OpenAI usando HTTP directo.",
    no_args_is_help=True,
)
console = Console()


BaseUrlOption = Annotated[
    str | None,
    typer.Option("--base-url", help="Endpoint base compatible con OpenAI, por ejemplo https://host/v1."),
]
ApiKeyOption = Annotated[
    str | None,
    typer.Option("--api-key", help="API key. También puede venir de LLM_TESTER_API_KEY."),
]
ModelOption = Annotated[
    str | None,
    typer.Option("--model", help="Modelo de chat a probar."),
]
EmbeddingModelOption = Annotated[
    str | None,
    typer.Option("--embedding-model", help="Modelo de embeddings opcional."),
]
TimeoutOption = Annotated[
    float,
    typer.Option("--timeout", min=1.0, help="Timeout por petición en segundos."),
]
VerboseOption = Annotated[
    bool,
    typer.Option("--verbose", help="Muestra payloads, headers enmascarados y errores detallados."),
]


def _resolve_base_url(base_url: str | None, interactive: bool) -> str:
    value = base_url or os.getenv("LLM_TESTER_BASE_URL")
    if not value and interactive:
        value = typer.prompt("Base URL")
    if not value:
        raise typer.BadParameter("Falta --base-url o LLM_TESTER_BASE_URL.")
    return normalize_base_url(value)


def _resolve_api_key(api_key: str | None, interactive: bool) -> str | None:
    value = api_key or os.getenv("LLM_TESTER_API_KEY")
    if value:
        return value
    if interactive:
        typed = getpass.getpass("API key (vacío si el endpoint no requiere auth): ")
        return typed or None
    return None


def _resolve_model(model: str | None, interactive: bool) -> str | None:
    value = model or os.getenv("LLM_TESTER_MODEL")
    if not value and interactive:
        typed = typer.prompt("Modelo", default="", show_default=False)
        value = typed or None
    return value


def _resolve_embedding_model(value: str | None) -> str | None:
    return value or os.getenv("LLM_TESTER_EMBEDDING_MODEL")


def _select_model(
    requested_model: str | None,
    models: list[str],
    *,
    interactive: bool,
) -> str:
    if requested_model:
        return requested_model
    if models:
        chat_like = [
            item
            for item in models
            if not any(word in item.lower() for word in ("embed", "embedding", "rerank"))
        ]
        selected = chat_like[0] if chat_like else models[0]
        if interactive:
            console.print(f"[cyan]Modelo no indicado. Usando detectado:[/] {selected}")
        return selected
    if interactive:
        value = typer.prompt("No se pudo listar /models. Indica modelo manualmente")
        if value:
            return value
    raise typer.BadParameter("Falta --model y no se pudo detectar ninguno desde /models.")


def _context_sizes(max_test: int, stress: bool = False) -> list[int]:
    base = [4_000, 8_000, 16_000, 32_000]
    if stress:
        base.extend([64_000, 128_000])
    return [size for size in base if size <= max_test]


def _make_skipped(name: str, reason: str) -> TestResult:
    return TestResult(name=name, supported=None, status="skipped", details={"reason": reason})


def _safe_markdown_output_path(path: Path) -> Path:
    if path.suffix.lower() in {".json", ".ndjson"}:
        corrected = path.with_suffix(".md")
        console.print(
            f"[yellow]Aviso:[/] --markdown-output apuntaba a {path}. "
            f"Uso {corrected} para no sobrescribir JSON."
        )
        return corrected
    return path


def _run_suite(
    *,
    base_url: str,
    api_key: str | None,
    model: str | None,
    embedding_model: str | None,
    timeout: float,
    verbose: bool,
    skip_vision: bool,
    skip_context: bool,
    skip_embeddings: bool,
    skip_tools: bool,
    context_max_test: int,
    stress: bool = False,
    quick: bool = False,
    markdown: bool = False,
    json_output: Path | None = None,
    markdown_output: Path | None = None,
    interactive: bool = True,
) -> Report:
    console.print(
        f"[bold]Endpoint:[/] {base_url}  [bold]API key:[/] {mask_api_key(api_key)}  "
        f"[bold]Timeout:[/] {timeout:.0f}s"
    )
    results: list[TestResult] = []
    with LLMClient(base_url, api_key, timeout=timeout, verbose=verbose) as client:
        connectivity = run_connectivity_test(client)
        results.append(connectivity)
        _print_progress(connectivity)
        models = available_models_from_result(connectivity)
        selected_model = _select_model(model, models, interactive=interactive)
        selected_embedding_model = embedding_model or infer_embedding_model(models)

        test_plan = [
            lambda: run_basic_chat_test(client, selected_model),
            lambda: run_streaming_test(client, selected_model),
            lambda: run_system_prompt_test(client, selected_model),
        ]
        if not quick:
            if skip_tools:
                test_plan.append(lambda: _make_skipped("Tools/function calling", "--skip-tools activado."))
            else:
                test_plan.append(lambda: run_tools_test(client, selected_model))
            test_plan.extend(
                [
                    lambda: run_json_mode_test(client, selected_model),
                    lambda: run_structured_outputs_test(client, selected_model),
                ]
            )
            if skip_vision:
                test_plan.append(lambda: _make_skipped("Vision", "--skip-vision activado."))
            else:
                test_plan.append(lambda: run_vision_test(client, selected_model))
            test_plan.append(lambda: run_reasoning_test(client, selected_model))
            if skip_context:
                test_plan.append(lambda: _make_skipped("Contexto aproximado", "--skip-context activado."))
            else:
                sizes = _context_sizes(context_max_test, stress=stress)
                test_plan.append(lambda: run_context_test(client, selected_model, sizes))
            if skip_embeddings:
                test_plan.append(lambda: _make_skipped("Embeddings", "--skip-embeddings activado."))
            else:
                test_plan.append(lambda: run_embeddings_test(client, selected_embedding_model))

        for index, run_test in enumerate(test_plan, start=1):
            try:
                result = run_test()
            except Exception as exc:  # noqa: BLE001 - herramienta diagnóstica, no debe caer por un test.
                result = TestResult(
                    name=f"Test {index}",
                    supported=None,
                    status="fail",
                    raw_error=f"Error inesperado en el test: {type(exc).__name__}: {exc}",
                )
            results.append(result)
            _print_progress(result)

    compatibility = detect_provider(
        base_url,
        error_text=summarize_error_for_detection(results),
        model_ids=models,
    )
    report = Report(
        endpoint=base_url,
        tested_model=selected_model,
        created_at=utc_now(),
        compatibility_guess=compatibility,
        results=results,
    )
    report.errors = collect_errors(results)
    report.recommendations = build_recommendations(report)

    print_report(report, console)
    if json_output is None:
        json_output = Path("llm-tester-report.json")
    saved_json = save_json_report(report, json_output)
    console.print(f"[green]JSON guardado:[/] {saved_json}")
    if markdown:
        if markdown_output is None:
            markdown_output = Path("llm-tester-report.md")
        markdown_output = _safe_markdown_output_path(markdown_output)
        saved_md = save_markdown_report(report, markdown_output)
        console.print(f"[green]Markdown guardado:[/] {saved_md}")
    return report


def _print_progress(result: TestResult) -> None:
    style = {"ok": "green", "partial": "yellow", "fail": "red", "skipped": "cyan"}[result.status]
    supported = "sí" if result.supported is True else "no" if result.supported is False else "dudoso"
    latency = f" ({result.latency_ms:.0f} ms)" if result.latency_ms is not None else ""
    console.print(f"[{style}]{result.status.upper()}[/] {result.name}: {supported}{latency}")


@app.command()
def run(
    base_url: BaseUrlOption = None,
    api_key: ApiKeyOption = None,
    model: ModelOption = None,
    embedding_model: EmbeddingModelOption = None,
    timeout: TimeoutOption = 60.0,
    skip_vision: Annotated[bool, typer.Option("--skip-vision")] = False,
    skip_context: Annotated[bool, typer.Option("--skip-context")] = False,
    skip_embeddings: Annotated[bool, typer.Option("--skip-embeddings")] = False,
    skip_tools: Annotated[bool, typer.Option("--skip-tools")] = False,
    context_max_test: Annotated[
        int,
        typer.Option("--context-max-test", min=1000, help="Máximo aproximado de tokens para contexto."),
    ] = 32_000,
    markdown: Annotated[bool, typer.Option("--markdown", help="Guarda también informe Markdown.")] = False,
    json_output: Annotated[
        Path | None,
        typer.Option("--json-output", help="Ruta del informe JSON."),
    ] = None,
    markdown_output: Annotated[
        Path | None,
        typer.Option("--markdown-output", help="Ruta del informe Markdown."),
    ] = None,
    non_interactive: Annotated[
        bool,
        typer.Option("--non-interactive", help="No pregunta por datos faltantes."),
    ] = False,
    verbose: VerboseOption = False,
) -> None:
    """Ejecuta la batería completa de pruebas."""
    load_dotenv()
    interactive = not non_interactive
    resolved_base_url = _resolve_base_url(base_url, interactive)
    resolved_api_key = _resolve_api_key(api_key, interactive)
    resolved_model = _resolve_model(model, interactive=False)
    _run_suite(
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        model=resolved_model,
        embedding_model=_resolve_embedding_model(embedding_model),
        timeout=timeout,
        verbose=verbose,
        skip_vision=skip_vision,
        skip_context=skip_context,
        skip_embeddings=skip_embeddings,
        skip_tools=skip_tools,
        context_max_test=context_max_test,
        markdown=markdown,
        json_output=json_output,
        markdown_output=markdown_output,
        interactive=interactive,
    )


@app.command("quick")
def quick(
    base_url: BaseUrlOption = None,
    api_key: ApiKeyOption = None,
    model: ModelOption = None,
    timeout: TimeoutOption = 30.0,
    json_output: Annotated[
        Path | None,
        typer.Option("--json-output", help="Ruta del informe JSON."),
    ] = Path("llm-tester-quick.json"),
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    verbose: VerboseOption = False,
) -> None:
    """Ejecuta pruebas rápidas: conectividad, chat, streaming y system prompt."""
    load_dotenv()
    interactive = not non_interactive
    _run_suite(
        base_url=_resolve_base_url(base_url, interactive),
        api_key=_resolve_api_key(api_key, interactive),
        model=_resolve_model(model, interactive=False),
        embedding_model=None,
        timeout=timeout,
        verbose=verbose,
        skip_vision=True,
        skip_context=True,
        skip_embeddings=True,
        skip_tools=True,
        context_max_test=4_000,
        quick=True,
        json_output=json_output,
        interactive=interactive,
    )


@app.command("list-models")
def list_models(
    base_url: BaseUrlOption = None,
    api_key: ApiKeyOption = None,
    timeout: TimeoutOption = 30.0,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    verbose: VerboseOption = False,
) -> None:
    """Lista modelos desde /v1/models si el endpoint lo permite."""
    load_dotenv()
    interactive = not non_interactive
    resolved_base_url = _resolve_base_url(base_url, interactive)
    resolved_api_key = _resolve_api_key(api_key, interactive)
    with LLMClient(resolved_base_url, resolved_api_key, timeout=timeout, verbose=verbose) as client:
        try:
            response = client.list_models()
        except LLMAPIError as exc:
            console.print(f"[red]No se pudo listar modelos:[/] {exc.readable()}")
            raise typer.Exit(code=1) from exc
    data = response.data if isinstance(response.data, dict) else {}
    models = data.get("data", []) if isinstance(data, dict) else []
    table = Table(title=f"Modelos disponibles ({len(models)})")
    table.add_column("ID", style="bold")
    table.add_column("Owner")
    table.add_column("Objeto")
    for item in models:
        if not isinstance(item, dict):
            continue
        table.add_row(str(item.get("id", "")), str(item.get("owned_by", "-")), str(item.get("object", "-")))
    console.print(table)


@app.command("stress-context")
def stress_context(
    base_url: BaseUrlOption = None,
    api_key: ApiKeyOption = None,
    model: ModelOption = None,
    timeout: TimeoutOption = 120.0,
    max_test: Annotated[
        int,
        typer.Option("--max-test", min=1000, help="Máximo aproximado de tokens a probar."),
    ] = 32_768,
    json_output: Annotated[
        Path | None,
        typer.Option("--json-output", help="Ruta del informe JSON."),
    ] = Path("llm-tester-context.json"),
    markdown: Annotated[bool, typer.Option("--markdown")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    verbose: VerboseOption = False,
) -> None:
    """Ejecuta una prueba de contexto larga, explícita y potencialmente más cara."""
    load_dotenv()
    interactive = not non_interactive
    base = _resolve_base_url(base_url, interactive)
    key = _resolve_api_key(api_key, interactive)
    with LLMClient(base, key, timeout=timeout, verbose=verbose) as client:
        connectivity = run_connectivity_test(client)
        models = available_models_from_result(connectivity)
        selected_model = _select_model(_resolve_model(model, interactive=False), models, interactive=interactive)
        sizes = _context_sizes(max_test, stress=True)
        result = run_context_test(client, selected_model, sizes)
    report = Report(
        endpoint=base,
        tested_model=selected_model,
        created_at=utc_now(),
        compatibility_guess=detect_provider(base, model_ids=models, error_text=result.raw_error),
        results=[connectivity, result],
    )
    report.errors = collect_errors(report.results)
    report.recommendations = build_recommendations(report)
    print_report(report, console)
    saved_json = save_json_report(report, json_output)
    console.print(f"[green]JSON guardado:[/] {saved_json}")
    if markdown:
        saved_md = save_markdown_report(report, Path("llm-tester-context.md"))
        console.print(f"[green]Markdown guardado:[/] {saved_md}")


if __name__ == "__main__":
    app()

# LLM Tester

CLI tool for testing LLM models and OpenAI-compatible API endpoints.

`llm-tester` lets you provide a base URL, API key, and model name, then runs an automated capability suite covering chat, streaming, system prompts, tool calling, JSON mode, structured outputs, vision, reasoning parameters, approximate context limits, embeddings, and provider compatibility hints.

It uses direct HTTP calls through `httpx`; the official OpenAI SDK is not required.

## Features

- OpenAI-compatible HTTP client built on `httpx`
- Clear `typer` CLI with colored `rich` reports
- Pydantic models for config and test results
- Safe API key handling with masked output
- JSON report export
- Optional Markdown report export
- API feature detection for reasoning/thinking/generation parameters
- No telemetry
- No external calls except the endpoint you configure
- Dockerfile included

## Tested Capabilities

| Area | What it checks |
|---|---|
| Connectivity | Calls `/v1/models` when available and lists models |
| Basic chat | Sends `Responde unicamente con OK`, checks response, latency, and usage |
| Streaming | Sends `stream: true`, measures time to first token and total time |
| System prompt | Checks whether `system` messages are followed |
| Tool calling | Sends a fake `get_server_status` tool and detects `tool_calls` |
| JSON mode | Uses `response_format: {"type": "json_object"}` and validates parseable JSON |
| Structured outputs | Uses `response_format: {"type": "json_schema"}` and validates schema |
| Vision | Sends text plus a generated PNG image as a data URL |
| Reasoning | Tests common reasoning params without requesting private chain of thought |
| API feature detection | Probes accepted/rejected generation, reasoning, and thinking parameters |
| Context | Tries configurable approximate context sizes |
| Embeddings | Calls `/v1/embeddings` when an embeddings model is provided or detected |
| Compatibility | Heuristics for OpenAI, Ollama, llama.cpp, LM Studio, vLLM, OpenRouter, LiteLLM, Groq, Together, DeepInfra, and other compatible APIs |

## Requirements

- Python 3.11+
- Windows, macOS, or Linux

## Installation

Using `uv`:

```bash
uv sync --python 3.11
uv run llm-tester --help
```

Using `pip`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
llm-tester --help
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
llm-tester --help
```

## Configuration

You can pass values through CLI options:

```bash
llm-tester run --base-url https://endpoint.com/v1 --api-key sk-xxx --model model-name
```

Or through environment variables:

```bash
export LLM_TESTER_BASE_URL=https://endpoint.com/v1
export LLM_TESTER_API_KEY=sk-xxx
export LLM_TESTER_MODEL=model-name
```

PowerShell:

```powershell
$env:LLM_TESTER_BASE_URL="https://endpoint.com/v1"
$env:LLM_TESTER_API_KEY="sk-xxx"
$env:LLM_TESTER_MODEL="model-name"
```

You can also copy `.env.example` to `.env`.

## Commands

List models:

```bash
llm-tester list-models --base-url https://endpoint.com/v1 --api-key sk-xxx
```

Quick test:

```bash
llm-tester quick --base-url https://endpoint.com/v1 --api-key sk-xxx --model model-name
```

Full test:

```bash
llm-tester run \
  --base-url https://endpoint.com/v1 \
  --api-key sk-xxx \
  --model model-name \
  --markdown
```

Cheaper full test:

```bash
llm-tester run \
  --base-url https://endpoint.com/v1 \
  --api-key sk-xxx \
  --model model-name \
  --skip-context \
  --skip-vision \
  --skip-embeddings \
  --markdown
```

Context stress test:

```bash
llm-tester stress-context \
  --base-url https://endpoint.com/v1 \
  --api-key sk-xxx \
  --model model-name \
  --max-test 32768 \
  --markdown
```

API feature detection:

```bash
llm-tester detect-features \
  --base-url https://endpoint.com/v1 \
  --api-key sk-xxx \
  --model model-name \
  --output-token-caps 1024,4096,8192,16384 \
  --markdown
```

Verbose diagnostics:

```bash
llm-tester run \
  --base-url https://endpoint.com/v1 \
  --api-key sk-xxx \
  --model model-name \
  --verbose
```

## Local Endpoint Examples

Ollama:

```bash
llm-tester run --base-url http://localhost:11434/v1 --model llama3.1 --skip-embeddings
```

LM Studio:

```bash
llm-tester run --base-url http://localhost:1234/v1 --model local-model --skip-embeddings
```

llama.cpp server:

```bash
llm-tester run --base-url http://localhost:8080/v1 --model local-model --skip-vision --skip-embeddings
```

OpenAI-compatible hosted endpoint:

```bash
llm-tester run --base-url https://api.openai.com/v1 --api-key sk-xxx --model gpt-4o-mini --markdown
```

## Output

The CLI prints a colored report and writes JSON by default:

```text
llm-tester-report.json
```

With `--markdown`, it also writes:

```text
llm-tester-report.md
```

Each test returns:

- `name`
- `supported`: `true`, `false`, or `null`
- `status`: `ok`, `fail`, `partial`, or `skipped`
- `latency_ms`
- `details`
- `raw_error`
- `usage`

## Security Notes

- API keys are masked in console output.
- API keys are not written into reports.
- The tool has no telemetry.
- The tool only calls the endpoint you configure.
- Avoid passing secrets directly in shell history. Prefer environment variables or the hidden interactive prompt.

## Cost Notes

Some tests consume tokens. Context tests can be expensive. By default, context testing is conservative, and larger context tests require explicit `stress-context` usage.

Recommended low-cost first run:

```bash
llm-tester quick --base-url https://endpoint.com/v1 --api-key sk-xxx --model model-name
```

## Project Structure

```text
llm_tester/
  __init__.py
  cli.py
  client.py
  models.py
  reporting.py
  utils.py
  tests/
    chat.py
    streaming.py
    vision.py
    tools.py
    json_mode.py
    reasoning.py
    api_features.py
    context.py
    embeddings.py
```

## Docker

Build:

```bash
docker build -t llm-tester .
```

Run:

```bash
docker run --rm llm-tester --help
```

## License

MIT

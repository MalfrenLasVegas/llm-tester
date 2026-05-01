from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import httpx

from .models import ClientResponse
from .utils import compact_error, mask_api_key, normalize_base_url


class LLMAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}

    def readable(self) -> str:
        prefix = f"HTTP {self.status_code}: " if self.status_code else ""
        body = compact_error(self.body)
        return f"{prefix}{self.args[0]}{': ' + body if body else ''}"


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout: float = 60.0,
        verbose: bool = False,
    ) -> None:
        self.base_url = normalize_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout
        self.verbose = verbose
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def list_models(self) -> ClientResponse:
        return self._request("GET", "models")

    def chat_completion(self, payload: dict[str, Any]) -> ClientResponse:
        return self._request("POST", "chat/completions", json_payload=payload)

    def embeddings(self, payload: dict[str, Any]) -> ClientResponse:
        return self._request("POST", "embeddings", json_payload=payload)

    def chat_completion_stream(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        stream_payload = {**payload, "stream": True}
        if self.verbose:
            self._print_payload("POST", "chat/completions", stream_payload)
        started = time.perf_counter()
        try:
            with self._client.stream(
                "POST",
                self.url("chat/completions"),
                headers=self.headers,
                json=stream_payload,
            ) as response:
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", errors="replace")
                    raise LLMAPIError(
                        response.reason_phrase,
                        status_code=response.status_code,
                        body=body,
                        headers=dict(response.headers),
                    )
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        yield {
                            "raw": line,
                            "elapsed_ms": (time.perf_counter() - started) * 1000,
                        }
                        continue
                    data["_elapsed_ms"] = (time.perf_counter() - started) * 1000
                    yield data
        except httpx.TimeoutException as exc:
            raise LLMAPIError(f"Timeout tras {self.timeout}s", body=str(exc)) from exc
        except httpx.RequestError as exc:
            raise LLMAPIError("Error de conectividad", body=str(exc)) from exc

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
    ) -> ClientResponse:
        if self.verbose:
            self._print_payload(method, path, json_payload)
        started = time.perf_counter()
        try:
            response = self._client.request(
                method,
                self.url(path),
                headers=self.headers,
                json=json_payload,
            )
        except httpx.TimeoutException as exc:
            raise LLMAPIError(f"Timeout tras {self.timeout}s", body=str(exc)) from exc
        except httpx.RequestError as exc:
            raise LLMAPIError("Error de conectividad", body=str(exc)) from exc

        latency_ms = (time.perf_counter() - started) * 1000
        body_text = response.text
        try:
            data: Any = response.json() if body_text else {}
        except ValueError:
            data = body_text

        if response.status_code >= 400:
            raise LLMAPIError(
                response.reason_phrase,
                status_code=response.status_code,
                body=data,
                headers=dict(response.headers),
            )
        return ClientResponse(
            data=data,
            status_code=response.status_code,
            headers=dict(response.headers),
            latency_ms=latency_ms,
        )

    def _print_payload(self, method: str, path: str, payload: dict[str, Any] | None) -> None:
        safe_headers = {**self.headers}
        if "Authorization" in safe_headers:
            safe_headers["Authorization"] = f"Bearer {mask_api_key(self.api_key)}"
        print(f"\n[{method}] {self.url(path)}")
        print(f"headers={safe_headers}")
        if payload is not None:
            print(json.dumps(payload, ensure_ascii=False, indent=2))

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


TestStatus = Literal["ok", "fail", "partial", "skipped"]


class TestResult(BaseModel):
    name: str
    supported: bool | None
    status: TestStatus
    latency_ms: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    raw_error: str | None = None
    usage: dict[str, Any] | None = None


class ClientResponse(BaseModel):
    data: Any
    status_code: int
    headers: dict[str, str] = Field(default_factory=dict)
    latency_ms: float


class LLMTesterConfig(BaseModel):
    base_url: str
    api_key: str | None = None
    model: str | None = None
    embedding_model: str | None = None
    timeout: float = 60.0
    verbose: bool = False


class Report(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    endpoint: str
    tested_model: str | None
    created_at: datetime
    compatibility_guess: str = "otro compatible"
    results: list[TestResult] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def latency_values(self) -> list[float]:
        return [r.latency_ms for r in self.results if r.latency_ms is not None]

    @property
    def average_latency_ms(self) -> float | None:
        values = self.latency_values
        if not values:
            return None
        return sum(values) / len(values)

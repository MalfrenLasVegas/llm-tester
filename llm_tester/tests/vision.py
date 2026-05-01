from __future__ import annotations

import base64
import binascii
import struct
import zlib

from llm_tester.client import LLMAPIError, LLMClient
from llm_tester.models import TestResult
from llm_tester.utils import extract_chat_text


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = binascii.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _make_test_png_data_url() -> str:
    width = 8
    height = 8
    # RGB image with red pixels on the left half and blue pixels on the right half.
    rows = []
    for _ in range(height):
        raw = bytearray([0])
        for x in range(width):
            raw.extend((255, 0, 0) if x < width // 2 else (0, 0, 255))
        rows.append(bytes(raw))
    png = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(b"".join(rows))),
            _png_chunk(b"IEND", b""),
        ]
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


TEST_IMAGE_DATA_URL = _make_test_png_data_url()


def run_vision_test(client: LLMClient, model: str) -> TestResult:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe brevemente lo que ves en la imagen."},
                    {"type": "image_url", "image_url": {"url": TEST_IMAGE_DATA_URL}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 80,
    }
    try:
        response = client.chat_completion(payload)
        text = extract_chat_text(response.data)
        supported = bool(text.strip())
        return TestResult(
            name="Vision",
            supported=supported,
            status="ok" if supported else "partial",
            latency_ms=response.latency_ms,
            details={"response": text[:500], "image": "8x8 PNG base64 generado en memoria"},
            usage=response.data.get("usage") if isinstance(response.data, dict) else None,
        )
    except LLMAPIError as exc:
        raw = exc.readable()
        markers = [
            "image",
            "vision",
            "multimodal",
            "content type",
            "invalid message content",
            "unsupported",
        ]
        likely_unsupported = any(marker in raw.lower() for marker in markers)
        return TestResult(
            name="Vision",
            supported=False if likely_unsupported else None,
            status="fail",
            details={"classification": "no soportado" if likely_unsupported else "dudoso"},
            raw_error=raw,
        )

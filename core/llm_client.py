from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def call_ollama_generate(
    prompt: str,
    model: str,
    url: str,
    options: dict[str, Any] | None,
    timeout: int = 180,
) -> dict[str, Any]:
    if not url:
        return {
            "ok": False,
            "response_text": "",
            "raw": None,
            "error": "Ollama URL is required.",
        }
    if not model:
        return {
            "ok": False,
            "response_text": "",
            "raw": None,
            "error": "Model name is required.",
        }

    endpoint = url.rstrip("/")
    if not endpoint.endswith("/api/generate"):
        endpoint = f"{endpoint}/api/generate"

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if options:
        payload["options"] = options

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.getcode()
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        message = error_body or getattr(exc, "reason", "HTTP error")
        return {
            "ok": False,
            "response_text": "",
            "raw": None,
            "error": f"HTTP {exc.code}: {message}",
        }
    except urllib.error.URLError as exc:
        return {
            "ok": False,
            "response_text": "",
            "raw": None,
            "error": f"Connection error: {exc.reason}",
        }
    except Exception as exc:  # pragma: no cover - unexpected transport errors
        return {
            "ok": False,
            "response_text": "",
            "raw": None,
            "error": str(exc),
        }

    if status != 200:
        return {
            "ok": False,
            "response_text": "",
            "raw": None,
            "error": f"Unexpected status {status}",
        }

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "response_text": "",
            "raw": None,
            "error": "Invalid JSON response from Ollama.",
        }

    return {
        "ok": True,
        "response_text": str(payload.get("response", "")),
        "raw": payload,
        "error": None,
    }

#!/usr/bin/env python3
"""Live smoke checks for DeepThink and DeepSeek built-in web search routing."""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from _path import PROJECT_ROOT  # noqa: F401


def request_json(base_url: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 600):
    url = base_url.rstrip("/") + path
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw), time.monotonic() - started
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        return exc.code, json.loads(raw), time.monotonic() - started


def assistant_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"ok: {message}")


def has_reasoning_leak(text: str) -> bool:
    lowered = text.lower().replace("deepbill_deepthink_done", "")
    markers = (
        "deepthink",
        "reasoning",
        "thinking",
        "рассужд",
        "we need to",
        "the user said",
        "the assistant already",
        "so just output",
    )
    return any(marker in lowered for marker in markers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live DeepThink/web-search smoke test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    status, health, _elapsed = request_json(args.base_url, "/health", timeout=30)
    assert_true(status == 200 and health.get("ready"), "adapter health is ready")
    assert_true("diagnostics" in health, "health includes diagnostics")

    reasoning_payload = {
        "model": "deepbill-deepthink",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ответь только этой строкой: 323 DEEPBILL_RT_DONE"
                ),
            }
        ],
        "timeout": args.timeout,
    }
    status, reasoning_body, elapsed = request_json(args.base_url, "/chat/completions", reasoning_payload, args.timeout)
    text = assistant_text(reasoning_body)
    assert_true(status == 200, "reasoning request returns HTTP 200")
    assert_true("DEEPBILL_RT_DONE" in text, "reasoning final answer marker is returned")
    assert_true(not has_reasoning_leak(text), "reasoning panel text is not returned")
    print(f"reasoning elapsed={elapsed:.1f}s")

    web_payload = {
        "model": "deepbill-deepthink",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Используй встроенный интернет-поиск DeepSeek, дождись результатов и кратко ответь: "
                    "какая сегодня дата? В конце напиши DEEPBILL_WEB_SEARCH_DONE."
                ),
            }
        ],
        "timeout": args.timeout,
    }
    result: dict[str, Any] = {}

    def run_web() -> None:
        result["status"], result["body"], result["elapsed"] = request_json(
            args.base_url,
            "/chat/completions",
            web_payload,
            args.timeout,
        )

    thread = threading.Thread(target=run_web, name="live-web-search")
    thread.start()
    time.sleep(1.0)
    busy_status, busy_body, _busy_elapsed = request_json(
        args.base_url,
        "/chat/completions",
        {"messages": [{"role": "user", "content": "короткий запрос во время web search"}], "timeout": 60},
        timeout=90,
    )
    if busy_status == 429:
        assert_true(busy_body.get("error", {}).get("type") == "server_busy", "overlap during web search returns server_busy")
    else:
        print(f"note: overlap request finished with status={busy_status}; web-search request may have completed quickly")
    thread.join(timeout=args.timeout + 30)
    assert_true(not thread.is_alive(), "web-search request completed")
    web_text = assistant_text(result.get("body") or {})
    assert_true(result.get("status") == 200, "web-search request returns HTTP 200")
    assert_true("DEEPBILL_WEB_SEARCH_DONE" in web_text, "web-search final answer marker is returned")
    assert_true(not has_reasoning_leak(web_text), "web-search answer does not leak reasoning text")
    print(f"web_search elapsed={float(result.get('elapsed') or 0):.1f}s")


if __name__ == "__main__":
    main()

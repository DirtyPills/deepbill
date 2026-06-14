#!/usr/bin/env python3
"""Headless DBill adapter service used by the quiet shell aliases."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepseek_runtime import (  # noqa: E402
    BrowserWorker,
    DEFAULT_ANSWER_STABLE_SEC,
    DEFAULT_TIMEOUT_SEC,
    DEFAULT_USER_DATA_DIR,
    clamp_answer_stable_sec,
    normalize_reasoning_mode,
)
from openai_adapter import OpenAIAdapter  # noqa: E402


SETTINGS_FILE = PROJECT_ROOT / "dbill_settings.json"


def load_settings() -> dict[str, Any]:
    try:
        raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8")) if SETTINGS_FILE.exists() else {}
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    return raw


def int_setting(settings: dict[str, Any], name: str, default: int) -> int:
    try:
        return int(settings.get(name, default))
    except Exception:
        return default


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Run the DBill adapter without the Tk GUI")
    parser.add_argument("--port", type=int, default=int_setting(settings, "adapter_port", 8080))
    parser.add_argument("--timeout", type=int, default=int_setting(settings, "timeout_sec", DEFAULT_TIMEOUT_SEC))
    parser.add_argument("--answer-stable-sec", default=settings.get("answer_stable_sec", DEFAULT_ANSWER_STABLE_SEC))
    parser.add_argument("--reasoning-mode", default=settings.get("reasoning_mode", "off"))
    parser.add_argument("--user-data-dir", default=DEFAULT_USER_DATA_DIR)
    parser.add_argument("--visible", action="store_true", help="Run the browser visibly while keeping the GUI disabled")
    return parser.parse_args()


def main() -> None:
    os.chdir(PROJECT_ROOT)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
    )
    args = parse_args()
    stop_event = threading.Event()

    def request_stop(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    worker = BrowserWorker(
        headless=not args.visible,
        user_data_dir=str(args.user_data_dir),
        answer_stable_sec=clamp_answer_stable_sec(args.answer_stable_sec),
        reasoning_mode=normalize_reasoning_mode(args.reasoning_mode),
    )
    adapter = OpenAIAdapter(browser_worker=worker, port=max(1024, min(int(args.port), 65535)))
    try:
        worker.start()
        adapter.start()
        time.sleep(0.5)
        if not adapter.is_running:
            raise RuntimeError(f"DBill adapter HTTP server did not stay running on port {adapter.port}")
        logging.info(
            "DBill quiet adapter started url=%s headless=%s timeout=%s",
            adapter.get_url(),
            not args.visible,
            max(30, int(args.timeout)),
        )
        while not stop_event.wait(1.0):
            if not adapter.is_running:
                raise RuntimeError("DBill adapter HTTP server stopped unexpectedly")
    finally:
        logging.info("DBill quiet adapter stopping.")
        try:
            adapter.stop()
        finally:
            worker.stop()
        logging.info("DBill quiet adapter stopped.")


if __name__ == "__main__":
    main()

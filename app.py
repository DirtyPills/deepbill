#!/usr/bin/env python3
"""Chat-only GUI for DeepSeek Web and the DBill OpenAI-compatible adapter."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import scrolledtext, ttk

from deepseek_runtime import (
    BrowserWorker,
    DEFAULT_ANSWER_STABLE_SEC,
    DEFAULT_TIMEOUT_SEC,
    DEFAULT_USER_DATA_DIR,
    clamp_answer_stable_sec,
    normalize_reasoning_mode,
)
from openai_adapter import OpenAIAdapter


SETTINGS_FILE = "dbill_settings.json"


def app_dir() -> Path:
    return Path(__file__).resolve().parent


def default_settings() -> dict[str, Any]:
    return {
        "adapter_enabled": False,
        "adapter_port": 8080,
        "timeout_sec": DEFAULT_TIMEOUT_SEC,
        "answer_stable_sec": DEFAULT_ANSWER_STABLE_SEC,
        "reasoning_mode": "off",
    }


def load_settings(path: Path) -> dict[str, Any]:
    base = default_settings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    try:
        port = int(raw.get("adapter_port", base["adapter_port"]))
    except Exception:
        port = int(base["adapter_port"])
    try:
        timeout = int(raw.get("timeout_sec", base["timeout_sec"]))
    except Exception:
        timeout = int(base["timeout_sec"])
    stable = clamp_answer_stable_sec(raw.get("answer_stable_sec", base["answer_stable_sec"]))
    reasoning_mode = normalize_reasoning_mode(raw.get("reasoning_mode", base["reasoning_mode"]))
    return {
        "adapter_enabled": bool(raw.get("adapter_enabled", base["adapter_enabled"])),
        "adapter_port": max(1024, min(port, 65535)),
        "timeout_sec": max(30, timeout),
        "answer_stable_sec": stable,
        "reasoning_mode": reasoning_mode,
    }


def save_settings(path: Path, settings: dict[str, Any]) -> None:
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


class DBillChatApp:
    def __init__(self, root: tk.Tk, headless: bool, user_data_dir: str, timeout_sec: int | None):
        self.root = root
        self.settings_path = app_dir() / SETTINGS_FILE
        self.settings = load_settings(self.settings_path)
        if timeout_sec is not None:
            self.settings["timeout_sec"] = max(30, int(timeout_sec))
        self.timeout_sec = int(self.settings["timeout_sec"])
        self.answer_stable_sec = clamp_answer_stable_sec(self.settings["answer_stable_sec"])
        self.reasoning_mode = normalize_reasoning_mode(self.settings["reasoning_mode"])
        self.adapter_port = int(self.settings["adapter_port"])
        self.openai_adapter: OpenAIAdapter | None = None

        self.root.title("DBill Chat Adapter")
        self.root.geometry("980x720+120+80")
        self.root.minsize(760, 560)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.browser_status_var = tk.StringVar(value="Browser: starting")
        self.adapter_url_var = tk.StringVar(value="")
        self.adapter_port_var = tk.StringVar(value=str(self.adapter_port))
        self.timeout_var = tk.StringVar(value=str(self.timeout_sec))
        self.answer_stable_var = tk.StringVar(value=f"{self.answer_stable_sec:g}")
        self.reasoning_mode_var = tk.StringVar(value=self.reasoning_mode)
        self.chat_prompt_var = tk.StringVar(value="")

        self.browser = BrowserWorker(
            headless=headless,
            user_data_dir=user_data_dir,
            answer_stable_sec=self.answer_stable_sec,
            reasoning_mode=self.reasoning_mode,
        )
        self.browser.start()
        self._build_ui()
        self._log("System", "DeepSeek browser runtime is starting.")

        if self.settings["adapter_enabled"]:
            self._start_adapter()
        self.root.after(1000, self._poll_status)

    def _build_ui(self) -> None:
        shell = ttk.Frame(self.root, padding=12)
        shell.pack(fill=tk.BOTH, expand=True)

        browser_row = ttk.Frame(shell)
        browser_row.pack(fill=tk.X)
        ttk.Label(browser_row, textvariable=self.browser_status_var).pack(side=tk.LEFT)
        ttk.Button(browser_row, text="Open Browser", command=self._open_browser).pack(side=tk.RIGHT)
        ttk.Button(browser_row, text="New Chat", command=self._new_chat).pack(side=tk.RIGHT, padx=(0, 8))

        adapter = ttk.LabelFrame(shell, text="OpenAI API Adapter", padding=10)
        adapter.pack(fill=tk.X, pady=(10, 10))
        controls = ttk.Frame(adapter)
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="Port").pack(side=tk.LEFT)
        ttk.Entry(controls, textvariable=self.adapter_port_var, width=7).pack(side=tk.LEFT, padx=(6, 8))
        ttk.Button(controls, text="Apply Port", command=self._apply_port).pack(side=tk.LEFT, padx=(0, 8))
        self.adapter_toggle_button = ttk.Button(controls, text="Start Adapter", command=self._toggle_adapter)
        self.adapter_toggle_button.pack(side=tk.LEFT, padx=(0, 8))
        self.adapter_status_label = ttk.Label(controls, text="Status: stopped", foreground="gray")
        self.adapter_status_label.pack(side=tk.LEFT)

        url_row = ttk.Frame(adapter)
        url_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(url_row, text="OpenAI base URL").pack(side=tk.LEFT)
        ttk.Entry(url_row, textvariable=self.adapter_url_var, state="readonly", width=48).pack(
            side=tk.LEFT, padx=(8, 8)
        )
        ttk.Button(url_row, text="Copy URL", command=self._copy_url).pack(side=tk.LEFT)
        ttk.Button(url_row, text="Open Browser", command=self._open_browser).pack(side=tk.LEFT, padx=(8, 0))

        timeout_row = ttk.Frame(adapter)
        timeout_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(timeout_row, text="Answer timeout, sec").pack(side=tk.LEFT)
        ttk.Entry(timeout_row, textvariable=self.timeout_var, width=8).pack(side=tk.LEFT, padx=(8, 8))
        ttk.Button(timeout_row, text="Apply Timeout", command=self._apply_timeout).pack(side=tk.LEFT)
        ttk.Label(timeout_row, text="Finish wait, sec").pack(side=tk.LEFT, padx=(20, 0))
        ttk.Entry(timeout_row, textvariable=self.answer_stable_var, width=6).pack(side=tk.LEFT, padx=(8, 8))
        ttk.Button(timeout_row, text="Apply Wait", command=self._apply_answer_stable).pack(side=tk.LEFT)

        reasoning_row = ttk.Frame(adapter)
        reasoning_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(reasoning_row, text="Использовать рассуждения").pack(side=tk.LEFT)
        reasoning_combo = ttk.Combobox(
            reasoning_row,
            textvariable=self.reasoning_mode_var,
            values=("off", "auto", "on"),
            state="readonly",
            width=8,
        )
        reasoning_combo.pack(side=tk.LEFT, padx=(8, 8))
        reasoning_combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_reasoning_mode())
        ttk.Button(reasoning_row, text="Apply", command=self._apply_reasoning_mode).pack(side=tk.LEFT)

        chat = ttk.LabelFrame(shell, text="Manual Chat", padding=10)
        chat.pack(fill=tk.BOTH, expand=True)
        ask_row = ttk.Frame(chat)
        ask_row.pack(fill=tk.X)
        ask_entry = ttk.Entry(ask_row, textvariable=self.chat_prompt_var)
        ask_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ask_entry.bind("<Return>", lambda _event: self._send_chat())
        self.send_button = ttk.Button(ask_row, text="Send", command=self._send_chat)
        self.send_button.pack(side=tk.LEFT, padx=(8, 0))
        self.log = scrolledtext.ScrolledText(chat, wrap=tk.WORD, state=tk.NORMAL)
        self.log.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self._update_adapter_ui()

    def _log(self, who: str, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.insert(tk.END, f"[{stamp}] {who}: {text}\n")
        self.log.see(tk.END)

    def _async(self, name: str, action) -> None:
        threading.Thread(target=action, name=name, daemon=True).start()

    def _start_adapter(self) -> None:
        if self.openai_adapter is not None and self.openai_adapter.is_running:
            return
        self.openai_adapter = OpenAIAdapter(browser_worker=self.browser, port=self.adapter_port)
        self.openai_adapter.start()
        self.settings["adapter_enabled"] = True
        self._persist_settings()
        self._log("System", f"Adapter started at {self.openai_adapter.get_url()}")
        self._update_adapter_ui()

    def _stop_adapter(self) -> None:
        if self.openai_adapter is None:
            return
        self.openai_adapter.stop()
        self.openai_adapter = None
        self.settings["adapter_enabled"] = False
        self._persist_settings()
        self._log("System", "Adapter stopped.")
        self._update_adapter_ui()

    def _toggle_adapter(self) -> None:
        if self.openai_adapter is not None and self.openai_adapter.is_running:
            self._stop_adapter()
        else:
            self._start_adapter()

    def _apply_port(self) -> None:
        try:
            port = int(self.adapter_port_var.get().strip())
            if port < 1024 or port > 65535:
                raise ValueError
        except Exception:
            self._log("Error", "Port must be between 1024 and 65535.")
            return
        running = self.openai_adapter is not None and self.openai_adapter.is_running
        if running:
            self._stop_adapter()
        self.adapter_port = port
        self.settings["adapter_port"] = port
        self._persist_settings()
        self._log("System", f"Adapter port set to {port}.")
        if running:
            self._start_adapter()
        self._update_adapter_ui()

    def _apply_timeout(self) -> None:
        try:
            timeout = max(30, int(self.timeout_var.get().strip()))
        except Exception:
            self._log("Error", "Timeout must be an integer.")
            return
        self.timeout_sec = timeout
        self.timeout_var.set(str(timeout))
        self.settings["timeout_sec"] = timeout
        self._persist_settings()
        self._log("System", f"Answer timeout set to {timeout} seconds.")

    def _apply_answer_stable(self) -> None:
        try:
            value = float(self.answer_stable_var.get().strip().replace(",", "."))
        except Exception:
            self._log("Error", "Finish wait must be a number of seconds.")
            return
        stable = self.browser.set_answer_stable_sec(value)
        self.answer_stable_sec = stable
        self.answer_stable_var.set(f"{stable:g}")
        self.settings["answer_stable_sec"] = stable
        self._persist_settings()
        self._log("System", f"Answer finish wait set to {stable:g} seconds.")

    def _apply_reasoning_mode(self) -> None:
        mode = normalize_reasoning_mode(self.reasoning_mode_var.get())
        self.reasoning_mode = self.browser.set_reasoning_mode(mode)
        self.reasoning_mode_var.set(self.reasoning_mode)
        self.settings["reasoning_mode"] = self.reasoning_mode
        self._persist_settings()
        self._log("System", f"Reasoning mode set to {self.reasoning_mode}.")

    def _copy_url(self) -> None:
        value = self.adapter_url_var.get().strip()
        if not value:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(value)
        self._log("System", f"Copied adapter URL: {value}")

    def _open_browser(self) -> None:
        def run() -> None:
            try:
                self.browser.open_browser()
                self.root.after(0, lambda: self._log("System", "DeepSeek browser tab opened."))
            except Exception as exc:
                self.root.after(0, lambda: self._log("Error", f"Open browser failed: {exc}"))

        self._async("DBillOpenBrowser", run)

    def _new_chat(self) -> None:
        def run() -> None:
            try:
                self.browser.new_chat()
                self.root.after(0, lambda: self._log("System", "New DeepSeek chat opened."))
            except Exception as exc:
                self.root.after(0, lambda: self._log("Error", f"New chat failed: {exc}"))

        self._async("DBillNewChat", run)

    def _send_chat(self) -> None:
        prompt = self.chat_prompt_var.get().strip()
        if not prompt:
            return
        self.chat_prompt_var.set("")
        self.send_button.config(state=tk.DISABLED)
        self._log("You", prompt)

        def run() -> None:
            try:
                answer = self.browser.ask_text(prompt, timeout=self.timeout_sec)
                self.root.after(0, lambda: self._finish_answer(answer))
            except Exception as exc:
                self.root.after(0, lambda: self._finish_answer(f"Request failed: {exc}", error=True))

        self._async("DBillManualChat", run)

    def _finish_answer(self, answer: str, error: bool = False) -> None:
        self._log("Error" if error else "DeepSeek", answer.strip() or "<empty>")
        self.send_button.config(state=tk.NORMAL)

    def _poll_status(self) -> None:
        started, ready, error = self.browser.status()
        if error:
            self.browser_status_var.set(f"Browser: error - {error}")
        elif not started:
            self.browser_status_var.set("Browser: starting")
        elif ready:
            self.browser_status_var.set("Browser: ready")
        else:
            self.browser_status_var.set("Browser: waiting for login")
        self._update_adapter_ui()
        self.root.after(2000, self._poll_status)

    def _update_adapter_ui(self) -> None:
        if self.openai_adapter is not None and self.openai_adapter.is_running:
            self.adapter_toggle_button.config(text="Stop Adapter")
            self.adapter_status_label.config(text="Status: running", foreground="green")
            self.adapter_url_var.set(self.openai_adapter.get_url())
            return
        self.adapter_toggle_button.config(text="Start Adapter")
        self.adapter_status_label.config(text="Status: stopped", foreground="gray")
        self.adapter_url_var.set("")

    def _persist_settings(self) -> None:
        save_settings(self.settings_path, self.settings)

    def _on_close(self) -> None:
        self._persist_settings()
        if self.openai_adapter is not None:
            self.openai_adapter.stop()
        self.browser.stop()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DBill text chat and OpenAI-compatible adapter")
    parser.add_argument("--headless", action="store_true", help="Run the Playwright DeepSeek browser headlessly")
    parser.add_argument("--user-data-dir", default=DEFAULT_USER_DATA_DIR, help="Persistent DeepSeek browser profile")
    parser.add_argument("--timeout", type=int, default=None, help="Override answer timeout in seconds")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s")
    args = parse_args()
    root = tk.Tk()
    DBillChatApp(root, headless=args.headless, user_data_dir=args.user_data_dir, timeout_sec=args.timeout)
    root.mainloop()


if __name__ == "__main__":
    main()

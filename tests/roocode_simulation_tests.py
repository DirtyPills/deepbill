#!/usr/bin/env python3
"""Roo Code-like adapter simulation with real local tool execution."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from _path import PROJECT_ROOT  # noqa: F401
import openai_adapter as adapter_module
from openai_adapter import OpenAIAdapter


BASE_URL = "http://127.0.0.1:18081/v1"


class RooSimulationWorker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.reasoning_mode = "off"

    def status(self):
        return True, True, None

    def diagnostics(self) -> dict[str, Any]:
        last = self.calls[-1] if self.calls else {}
        return {
            "reasoning_mode": self.reasoning_mode,
            "last_use_reasoning": bool(last.get("use_reasoning")),
            "last_request_journal": {
                "request_id": last.get("request_id", ""),
                "events": [{"stage": "roo_sim_worker"}] if last else [],
            },
            "watchdog_restarts": 0,
            "consecutive_hangs": 0,
        }

    def get_reasoning_mode(self) -> str:
        return self.reasoning_mode

    def new_chat(self, timeout: int = 30) -> None:
        self.calls.append({"stage": "new_chat", "timeout": timeout})

    def ask_text(
        self,
        prompt: str,
        timeout: int = 180,
        use_reasoning: bool = False,
        request_id: str = "",
    ) -> str:
        self.calls.append(
            {
                "stage": "ask_text",
                "timeout": timeout,
                "prompt_chars": len(prompt),
                "use_reasoning": bool(use_reasoning),
                "request_id": request_id,
            }
        )
        return self._answer(prompt)

    def _answer(self, prompt: str) -> str:
        low = prompt.lower()
        if "roo create read marker" in low and "tool result (read_file" in low:
            return "Verified: app.py was created and read successfully."
        if "roo create read marker" in low and "tool result (create_new_file" in low:
            return '```tool_call\n{"name":"read_file","arguments":{"filepath":"app.py"}}\n```'
        if "roo create read marker" in low:
            return (
                '```tool_call\n{"name":"create_new_file","arguments":'
                '{"filepath":"app.py","contents":"print(\\"hello from roo\\")\\n"}}\n```'
            )

        if "roo edit terminal marker" in low and "tool result (run_terminal_command" in low:
            return "Done: calc.py prints green and the terminal check passed."
        read_count = low.count("tool result (read_file")
        if "roo edit terminal marker" in low and read_count >= 2:
            return '```tool_call\n{"name":"run_terminal_command","arguments":{"command":"python calc.py"}}\n```'
        if "roo edit terminal marker" in low and "tool result (edit_existing_file" in low:
            return '```tool_call\n{"name":"read_file","arguments":{"filepath":"calc.py"}}\n```'
        if "roo edit terminal marker" in low and read_count >= 1:
            return (
                '```tool_call\n{"name":"edit_existing_file","arguments":'
                '{"filepath":"calc.py","changes":"print(\\"green\\")\\n"}}\n```'
            )
        if "roo edit terminal marker" in low:
            return '```tool_call\n{"name":"read_file","arguments":{"filepath":"calc.py"}}\n```'

        if "roo multi tool marker" in low and low.count("tool result (create_new_file") >= 2:
            return "Done: both files were created from one assistant step."
        if "roo multi tool marker" in low:
            return (
                "```tool_call\n"
                "["
                '{"name":"create_new_file","arguments":{"filepath":"one.txt","contents":"one"}},'
                '{"name":"create_new_file","arguments":{"filepath":"two.txt","contents":"two"}}'
                "]\n"
                "```"
            )

        if "roo deepthink marker" in low:
            return "DeepThink route completed with final answer only."
        return "ok"


def request_json(path: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    url = BASE_URL + path
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        return exc.code, json.loads(raw)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    logging.info("ASSERT ok: %s", message)


def tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "create_new_file",
                "description": "Create a file in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"filepath": {"type": "string"}, "contents": {"type": "string"}},
                    "required": ["filepath", "contents"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"filepath": {"type": "string"}},
                    "required": ["filepath"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_existing_file",
                "description": "Replace a file in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"filepath": {"type": "string"}, "changes": {"type": "string"}},
                    "required": ["filepath", "changes"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_terminal_command",
                "description": "Run a workspace command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
    ]


def safe_path(workspace: Path, filepath: str) -> Path:
    target = (workspace / filepath).resolve()
    if not str(target).startswith(str(workspace.resolve()) + "/"):
        raise ValueError(f"unsafe filepath: {filepath}")
    return target


def execute_tool(workspace: Path, name: str, args: dict[str, Any]) -> str:
    if name == "create_new_file":
        path = safe_path(workspace, str(args["filepath"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(args.get("contents", "")), encoding="utf-8")
        return f"created {args['filepath']} chars={path.stat().st_size}"
    if name == "read_file":
        return safe_path(workspace, str(args["filepath"])).read_text(encoding="utf-8")
    if name == "edit_existing_file":
        path = safe_path(workspace, str(args["filepath"]))
        path.write_text(str(args.get("changes", "")), encoding="utf-8")
        return f"edited {args['filepath']} chars={path.stat().st_size}"
    if name == "run_terminal_command":
        command = str(args.get("command", ""))
        if command != "python calc.py":
            raise ValueError(f"unexpected command: {command}")
        completed = subprocess.run(
            [sys.executable, "calc.py"],
            cwd=workspace,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        return f"exit={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}"
    raise ValueError(f"unknown tool: {name}")


def run_agent(workspace: Path, model: str, user_content: str, max_steps: int = 8) -> tuple[str, list[dict[str, Any]]]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    for _step in range(max_steps):
        status, body = request_json(
            "/chat/completions",
            {"model": model, "messages": messages, "tools": tool_specs(), "timeout": 30},
        )
        assert_true(status == 200, "agent step returns HTTP 200")
        choice = body["choices"][0]
        message = choice["message"]
        if choice["finish_reason"] != "tool_calls":
            return str(message.get("content") or ""), messages
        tool_calls = message["tool_calls"]
        messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        for call in tool_calls:
            function = call["function"]
            args = json.loads(function.get("arguments") or "{}")
            result = execute_tool(workspace, function["name"], args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": function["name"],
                    "content": result,
                }
            )
    raise AssertionError("agent did not finish within max_steps")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    adapter_module.BROWSER_BUSY_TIMEOUT_SEC = 0.2
    worker = RooSimulationWorker()
    adapter = OpenAIAdapter(worker, port=18081)
    adapter.start()
    time.sleep(0.6)
    try:
        status, models = request_json("/models")
        assert_true(status == 200 and models["data"], "models endpoint is available")
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            final, _messages = run_agent(
                workspace,
                "deepseek-chat",
                "Roo create read marker: создай app.py и прочитай его после создания.",
            )
            assert_true("verified" in final.lower(), "create/read agent task finishes")
            assert_true((workspace / "app.py").read_text(encoding="utf-8").strip() == 'print("hello from roo")', "app.py content is real")

            final, messages = run_agent(
                workspace,
                "deepseek-chat",
                "Roo multi tool marker: создай два файла одним ответом инструмента.",
            )
            multi_tool_messages = [
                msg for msg in messages
                if msg.get("role") == "assistant" and len(msg.get("tool_calls") or []) >= 2
            ]
            assert_true("both files" in final.lower(), "multi-tool agent task finishes")
            assert_true(bool(multi_tool_messages), "Roo simulation receives multiple tool calls in one assistant message")
            assert_true((workspace / "one.txt").read_text(encoding="utf-8") == "one", "first parallel tool call creates file")
            assert_true((workspace / "two.txt").read_text(encoding="utf-8") == "two", "second parallel tool call creates file")

            (workspace / "calc.py").write_text('print("blue")\n', encoding="utf-8")
            calls_before_auto = len(worker.calls)
            final, _messages = run_agent(
                workspace,
                "deepbill-auto",
                "Roo edit terminal marker: проанализируй тест, исправь код calc.py, прочитай и запусти проверку.",
            )
            auto_calls = [call for call in worker.calls[calls_before_auto:] if call["stage"] == "ask_text"]
            assert_true("green" in final.lower(), "edit/read/terminal agent task finishes")
            assert_true((workspace / "calc.py").read_text(encoding="utf-8").strip() == 'print("green")', "calc.py was edited for real")
            assert_true(any(call["use_reasoning"] for call in auto_calls), "auto mode used reasoning for complex Roo task")

        status, deepthink = request_json(
            "/chat/completions",
            {
                "model": "deepbill-deepthink",
                "messages": [{"role": "user", "content": "Roo deepthink marker: короткая проверка маршрута."}],
                "timeout": 30,
            },
        )
        assert_true(status == 200, "deepthink model route returns HTTP 200")
        assert_true(
            worker.calls[-1]["use_reasoning"] and worker.calls[-1]["request_id"].startswith("chatcmpl-"),
            "deepthink route uses reasoning and request_id",
        )
        status, health = request_json("/health")
        assert_true(status == 200, "health returns HTTP 200 after Roo simulation")
        assert_true("last_request_journal" in health["diagnostics"], "health keeps request journal after Roo simulation")
    finally:
        adapter.stop()
    print("roocode_simulation_tests: ok")


if __name__ == "__main__":
    main()

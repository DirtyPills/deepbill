#!/usr/bin/env python3
"""Heavy Roo Code-like integration simulation for the OpenAI adapter.

The test keeps all generated files inside a temporary workspace, but talks to a
real Flask adapter over HTTP and executes Roo-style tool calls locally.
"""

from __future__ import annotations

import json
import logging
import re
import socket
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


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def line_count(text: str) -> int:
    return 0 if text == "" else text.count("\n") + 1


def fenced_tool(payload: Any) -> str:
    return "```tool_call\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


def big_python_source() -> str:
    lines = [
        '"""Generated stress module."""',
        "",
        "def total() -> int:",
        "    values = [",
    ]
    lines.extend(f"        {idx}," for idx in range(1500))
    lines.extend(
        [
            "    ]",
            "    return sum(values)",
            "",
            "if __name__ == '__main__':",
            "    print(total())",
            "",
        ]
    )
    return "\n".join(lines)


def big_markdown() -> str:
    return "\n".join(f"section {idx}: heavy-response-marker-{idx}" for idx in range(2200))


class HeavyRooWorker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.prompts: list[str] = []
        self.reasoning_mode = "off"
        self.todo_updates: list[Any] = []

    def status(self):
        return True, True, None

    def get_reasoning_mode(self) -> str:
        return self.reasoning_mode

    def set_reasoning_mode(self, mode: str) -> str:
        self.reasoning_mode = str(mode or "off")
        return self.reasoning_mode

    def diagnostics(self) -> dict[str, Any]:
        last = next((call for call in reversed(self.calls) if call["stage"] == "ask_text"), {})
        return {
            "reasoning_mode": self.reasoning_mode,
            "last_use_reasoning": bool(last.get("use_reasoning")),
            "last_request_journal": {
                "request_id": last.get("request_id", ""),
                "events": [{"stage": "heavy_roo_worker"}] if last else [],
            },
            "watchdog_restarts": 0,
            "consecutive_hangs": 0,
        }

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
        self.prompts.append(prompt)
        return self._answer(prompt)

    def _answer(self, prompt: str) -> str:
        low = prompt.lower()
        if "attempt completion marker" in low:
            if "previous assistant response contained invalid tool_call" in low:
                return fenced_tool({"name": "attempt_completion", "arguments": {"result": "attempt completion repaired and complete"}})
            return fenced_tool({"name": "attempt_completion", "arguments": {}})

        if "invalid native marker" in low:
            if "previous assistant response contained invalid tool_call" in low:
                content = "<html><body><h1>Fixed native write</h1></body></html>"
                return fenced_tool(
                    {
                        "name": "write_to_file",
                        "arguments": {
                            "path": "fixed.html",
                            "content": content,
                            "line_count": line_count(content),
                        },
                    }
                )
            if "tool result (write_to_file" in low:
                return fenced_tool({"name": "attempt_completion", "arguments": {"result": "invalid native flow recovered"}})
            return fenced_tool({"name": "write_to_file", "arguments": {"path": "fixed.html"}})

        if "line count autofill marker" in low:
            if "tool result (write_to_file" in low:
                return "Line count autofill flow completed."
            content = "first line\nsecond line\nthird line"
            return fenced_tool({"name": "write_to_file", "arguments": {"path": "line_count.txt", "content": content}})

        if "heavy project marker" in low:
            if "tool result (execute_command" in low:
                return fenced_tool({"name": "attempt_completion", "arguments": {"result": "heavy project completed"}})
            if "tool result (read_file" in low and "tool result (list_files" in low:
                return fenced_tool({"name": "execute_command", "arguments": {"command": "python -m py_compile src/app.py"}})
            if low.count("tool result (write_to_file") >= 3:
                return fenced_tool(
                    [
                        {"name": "list_files", "arguments": {"path": "."}},
                        {"name": "read_file", "arguments": {"path": "src/app.py"}},
                    ]
                )
            if "tool result (update_todo_list" in low:
                app_source = big_python_source()
                test_source = (
                    "from src.app import total\n\n"
                    "def test_total():\n"
                    "    assert total() == sum(range(1500))\n"
                )
                config = "[project]\nname = \"heavy-roo-sim\"\nversion = \"0.0.1\"\n"
                return fenced_tool(
                    [
                        {
                            "name": "write_to_file",
                            "arguments": {
                                "path": "src/app.py",
                                "content": app_source,
                                "line_count": line_count(app_source),
                            },
                        },
                        {
                            "name": "write_to_file",
                            "arguments": {
                                "path": "tests/test_app.py",
                                "content": test_source,
                                "line_count": line_count(test_source),
                            },
                        },
                        {
                            "name": "write_to_file",
                            "arguments": {
                                "path": "pyproject.toml",
                                "content": config,
                                "line_count": line_count(config),
                            },
                        },
                    ]
                )
            return fenced_tool(
                {
                    "name": "update_todo_list",
                    "arguments": {
                        "todos": [
                            {"content": "write project files", "status": "in_progress"},
                            {"content": "read and compile", "status": "pending"},
                        ]
                    },
                }
            )

        if "serial mixed marker" in low:
            match = re.search(r"serial mixed marker\s+(\d+)", low)
            number = match.group(1) if match else "x"
            path = f"serial_{number}.txt"
            if "tool result (read_file" in low:
                return f"serial task {number} completed."
            if "tool result (write_to_file" in low:
                return fenced_tool({"name": "read_file", "arguments": {"path": path}})
            content = f"serial-content-{number}"
            return fenced_tool(
                {
                    "name": "write_to_file",
                    "arguments": {"path": path, "content": content, "line_count": line_count(content)},
                }
            )

        if "large final marker" in low:
            return big_markdown()

        if "streaming tool marker" in low:
            content = "streamed native tool content"
            return fenced_tool(
                {
                    "name": "write_to_file",
                    "arguments": {"path": "stream.txt", "content": content, "line_count": line_count(content)},
                }
            )

        return "ok"


def request_json(base_url: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 90) -> tuple[int, Any]:
    url = base_url + path
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        return exc.code, json.loads(raw)


def request_stream_text(base_url: str, payload: dict[str, Any]) -> tuple[int, str, str]:
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    chunks: list[str] = []
    finish_reason = ""
    with urllib.request.urlopen(req, timeout=90) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            choice = event["choices"][0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            if "content" in delta:
                chunks.append(str(delta["content"]))
    return 200, "".join(chunks), finish_reason


def request_stream_tool_calls(base_url: str, payload: dict[str, Any]) -> tuple[int, list[dict[str, Any]], str]:
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    calls_by_index: dict[int, dict[str, Any]] = {}
    finish_reason = ""
    with urllib.request.urlopen(req, timeout=90) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise AssertionError(event["error"])
            choice = event["choices"][0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            for tool_delta in delta.get("tool_calls") or []:
                index = int(tool_delta.get("index", 0))
                current = calls_by_index.setdefault(
                    index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if tool_delta.get("id"):
                    current["id"] = tool_delta["id"]
                if tool_delta.get("type"):
                    current["type"] = tool_delta["type"]
                function_delta = tool_delta.get("function") or {}
                if function_delta.get("name"):
                    current["function"]["name"] = function_delta["name"]
                if "arguments" in function_delta:
                    current["function"]["arguments"] += str(function_delta["arguments"])
    return 200, [calls_by_index[index] for index in sorted(calls_by_index)], finish_reason


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    logging.info("ASSERT ok: %s", message)


def tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "write_to_file",
                "description": "Write a complete file in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "line_count": {"type": "integer"},
                    },
                    "required": ["path", "content", "line_count"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a workspace file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "filepath": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files under a workspace path.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_command",
                "description": "Run an allowed workspace command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_todo_list",
                "description": "Update task progress.",
                "parameters": {
                    "type": "object",
                    "properties": {"todos": {"type": "array"}},
                    "required": ["todos"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "attempt_completion",
                "description": "Finish the task with a final result.",
                "parameters": {
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                    "required": ["result"],
                },
            },
        },
    ]


def safe_path(workspace: Path, filepath: str) -> Path:
    target = (workspace / filepath).resolve()
    workspace_root = workspace.resolve()
    if target != workspace_root and not str(target).startswith(str(workspace_root) + "/"):
        raise ValueError(f"unsafe filepath: {filepath}")
    return target


def execute_tool(workspace: Path, worker: HeavyRooWorker, name: str, args: dict[str, Any]) -> str:
    if name == "write_to_file":
        for key in ("path", "content", "line_count"):
            if key not in args:
                raise AssertionError(f"write_to_file missing {key}: {args}")
        content = str(args["content"])
        expected_line_count = line_count(content)
        if int(args["line_count"]) != expected_line_count:
            raise AssertionError(f"bad line_count for {args['path']}: {args['line_count']} != {expected_line_count}")
        path = safe_path(workspace, str(args["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"created {args['path']} chars={len(content)} lines={expected_line_count}"
    if name == "read_file":
        return safe_path(workspace, str(args.get("path") or args.get("filepath"))).read_text(encoding="utf-8")
    if name == "list_files":
        root = safe_path(workspace, str(args.get("path") or "."))
        if root.is_file():
            return root.name
        files = sorted(str(path.relative_to(workspace)) for path in root.rglob("*") if path.is_file())
        return "\n".join(files)
    if name == "execute_command":
        command = str(args.get("command", ""))
        allowed = {"python -m py_compile src/app.py"}
        if command not in allowed:
            raise ValueError(f"unexpected command: {command}")
        completed = subprocess.run(
            [sys.executable, "-m", "py_compile", "src/app.py"],
            cwd=workspace,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        return f"exit={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}"
    if name == "update_todo_list":
        worker.todo_updates.append(args.get("todos"))
        return f"todo_count={len(args.get('todos') or [])}"
    if name == "attempt_completion":
        result = str(args.get("result") or "")
        if not result:
            raise AssertionError("attempt_completion missing result")
        return result
    raise ValueError(f"unknown tool: {name}")


def run_agent(
    base_url: str,
    worker: HeavyRooWorker,
    workspace: Path,
    model: str,
    user_content: str,
    max_steps: int = 10,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    executed: list[dict[str, Any]] = []
    for _step in range(max_steps):
        status, body = request_json(
            base_url,
            "/chat/completions",
            {"model": model, "messages": messages, "tools": tool_specs(), "timeout": 60},
        )
        assert_true(status == 200, "agent step returns HTTP 200")
        choice = body["choices"][0]
        message = choice["message"]
        if choice["finish_reason"] != "tool_calls":
            return str(message.get("content") or ""), messages, executed
        tool_calls = message["tool_calls"]
        messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        for call in tool_calls:
            function = call["function"]
            name = function["name"]
            args = json.loads(function.get("arguments") or "{}")
            result = execute_tool(workspace, worker, name, args)
            executed.append({"name": name, "args": args, "result": result})
            if name == "attempt_completion":
                return result, messages, executed
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": result,
                }
            )
    raise AssertionError("agent did not finish within max_steps")


def wait_until_ready(base_url: str) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            status, body = request_json(base_url, "/models", timeout=3)
            if status == 200 and body.get("data"):
                return
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError("adapter did not start")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    adapter_module.BROWSER_BUSY_TIMEOUT_SEC = 0.5
    port = free_port()
    base_url = f"http://127.0.0.1:{port}/v1"
    worker = HeavyRooWorker()
    adapter = OpenAIAdapter(worker, port=port)
    adapter.start()
    wait_until_ready(base_url)
    try:
        with tempfile.TemporaryDirectory(prefix="deepbill-roo-heavy-") as temp_dir:
            workspace = Path(temp_dir)

            calls_before = len(worker.calls)
            final, _messages, executed = run_agent(
                base_url,
                worker,
                workspace,
                "deepbill-auto",
                "Heavy project marker: create a small Python project, read files, list files, compile it, then finish.",
            )
            project_calls = [call for call in worker.calls[calls_before:] if call["stage"] == "ask_text"]
            assert_true("heavy project completed" in final, "heavy project reaches attempt_completion")
            assert_true(any(call["use_reasoning"] for call in project_calls), "auto model uses reasoning on complex task")
            assert_true((workspace / "src/app.py").stat().st_size > 10000, "large Python file was written")
            assert_true((workspace / "tests/test_app.py").exists(), "test file was written")
            assert_true({item["name"] for item in executed} >= {"update_todo_list", "write_to_file", "read_file", "list_files", "execute_command", "attempt_completion"}, "heavy flow uses all expected tools")
            assert_true(worker.todo_updates and len(worker.todo_updates[-1]) == 2, "todo update payload is preserved")

            calls_before = len(worker.calls)
            final, _messages, executed = run_agent(
                base_url,
                worker,
                workspace,
                "deepseek-chat",
                "Invalid native marker: trigger missing content repair before Roo sees write_to_file.",
            )
            repair_calls = [call for call in worker.calls[calls_before:] if call["stage"] == "ask_text"]
            assert_true("invalid native flow recovered" in final, "invalid native tool call is repaired")
            assert_true((workspace / "fixed.html").read_text(encoding="utf-8").startswith("<html>"), "repaired write created the file")
            assert_true(any(str(call.get("request_id", "")).endswith("-tool-repair") for call in repair_calls), "repair request is visible")
            assert_true(executed[0]["args"]["content"], "Roo executor received content after repair")

            calls_before = len(worker.calls)
            final, _messages, executed = run_agent(
                base_url,
                worker,
                workspace,
                "deepseek-chat",
                "Line count autofill marker: omit line_count but include content.",
            )
            autofill_calls = [call for call in worker.calls[calls_before:] if call["stage"] == "ask_text"]
            assert_true("completed" in final.lower(), "line_count autofill task finishes")
            assert_true(executed[0]["args"]["line_count"] == 3, "adapter computes missing line_count from content")
            assert_true(not any(str(call.get("request_id", "")).endswith("-tool-repair") for call in autofill_calls), "line_count autofill avoids repair round-trip")

            calls_before = len(worker.calls)
            final, _messages, executed = run_agent(
                base_url,
                worker,
                workspace,
                "deepseek-chat",
                "Attempt completion marker: return invalid attempt_completion first.",
            )
            attempt_calls_after = len(worker.calls)
            time.sleep(0.2)
            assert_true(final == "attempt completion repaired and complete", "attempt_completion required result is repaired")
            assert_true([item["name"] for item in executed] == ["attempt_completion"], "attempt_completion stops the Roo loop")
            assert_true(len(worker.calls) == attempt_calls_after, "no extra model request happens after attempt_completion")
            assert_true(
                any(str(call.get("request_id", "")).endswith("-tool-repair") for call in worker.calls[calls_before:]),
                "invalid attempt_completion triggers repair before native call",
            )

            for index in range(5):
                calls_before = len(worker.calls)
                model = "deepbill-deepthink" if index % 2 else "deepseek-chat"
                final, _messages, executed = run_agent(
                    base_url,
                    worker,
                    workspace,
                    model,
                    f"Serial mixed marker {index + 1}: write a small file and read it back.",
                )
                recent_calls = [call for call in worker.calls[calls_before:] if call["stage"] == "ask_text"]
                expected_path = workspace / f"serial_{index + 1}.txt"
                assert_true(f"serial task {index + 1} completed" in final, f"serial request {index + 1} finishes")
                assert_true(expected_path.read_text(encoding="utf-8") == f"serial-content-{index + 1}", f"serial file {index + 1} content is correct")
                assert_true([item["name"] for item in executed] == ["write_to_file", "read_file"], f"serial request {index + 1} uses write/read only")
                if model == "deepbill-deepthink":
                    assert_true(any(call["use_reasoning"] for call in recent_calls), f"serial request {index + 1} uses reasoning")
                else:
                    assert_true(not any(call["use_reasoning"] for call in recent_calls), f"serial request {index + 1} stays non-reasoning")

            status, body = request_json(
                base_url,
                "/chat/completions",
                {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "Large final marker: produce a long direct response."}],
                    "timeout": 60,
                },
            )
            final_text = body["choices"][0]["message"]["content"]
            assert_true(status == 200, "large direct response returns HTTP 200")
            assert_true(body["choices"][0]["finish_reason"] == "stop", "large direct response stops normally")
            assert_true(len(final_text) > 70000 and "heavy-response-marker-2199" in final_text, "large direct response is preserved")

            status, streamed, finish_reason = request_stream_text(
                base_url,
                {
                    "model": "deepseek-chat",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Large final marker: stream a long direct response."}],
                    "timeout": 60,
                },
            )
            assert_true(status == 200, "streamed large response returns HTTP 200")
            assert_true(finish_reason == "stop", "streamed large response finishes with stop")
            assert_true(streamed == final_text, "streaming preserves complete long answer")

            status, stream_tool_calls, finish_reason = request_stream_tool_calls(
                base_url,
                {
                    "model": "deepseek-chat",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Streaming tool marker: write a file through streamed request."}],
                    "tools": tool_specs(),
                    "timeout": 60,
                },
            )
            assert_true(status == 200, "streamed native tool request returns HTTP 200")
            assert_true(finish_reason == "tool_calls", "streamed native tool request finishes with tool_calls")
            assert_true(len(stream_tool_calls) == 1, "streamed native tool request returns one tool call")
            stream_function = stream_tool_calls[0]["function"]
            stream_args = json.loads(stream_function["arguments"])
            execute_tool(workspace, worker, stream_function["name"], stream_args)
            assert_true((workspace / "stream.txt").read_text(encoding="utf-8") == "streamed native tool content", "streamed native tool call can be executed")

        logging.info("roocode_heavy_simulation_tests: ok")
    finally:
        adapter.stop()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
End-to-end HTTP contract tests for openai_adapter.py.

These tests run a real Flask HTTP server and talk to it through OpenAI-shaped
requests. The worker is deterministic on purpose: it exercises the adapter
contract without touching the user's workspace or depending on a logged-in
browser session.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Any

from openai_adapter import OpenAIAdapter


LOG_PATH = Path(__file__).resolve().parent / "adapter_test_run.log"
BASE_URL = "http://127.0.0.1:18080/v1"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


class DiagnosticWorker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.prompts: list[str] = []
        self._fail_once_markers: set[str] = set()

    def status(self):
        return True, True, None

    def new_chat(self, timeout: int = 30) -> None:
        self.calls.append({"stage": "worker.new_chat", "timeout": timeout})

    def ask_text(self, prompt: str, timeout: int = 180) -> str:
        self.calls.append({"stage": "worker.ask_text", "timeout": timeout, "prompt_chars": len(prompt)})
        self.prompts.append(prompt)
        if "retry marker" in prompt.lower() and "retry marker" not in self._fail_once_markers:
            self._fail_once_markers.add("retry marker")
            raise TimeoutError("Таймаут ожидания ответа DeepSeek. Последний фрагмент: ''")
        return self._answer(prompt)

    def ask_text_stream(self, prompt: str, timeout: int = 180):
        self.calls.append({"stage": "worker.ask_text_stream", "timeout": timeout, "prompt_chars": len(prompt)})
        self.prompts.append(prompt)
        answer = self._answer(prompt)
        for idx in range(0, len(answer), 9):
            yield answer[idx : idx + 9]

    def _answer(self, prompt: str) -> str:
        low = prompt.lower()
        if "agent scenario marker" in low and "tool result (read_file" in low:
            return "Готово: создал файл, прочитал его и подтвердил содержимое."
        if "agent scenario marker" in low and "tool result (create_new_file" in low:
            return "tool_call Copy Download {'name': 'read_file', 'arguments': {'filepath': 'index.html'}}"
        if "agent scenario marker" in low:
            return "tool_call Copy Download {'name': 'create_new_file', 'arguments': {'filepath': 'index.html', 'contents': '<h1>Hello</h1>'}}"
        if "edit chain marker" in low and "tool result (read_file" in low:
            return "Готово: создал файл, изменил текст и проверил чтением."
        if "edit chain marker" in low and "tool result (edit_existing_file" in low:
            return 'tool_call Copy Download {"name":"read_file","arguments":{"filepath":"note.txt"}}'
        if "edit chain marker" in low and "tool result (create_new_file" in low:
            return 'tool_call Copy Download {"name":"edit_existing_file","arguments":{"filepath":"note.txt","changes":"hello orange"}}'
        if "edit chain marker" in low:
            return 'tool_call Copy Download {"name":"create_new_file","arguments":{"filepath":"note.txt","contents":"hello blue"}}'
        read_result_count = low.count("tool result (read_file")
        if "read content chain marker" in low and read_result_count >= 2 and "beta secret" in low:
            return "Готово: после второго чтения вижу beta secret."
        if "read content chain marker" in low and "tool result (edit_existing_file" in low:
            return 'tool_call Copy Download {"name":"read_file","arguments":{"filepath":"note.txt"}}'
        if "read content chain marker" in low and read_result_count >= 1 and "alpha secret" in low:
            return 'tool_call Copy Download {"name":"edit_existing_file","arguments":{"filepath":"note.txt","changes":"beta secret"}}'
        if "read content chain marker" in low:
            return 'tool_call Copy Download {"name":"read_file","arguments":{"filepath":"note.txt"}}'
        if "python dict tool marker" in low:
            return (
                "tool_call\nCopy\nDownload\n"
                "{'name': 'create_new_file', 'arguments': {'filepath': 'index.html', "
                "'contents': '<html><body><h1>Hello</h1></body></html>'}}"
            )
        if "malformed code marker" in low:
            return (
                'tempfile\nimport os\n\ntool_call Copy Download {"name":"create_new_file",'
                '"arguments":{"filepath":"voice_assistant.py","contents":"#!/usr/bin/env python3\\n"""\\n'
                'Voice Assistant\\n"""\\nimport os\\nprint("hi")\\n"}}'
            )
        if "terminal tool marker" in low:
            return (
                'tool_call Copy Download {"name":"run_terminal_command",'
                '"arguments":{"command":"python -c "print(123)""}}'
            )
        if "dirty tool marker" in low:
            return (
                'tool_call Copy Download {"name":"create_new_file",'
                '"arguments":{"filepath":"test.py","contents":"print(\\"hello\\")"}}'
            )
        if "prose no native tools marker" in low:
            return (
                "Для изменения цвета отредактирую style.css.\n\n"
                'tool_call Copy Download {"name":"edit_existing_file",'
                '"arguments":{"filepath":"style.css","changes":"#header {\\n  background: #e67e22;\\n}\\na {\\n  color: #e67e22;\\n}"}}'
            )
        if "stream no native tools marker" in low:
            return (
                "Сейчас нужен инструмент.\n\n"
                'tool_call\nCopy\nDownload\n{"name":"edit_existing_file",'
                '"arguments":{"filepath":"style.css","changes":"body {\\n  color: #e67e22;\\n}"}}'
            )
        if "glued ui marker" in low:
            return (
                "text Copy Downloadtool_call Copy Download "
                '{"name":"create_new_file","arguments":{"filepath":"calculator.html",'
                '"contents":"<button>Copy Download should stay in file text</button>"}}'
            )
        if "multi tool array marker" in low:
            return (
                'tool_call Copy Download [{"name":"create_new_file","arguments":{"filepath":"a.txt","contents":"A"}},'
                '{"name":"edit_existing_file","arguments":{"filepath":"a.txt","changes":"AA"}}]'
            )
        if "stream tool marker" in low:
            return '```tool_call\n{"name":"read_file","arguments":{"filepath":"test.py"}}\n```'
        if "tool result" in low or "created" in low:
            return "Готово: результат инструмента учтен."
        if "content array marker" in low:
            return "Контент-массив принят."
        return "Короткий ответ: 2+2=4."


def request_json(path: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    url = BASE_URL + path
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    logging.info("HTTP request path=%s payload=%s", path, payload)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
        logging.info("HTTP response path=%s status=%s body=%s", path, resp.status, raw[:1000])
        return resp.status, json.loads(raw)


def request_sse(path: str, payload: dict[str, Any]) -> tuple[int, list[str]]:
    url = BASE_URL + path
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    logging.info("SSE request path=%s payload=%s", path, payload)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
        events = [chunk for chunk in raw.split("\n\n") if chunk.strip()]
        logging.info("SSE response path=%s status=%s events=%s raw=%s", path, resp.status, len(events), raw[:1600])
        return resp.status, events


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    logging.info("ASSERT ok: %s", message)


def main() -> None:
    setup_logging()
    logging.info("=== adapter route tests started ===")
    worker = DiagnosticWorker()
    adapter = OpenAIAdapter(worker, port=18080)
    adapter.start()
    time.sleep(0.8)

    status, models = request_json("/models")
    assert_true(status == 200 and models["data"], "models endpoint returns a model list")

    status, plain = request_json(
        "/chat/completions",
        {"messages": [{"role": "user", "content": "маленький запрос: 2+2?"}], "timeout": 30},
    )
    assert_true(status == 200, "plain chat returns HTTP 200")
    assert_true(plain["choices"][0]["finish_reason"] == "stop", "plain chat finish_reason=stop")

    calls_before_retry = len(worker.calls)
    status, retry_plain = request_json(
        "/chat/completions",
        {"messages": [{"role": "user", "content": "retry marker: маленький запрос"}], "timeout": 30},
    )
    retry_calls = worker.calls[calls_before_retry:]
    assert_true(status == 200, "retryable DeepSeek timeout recovers with HTTP 200")
    assert_true(retry_plain["choices"][0]["finish_reason"] == "stop", "retryable timeout returns final answer")
    assert_true(
        len([call for call in retry_calls if call["stage"] == "worker.ask_text"]) == 2,
        "retryable timeout repeats the DeepSeek request once",
    )
    assert_true(
        any(call["stage"] == "worker.new_chat" for call in retry_calls),
        "retryable timeout opens a fresh DeepSeek chat before retry",
    )

    status, dirty_tool = request_json(
        "/chat/completions",
        {
            "messages": [{"role": "user", "content": "dirty tool marker: создай маленький test.py"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "create_new_file",
                        "description": "Create a small file.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filepath": {"type": "string"},
                                "contents": {"type": "string"},
                            },
                            "required": ["filepath", "contents"],
                        },
                    },
                }
            ],
            "timeout": 30,
        },
    )
    choice = dirty_tool["choices"][0]
    assert_true(choice["finish_reason"] == "tool_calls", "dirty Continue-style tool marker becomes native tool_calls")
    assert_true(choice["message"]["tool_calls"][0]["function"]["name"] == "create_new_file", "tool name is preserved")

    status, malformed_code = request_json(
        "/chat/completions",
        {
            "messages": [{"role": "user", "content": "malformed code marker: создай файл с python кодом"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "create_new_file",
                        "description": "Create a code file.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filepath": {"type": "string"},
                                "contents": {"type": "string"},
                            },
                            "required": ["filepath", "contents"],
                        },
                    },
                }
            ],
            "timeout": 30,
        },
    )
    malformed_choice = malformed_code["choices"][0]
    malformed_call = malformed_choice["message"]["tool_calls"][0]
    malformed_args = json.loads(malformed_call["function"]["arguments"])
    assert_true(status == 200, "malformed code tool marker returns HTTP 200")
    assert_true(malformed_choice["finish_reason"] == "tool_calls", "malformed code marker becomes native tool_calls")
    assert_true(malformed_choice["message"]["content"] is None, "malformed code marker does not leak as assistant content")
    assert_true(malformed_call["function"]["name"] == "create_new_file", "malformed code marker keeps tool name")
    assert_true(malformed_args["filepath"] == "voice_assistant.py", "malformed code marker keeps filepath")
    assert_true('print("hi")' in malformed_args["contents"], "malformed code marker keeps code contents")
    assert_true(
        "tempfile" not in json.dumps(malformed_choice, ensure_ascii=False)
        and "Copy" not in json.dumps(malformed_choice, ensure_ascii=False)
        and "Download" not in json.dumps(malformed_choice, ensure_ascii=False),
        "malformed code UI/code prefix does not leak into response",
    )

    status, terminal_tool = request_json(
        "/chat/completions",
        {
            "messages": [{"role": "user", "content": "terminal tool marker: проверь терминал"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "run_terminal_command",
                        "description": "Run a terminal command.",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                }
            ],
            "timeout": 30,
        },
    )
    terminal_choice = terminal_tool["choices"][0]
    terminal_call = terminal_choice["message"]["tool_calls"][0]
    terminal_args = json.loads(terminal_call["function"]["arguments"])
    assert_true(status == 200, "terminal tool marker returns HTTP 200")
    assert_true(terminal_choice["finish_reason"] == "tool_calls", "terminal marker becomes native tool_calls")
    assert_true(terminal_call["function"]["name"] == "run_terminal_command", "terminal tool name is preserved")
    assert_true("print(123)" in terminal_args["command"], "terminal command argument is preserved")

    status, python_dict_tool = request_json(
        "/chat/completions",
        {
            "messages": [{"role": "user", "content": "python dict tool marker: создай маленький index.html"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "create_new_file",
                        "description": "Create a small file.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filepath": {"type": "string"},
                                "contents": {"type": "string"},
                            },
                            "required": ["filepath", "contents"],
                        },
                    },
                }
            ],
            "timeout": 30,
        },
    )
    py_choice = python_dict_tool["choices"][0]
    assert_true(status == 200, "Python-literal tool marker returns HTTP 200")
    assert_true(py_choice["finish_reason"] == "tool_calls", "Python-literal tool marker becomes native tool_calls")
    assert_true(
        py_choice["message"]["tool_calls"][0]["function"]["name"] == "create_new_file",
        "Python-literal tool name is preserved",
    )

    status, prose_no_tools = request_json(
        "/chat/completions",
        {
            "messages": [{"role": "user", "content": "prose no native tools marker: сделай style.css оранжевым"}],
            "timeout": 30,
        },
    )
    prose_choice = prose_no_tools["choices"][0]
    assert_true(status == 200, "prose + tool_call without native tools returns HTTP 200")
    assert_true(prose_choice["finish_reason"] == "tool_calls", "explicit tool_call is parsed even without request tools")
    assert_true(
        prose_choice["message"]["tool_calls"][0]["function"]["name"] == "edit_existing_file",
        "explicit no-tools tool_call keeps edit_existing_file name",
    )
    assert_true(
        "Copy" not in json.dumps(prose_choice, ensure_ascii=False)
        and "Download" not in json.dumps(prose_choice, ensure_ascii=False),
        "Copy/Download UI words do not leak into parsed response",
    )

    status, glued_ui = request_json(
        "/chat/completions",
        {
            "messages": [{"role": "user", "content": "glued ui marker: создай маленький калькулятор"}],
            "timeout": 30,
        },
    )
    glued_choice = glued_ui["choices"][0]
    glued_call = glued_choice["message"]["tool_calls"][0]
    glued_args = json.loads(glued_call["function"]["arguments"])
    assert_true(status == 200, "glued UI marker returns HTTP 200")
    assert_true(glued_choice["finish_reason"] == "tool_calls", "glued Downloadtool_call marker becomes tool_calls")
    assert_true(glued_call["function"]["name"] == "create_new_file", "glued UI marker tool name is preserved")
    assert_true(glued_args["filepath"] == "calculator.html", "glued UI marker arguments are preserved")
    assert_true("Copy Download should stay" in glued_args["contents"], "Copy/Download inside file content is preserved")

    status, multi_tool = request_json(
        "/chat/completions",
        {
            "messages": [{"role": "user", "content": "multi tool array marker: создай и отредактируй маленький файл"}],
            "timeout": 30,
        },
    )
    multi_choice = multi_tool["choices"][0]
    assert_true(status == 200, "multi-tool array marker returns HTTP 200")
    assert_true(multi_choice["finish_reason"] == "tool_calls", "multi-tool array marker becomes tool_calls")
    assert_true(len(multi_choice["message"]["tool_calls"]) == 2, "multi-tool array returns two calls")

    status, events = request_sse(
        "/chat/completions",
        {"stream": True, "messages": [{"role": "user", "content": "маленький stream запрос"}], "timeout": 30},
    )
    assert_true(status == 200, "text streaming returns HTTP 200")
    assert_true(any("data: [DONE]" in event for event in events), "text streaming terminates with [DONE]")

    status, events = request_sse(
        "/chat/completions",
        {
            "stream": True,
            "messages": [{"role": "user", "content": "stream tool marker: прочитай маленький файл"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a small file.",
                        "parameters": {
                            "type": "object",
                            "properties": {"filepath": {"type": "string"}},
                            "required": ["filepath"],
                        },
                    },
                }
            ],
            "timeout": 30,
        },
    )
    assert_true(status == 200, "tool streaming returns HTTP 200")
    assert_true(any('"tool_calls"' in event for event in events), "tool streaming emits tool_calls deltas")
    assert_true(any('"finish_reason":"tool_calls"' in event for event in events), "tool streaming finish_reason=tool_calls")

    status, events = request_sse(
        "/chat/completions",
        {
            "stream": True,
            "messages": [{"role": "user", "content": "stream no native tools marker: поменяй style.css"}],
            "timeout": 30,
        },
    )
    raw_events = "\n".join(events)
    assert_true(status == 200, "stream no-tools tool marker returns HTTP 200")
    assert_true('"tool_calls"' in raw_events, "stream no-tools tool marker emits tool_calls")
    assert_true('"finish_reason":"tool_calls"' in raw_events, "stream no-tools tool marker finish_reason=tool_calls")
    assert_true("Copy" not in raw_events and "Download" not in raw_events, "stream no-tools marker strips Copy/Download")

    status, tool_result = request_json(
        "/chat/completions",
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_test",
                            "type": "function",
                            "function": {
                                "name": "create_new_file",
                                "arguments": '{"filepath":"test.py","contents":"print(\\"hello\\")"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_test", "content": "created"},
                {"role": "user", "content": "tool result: подтверди одной короткой фразой"},
            ],
            "timeout": 30,
        },
    )
    assert_true(status == 200, "tool result follow-up returns HTTP 200")
    assert_true(tool_result["choices"][0]["finish_reason"] == "stop", "tool result follow-up finishes normally")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_new_file",
                "description": "Create a small file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string"},
                        "contents": {"type": "string"},
                    },
                    "required": ["filepath", "contents"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_existing_file",
                "description": "Edit a small file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string"},
                        "changes": {"type": "string"},
                    },
                    "required": ["filepath", "changes"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a small file.",
                "parameters": {
                    "type": "object",
                    "properties": {"filepath": {"type": "string"}},
                    "required": ["filepath"],
                },
            },
        },
    ]
    initial_user = {
        "role": "user",
        "content": "agent scenario marker: создай маленький index.html и проверь содержимое",
    }
    status, agent_step_1 = request_json(
        "/chat/completions",
        {"messages": [initial_user], "tools": tools, "timeout": 30},
    )
    step_1_choice = agent_step_1["choices"][0]
    assert_true(step_1_choice["finish_reason"] == "tool_calls", "agent step 1 requests a tool")
    first_tool_call = step_1_choice["message"]["tool_calls"][0]
    assert_true(first_tool_call["function"]["name"] == "create_new_file", "agent step 1 creates a file")

    status, agent_step_2 = request_json(
        "/chat/completions",
        {
            "messages": [
                initial_user,
                {"role": "assistant", "content": None, "tool_calls": [first_tool_call]},
                {
                    "role": "tool",
                    "tool_call_id": first_tool_call["id"],
                    "content": "created index.html",
                },
            ],
            "tools": tools,
            "timeout": 30,
        },
    )
    step_2_choice = agent_step_2["choices"][0]
    assert_true(step_2_choice["finish_reason"] == "tool_calls", "agent step 2 can request another tool")
    second_tool_call = step_2_choice["message"]["tool_calls"][0]
    assert_true(second_tool_call["function"]["name"] == "read_file", "agent step 2 reads the file")

    status, agent_step_3 = request_json(
        "/chat/completions",
        {
            "messages": [
                initial_user,
                {"role": "assistant", "content": None, "tool_calls": [first_tool_call]},
                {
                    "role": "tool",
                    "tool_call_id": first_tool_call["id"],
                    "content": "created index.html",
                },
                {"role": "assistant", "content": None, "tool_calls": [second_tool_call]},
                {
                    "role": "tool",
                    "tool_call_id": second_tool_call["id"],
                    "content": "<h1>Hello</h1>",
                },
            ],
            "tools": tools,
            "timeout": 30,
        },
    )
    assert_true(status == 200, "agent step 3 returns HTTP 200")
    assert_true(agent_step_3["choices"][0]["finish_reason"] == "stop", "agent step 3 finishes after multiple tools")

    edit_user = {
        "role": "user",
        "content": "edit chain marker: создай маленький note.txt, отредактируй и прочитай",
    }
    edit_messages = [edit_user]
    expected_tools = ["create_new_file", "edit_existing_file", "read_file"]
    seen_tools: list[str] = []
    tool_results = {
        "create_new_file": "created note.txt",
        "edit_existing_file": "edited note.txt",
        "read_file": "hello orange",
    }
    final_answer = ""
    for step in range(1, 5):
        status, edit_step = request_json(
            "/chat/completions",
            {"messages": edit_messages, "tools": tools, "timeout": 30},
        )
        assert_true(status == 200, f"edit chain step {step} returns HTTP 200")
        edit_choice = edit_step["choices"][0]
        if edit_choice["finish_reason"] == "tool_calls":
            tool_call = edit_choice["message"]["tool_calls"][0]
            tool_name = tool_call["function"]["name"]
            seen_tools.append(tool_name)
            edit_messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
            edit_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_results.get(tool_name, "ok"),
                }
            )
            continue
        final_answer = edit_choice["message"].get("content", "")
        break
    assert_true(seen_tools == expected_tools, "edit chain uses create, edit, read in order")
    assert_true(bool(final_answer), "edit chain finishes with final answer")

    read_content_user = {
        "role": "user",
        "content": "read content chain marker: прочитай note.txt, отредактируй после чтения и прочитай снова",
    }
    read_content_messages = [read_content_user]
    read_content_results: list[tuple[str, Any]] = [
        ("read_file", [{"type": "text", "text": "alpha secret\nline two"}]),
        ("edit_existing_file", {"text": "edited note.txt"}),
        ("read_file", {"content": [{"type": "text", "text": "beta secret\nline two"}]}),
    ]
    read_seen_tools: list[str] = []
    read_final_answer = ""
    prompts_before_read_chain = len(worker.prompts)
    for step in range(1, 5):
        status, read_step = request_json(
            "/chat/completions",
            {"messages": read_content_messages, "tools": tools, "timeout": 30},
        )
        assert_true(status == 200, f"read/edit/read chain step {step} returns HTTP 200")
        read_choice = read_step["choices"][0]
        if read_choice["finish_reason"] == "tool_calls":
            tool_call = read_choice["message"]["tool_calls"][0]
            tool_name = tool_call["function"]["name"]
            read_seen_tools.append(tool_name)
            expected_name, result_content = read_content_results[len(read_seen_tools) - 1]
            assert_true(tool_name == expected_name, f"read/edit/read chain step {step} requested {expected_name}")
            read_content_messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
            read_content_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_name,
                    "content": result_content,
                }
            )
            continue
        read_final_answer = read_choice["message"].get("content", "")
        break
    read_chain_prompts = worker.prompts[prompts_before_read_chain:]
    assert_true(read_seen_tools == ["read_file", "edit_existing_file", "read_file"], "read/edit/read chain uses expected tools")
    assert_true(any("alpha secret" in prompt for prompt in read_chain_prompts), "first read_file contents reach the model prompt")
    assert_true(any("beta secret" in prompt for prompt in read_chain_prompts), "second read_file contents reach the model prompt")
    assert_true("beta secret" in read_final_answer, "final answer is based on second read_file contents")

    status, content_array = request_json(
        "/chat/completions",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "content array marker"},
                        {"type": "text", "text": "маленький запрос"},
                    ],
                }
            ],
            "timeout": 30,
        },
    )
    assert_true(status == 200, "OpenAI content-array messages are accepted")
    assert_true(content_array["choices"][0]["finish_reason"] == "stop", "content-array chat finishes normally")

    logging.info("worker calls=%s", worker.calls)
    new_chat_calls = [call for call in worker.calls if call["stage"] == "worker.new_chat"]
    ask_calls = [call for call in worker.calls if call["stage"].startswith("worker.ask_text")]
    assert_true(len(new_chat_calls) < len(ask_calls), "new DeepSeek chats are not created for every agent step")
    logging.info("=== adapter route tests passed ===")


if __name__ == "__main__":
    main()

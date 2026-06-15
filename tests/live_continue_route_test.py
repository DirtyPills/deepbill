#!/usr/bin/env python3
"""
Continue-like live route test for the running DeepBill adapter.

The script talks to an already running OpenAI-compatible adapter, receives
native tool_calls, executes a tiny local tool sandbox, and feeds tool results
back just like an agent client would.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from _path import PROJECT_ROOT


LOG_PATH = PROJECT_ROOT / "logs" / "live_continue_route_test.log"
WORK_DIR = Path("/tmp/deepbill_live_continue_tools")


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=raw, headers={"Content-Type": "application/json"})
    logging.info("HTTP request path=%s payload=%s", path, payload)
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout + 60) as resp:
        body = resp.read().decode("utf-8")
        logging.info("HTTP response path=%s status=%s elapsed=%.1fs body=%s", path, resp.status, time.monotonic() - started, body[:2500])
        if resp.status != 200:
            raise RuntimeError(f"Unexpected HTTP status {resp.status}")
        return json.loads(body)


def post_sse(base_url: str, path: str, payload: dict[str, Any], timeout: int) -> list[str]:
    url = base_url.rstrip("/") + path
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=raw, headers={"Content-Type": "application/json"})
    logging.info("SSE request path=%s payload=%s", path, payload)
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout + 60) as resp:
        body = resp.read().decode("utf-8")
        events = [event for event in body.split("\n\n") if event.strip()]
        logging.info("SSE response path=%s status=%s elapsed=%.1fs events=%s raw=%s", path, resp.status, time.monotonic() - started, len(events), body[:2500])
        if resp.status != 200:
            raise RuntimeError(f"Unexpected SSE HTTP status {resp.status}")
        return events


def get_health(base_url: str) -> dict[str, Any]:
    raw = urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=10).read().decode("utf-8")
    logging.info("health=%s", raw)
    health = json.loads(raw)
    if not isinstance(health, dict):
        raise RuntimeError(f"Unexpected health payload: {raw}")
    return health


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    logging.info("ASSERT ok: %s", message)


def continue_total(health: dict[str, Any]) -> int:
    diagnostics = health.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return 0
    return int(diagnostics.get("total_continue_clicks") or 0)


def require_clean_message(message: dict[str, Any], label: str) -> None:
    raw = json.dumps(message, ensure_ascii=False)
    ui_fragments = ("tool_call Copy Download", "Copy\nDownload", "Downloadtool_call")
    require(not any(fragment in raw for fragment in ui_fragments), f"{label} does not leak DeepSeek UI words")
    content = str(message.get("content") or "")
    reasoning_fragments = (
        "We need to",
        "The user said",
        "The user asks",
        "So just output",
        "Now we need",
        "Since no additional tool",
        "The assistant already",
        "I need to",
        "Need to respond",
        "Нужно ответить",
        "Теперь нужно",
        "Мы получили результат",
        "Мы получили все результаты",
        "Мы получили",
        "Мы закончили",
        "Все операции выполнены",
        "Требуется коротко",
        "Ответ будет",
        "Ответ:",
        "По инструкции",
        "Инструменты не нужны",
        "Обычным текстом",
    )
    require(
        not any(fragment.lower() in content.lower() for fragment in reasoning_fragments),
        f"{label} does not leak DeepThink reasoning",
    )
    require_no_duplicate_tool_calls(message, label)


def require_no_duplicate_tool_calls(message: dict[str, Any], label: str) -> None:
    calls = message.get("tool_calls") or []
    keys: list[tuple[str, str]] = []
    for call in calls:
        function = call.get("function") or {}
        keys.append((str(function.get("name") or ""), str(function.get("arguments") or "")))
    require(len(keys) == len(set(keys)), f"{label} has no duplicate identical tool calls")


def safe_path(filepath: str) -> Path:
    rel = Path(filepath or "agent_test.html")
    if rel.is_absolute():
        rel = Path(rel.name)
    target = (WORK_DIR / rel).resolve()
    if WORK_DIR.resolve() not in target.parents and target != WORK_DIR.resolve():
        raise ValueError(f"Unsafe tool path: {filepath}")
    return target


def execute_tool(call: dict[str, Any]) -> str:
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    raw_args = function.get("arguments") or "{}"
    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    logging.info("TOOL execute name=%s args=%s", name, args)

    if name == "create_new_file":
        target = safe_path(str(args.get("filepath") or "agent_test.html"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(args.get("contents") or ""), encoding="utf-8")
        result = f"created {target.relative_to(WORK_DIR)} chars={target.stat().st_size}"
    elif name == "edit_existing_file":
        target = safe_path(str(args.get("filepath") or "agent_test.html"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(args.get("changes") or args.get("contents") or ""), encoding="utf-8")
        result = f"edited {target.relative_to(WORK_DIR)} chars={target.stat().st_size}"
    elif name == "read_file":
        target = safe_path(str(args.get("filepath") or "agent_test.html"))
        result = target.read_text(encoding="utf-8")
    elif name == "list_files":
        result = "\n".join(str(path.relative_to(WORK_DIR)) for path in sorted(WORK_DIR.rglob("*")) if path.is_file())
    elif name == "run_terminal_command":
        command = str(args.get("command") or "").strip()
        allowed = {'python -c "print(123)"', "python3 -c \"print(123)\""}
        if command not in allowed:
            raise ValueError(f"Unsafe terminal command in live test: {command}")
        completed = subprocess.run(command, shell=True, cwd=WORK_DIR, text=True, capture_output=True, timeout=10)
        result = f"exit={completed.returncode}\nstdout={completed.stdout.strip()}\nstderr={completed.stderr.strip()}"
    else:
        result = f"unsupported tool: {name}"

    logging.info("TOOL result name=%s result=%s", name, result[:1000])
    return result


def tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "create_new_file",
                "description": "Create a small file in the test workspace.",
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
                "name": "read_file",
                "description": "Read a small file from the test workspace.",
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
                "name": "list_files",
                "description": "List files in the test workspace.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_existing_file",
                "description": "Replace a small file with edited contents in the test workspace.",
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
                "name": "run_terminal_command",
                "description": "Run a safe terminal command from the test workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--large-attempts", type=int, default=3)
    parser.add_argument("--large-min-chars", type=int, default=8000)
    parser.add_argument(
        "--require-continuation",
        action="store_true",
        help="Fail if DeepSeek declines every large answer before a Continue button is observed.",
    )
    args = parser.parse_args()

    setup_logging()
    logging.info("=== live Continue-like route test started ===")
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    health = get_health(args.base_url)
    require(bool(health.get("ready")), "running adapter is ready")
    require(isinstance(health.get("diagnostics"), dict), "adapter exposes Continue diagnostics")

    plain = post_json(
        args.base_url,
        "/chat/completions",
        {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Ответь только словом OK."}],
            "timeout": args.timeout,
        },
        args.timeout,
    )
    require(plain["choices"][0]["finish_reason"] == "stop", "plain request finishes normally")
    require_clean_message(plain["choices"][0]["message"], "plain response")

    events = post_sse(
        args.base_url,
        "/chat/completions",
        {
            "model": "deepseek-chat",
            "stream": True,
            "messages": [{"role": "user", "content": "Ответь коротко: stream-ok"}],
            "timeout": args.timeout,
        },
        args.timeout,
    )
    require(any("data: [DONE]" in event for event in events), "streaming route finishes with DONE")

    continuation_seen = False
    long_answer = ""
    for attempt in range(1, max(0, args.large_attempts) + 1):
        marker = f"DEEPBILL_LONG_DONE_{attempt}"
        before_continue = continue_total(get_health(args.base_url))
        rule_count = 120 + (attempt * 20)
        long_data = post_json(
            args.base_url,
            "/chat/completions",
            {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Continuation stress test. Write one complete monolithic Python validation engine "
                            "file. Output code only in one fenced python block. Define "
                            f"{rule_count} concrete rule_001 through rule_{rule_count:03d} functions with "
                            f"different real validation bodies and {rule_count} matching unittest methods, "
                            "plus registries, CLI, JSON loading, reports, and sample records. Do not use loops "
                            "to generate the rule functions, placeholders, ellipsis, omitted sections, "
                            "multiple files, or tools. After the code block write "
                            f"the exact final marker {marker}."
                        ),
                    }
                ],
                "timeout": args.timeout,
            },
            args.timeout,
        )
        long_choice = long_data["choices"][0]
        long_message = long_choice["message"]
        require(long_choice["finish_reason"] == "stop", f"large answer attempt {attempt} finishes normally")
        require_clean_message(long_message, f"large answer attempt {attempt}")
        long_answer = str(long_message.get("content") or "")
        after_continue = continue_total(get_health(args.base_url))
        logging.info(
            "LARGE attempt=%s rule_count=%s chars=%s continue_before=%s continue_after=%s",
            attempt,
            rule_count,
            len(long_answer),
            before_continue,
            after_continue,
        )
        if after_continue <= before_continue:
            continue
        continuation_seen = True
        require(len(long_answer) >= args.large_min_chars, "continued large answer keeps substantial code")
        logging.info("LARGE final_marker_seen=%s marker=%s", marker in long_answer, marker)
        break

    if args.require_continuation:
        require(continuation_seen, "large answer triggered and captured a DeepSeek Continue click")
    elif not continuation_seen:
        logging.warning("Large-answer attempts ended before DeepSeek exposed a Continue button.")

    followup = post_json(
        args.base_url,
        "/chat/completions",
        {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Answer with only LONG-FOLLOWUP-OK."}],
            "timeout": args.timeout,
        },
        args.timeout,
    )
    followup_message = followup["choices"][0]["message"]
    followup_text = str(followup_message.get("content") or "")
    require_clean_message(followup_message, "large answer follow-up")
    require("LONG-FOLLOWUP-OK" in followup_text, "short request after continuation returns its answer")
    require("DEEPBILL_LONG_DONE" not in followup_text, "short request after continuation has no prior answer tail")

    terminal_messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                'Терминальный агентный тест. Вызови run_terminal_command с командой '
                'python3 -c "print(123)", затем коротко подтверди результат.'
            ),
        }
    ]
    terminal_tool_names: list[str] = []
    terminal_outputs: list[str] = []
    terminal_final_seen = False
    for step in range(1, 4):
        data = post_json(
            args.base_url,
            "/chat/completions",
            {
                "model": "deepseek-chat",
                "messages": terminal_messages,
                "tools": tools(),
                "timeout": args.timeout,
            },
            args.timeout,
        )
        choice = data["choices"][0]
        message = choice["message"]
        logging.info("TERMINAL_AGENT step=%s finish_reason=%s message=%s", step, choice["finish_reason"], message)
        require_clean_message(message, f"terminal agent step {step}")
        if choice["finish_reason"] == "tool_calls":
            calls = message.get("tool_calls") or []
            require(bool(calls), f"terminal agent step {step} returned at least one tool call")
            terminal_messages.append(message)
            for call in calls:
                terminal_tool_names.append(call["function"]["name"])
                result = execute_tool(call)
                terminal_outputs.append(result)
                terminal_messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            continue
        require(choice["finish_reason"] == "stop", f"terminal agent step {step} finishes with stop or tool_calls")
        require(bool(str(message.get("content") or "").strip()), "terminal agent final answer is not empty")
        terminal_final_seen = True
        break

    require("run_terminal_command" in terminal_tool_names, "terminal agent used run_terminal_command")
    require(any("stdout=123" in output for output in terminal_outputs), "terminal command actually executed")
    require(terminal_final_seen, "terminal agent produced a final answer after command result")

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Агентный тест. Создай маленький файл agent_test.html с содержимым "
                "<h1>OK</h1>, затем прочитай его через инструмент и в финале коротко подтверди."
            ),
        }
    ]
    tool_names: list[str] = []
    final_seen = False
    for step in range(1, 5):
        data = post_json(
            args.base_url,
            "/chat/completions",
            {
                "model": "deepseek-chat",
                "messages": messages,
                "tools": tools(),
                "timeout": args.timeout,
            },
            args.timeout,
        )
        choice = data["choices"][0]
        message = choice["message"]
        logging.info("AGENT step=%s finish_reason=%s message=%s", step, choice["finish_reason"], message)
        require_clean_message(message, f"agent step {step}")
        if choice["finish_reason"] == "tool_calls":
            calls = message.get("tool_calls") or []
            require(bool(calls), f"agent step {step} returned at least one tool call")
            messages.append(message)
            for call in calls:
                tool_names.append(call["function"]["name"])
                result = execute_tool(call)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            continue
        require(choice["finish_reason"] == "stop", f"agent step {step} finishes with stop or tool_calls")
        require(bool(str(message.get("content") or "").strip()), "agent final answer is not empty")
        final_seen = True
        break

    require("create_new_file" in tool_names, "agent used create_new_file")
    require(WORK_DIR.joinpath("agent_test.html").exists(), "test file was actually created by tool executor")
    require(final_seen, "agent produced a final answer after tool results")
    logging.info("tool_names=%s", tool_names)

    edit_messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Агентный тест редактирования. Сначала создай tiny_note.txt с текстом blue. "
                "Затем обязательно прочитай файл через read_file. Только после чтения вызови "
                "edit_existing_file и замени содержимое на orange. После этого ещё раз прочитай "
                "файл через read_file и коротко подтверди."
            ),
        }
    ]
    edit_tool_names: list[str] = []
    edit_final_seen = False
    for step in range(1, 8):
        data = post_json(
            args.base_url,
            "/chat/completions",
            {
                "model": "deepseek-chat",
                "messages": edit_messages,
                "tools": tools(),
                "timeout": args.timeout,
            },
            args.timeout,
        )
        choice = data["choices"][0]
        message = choice["message"]
        logging.info("EDIT_AGENT step=%s finish_reason=%s message=%s", step, choice["finish_reason"], message)
        require_clean_message(message, f"edit agent step {step}")
        if choice["finish_reason"] == "tool_calls":
            calls = message.get("tool_calls") or []
            require(bool(calls), f"edit agent step {step} returned at least one tool call")
            edit_messages.append(message)
            for call in calls:
                edit_tool_names.append(call["function"]["name"])
                result = execute_tool(call)
                edit_messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            continue
        require(choice["finish_reason"] == "stop", f"edit agent step {step} finishes with stop or tool_calls")
        require(bool(str(message.get("content") or "").strip()), "edit agent final answer is not empty")
        edit_final_seen = True
        break

    require("create_new_file" in edit_tool_names, "edit agent used create_new_file")
    require("edit_existing_file" in edit_tool_names, "edit agent used edit_existing_file")
    require("read_file" in edit_tool_names, "edit agent used read_file")
    require(edit_tool_names.count("read_file") >= 2, "edit agent read before and after editing")
    require(
        edit_tool_names.index("read_file") < edit_tool_names.index("edit_existing_file"),
        "edit agent received read_file result before editing",
    )
    require(WORK_DIR.joinpath("tiny_note.txt").read_text(encoding="utf-8").strip() == "orange", "edit agent actually changed file contents")
    require(edit_final_seen, "edit agent produced a final answer after tool results")
    logging.info("edit_tool_names=%s", edit_tool_names)
    logging.info("=== live Continue-like route test passed ===")


if __name__ == "__main__":
    main()

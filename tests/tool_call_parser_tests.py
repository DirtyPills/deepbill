#!/usr/bin/env python3
"""Focused regression tests for tool_call_parser.py."""

from __future__ import annotations

import json

from _path import PROJECT_ROOT  # noqa: F401
from tool_call_parser import ToolCallParser


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"ok - {message}")


def names(result):
    return [call.name for call in result.calls]


def main() -> None:
    parser = ToolCallParser()

    result = parser.parse(
        'text Copy Downloadtool_call Copy Download {"name":"create_new_file","arguments":{"filepath":"calculator.html","contents":"<p>Copy Download stays</p>"}}'
    )
    assert_true(names(result) == ["create_new_file"], "glued UI words before tool_call are tolerated")
    assert_true(result.calls[0].arguments["filepath"] == "calculator.html", "glued marker arguments parse")
    assert_true("Copy Download stays" in result.calls[0].arguments["contents"], "argument content is not scrubbed")

    result = parser.parse(
        "Сделаю правку.\n\n"
        'tool_call Copy Download {"name":"edit_existing_file","arguments":{"filepath":"style.css","changes":"body { color: orange; }"}}'
    )
    assert_true(result.cleaned_text == "Сделаю правку.", "prose before a tool call is preserved as cleaned text")
    assert_true(names(result) == ["edit_existing_file"], "prose plus inline tool call parses")

    result = parser.parse(
        'tool_call Copy Download [{"name":"create_new_file","arguments":{"filepath":"a.txt","contents":"A"}},'
        '{"name":"read_file","arguments":{"filepath":"a.txt"}}]'
    )
    assert_true(names(result) == ["create_new_file", "read_file"], "array of tool calls parses")

    result = parser.parse(
        "```tool_call\n{'name': 'read_file', 'arguments': {'filepath': 'a.txt'}}\n```"
    )
    assert_true(names(result) == ["read_file"], "fenced Python-literal tool call parses")

    result = parser.parse(
        'tool_call\n{"name":"read_file","path":"TOGOSHOL/package.json","mode":"slice","offset":1,"limit":200}'
    )
    assert_true(names(result) == ["read_file"], "direct-argument DeepSeek tool call parses")
    assert_true(result.calls[0].arguments["path"] == "TOGOSHOL/package.json", "direct path argument is preserved")
    assert_true(result.calls[0].arguments["mode"] == "slice", "direct mode argument is preserved")

    result = parser.parse(
        '<tool_call><name>list_files</name><arguments>{"path":"."}</arguments></tool_call>'
    )
    assert_true(names(result) == ["list_files"], "XML tool call parses")

    result = parser.parse(
        '{"tool_calls":[{"type":"function","function":{"name":"read_file","arguments":"{\\"filepath\\":\\"a.txt\\"}"}}]}',
        allow_bare_json=True,
    )
    assert_true(names(result) == ["read_file"], "OpenAI-shaped bare JSON parses when allowed")

    result = parser.parse("Copy Download text only, no tool object")
    assert_true(not result.calls, "plain UI-looking text does not become a tool call")

    result = parser.parse('const s = "tool_call"; const x = { not: "a tool" };')
    assert_true(not result.calls, "tool_call word in ordinary code is ignored without valid schema")

    result = parser.parse(
        'tool_call Copy Download {"name":"create_new_file","arguments":{"filepath":"voice_assistant.py",'
        '"contents":"#!/usr/bin/env python3\\n"""\\nVoice Assistant\\n"""\\nprint("hi")\\n"}}'
    )
    assert_true(names(result) == ["create_new_file"], "malformed JSON code contents still parses as tool call")
    assert_true(result.cleaned_text == "", "malformed JSON tool call is fully removed from assistant text")
    assert_true('print("hi")' in result.calls[0].arguments["contents"], "malformed JSON code contents are preserved")

    result = parser.parse(
        'tool_call Copy Download {"name":"run_terminal_command","arguments":{"command":"python -c "print(123)""}}'
    )
    assert_true(names(result) == ["run_terminal_command"], "malformed terminal command parses as tool call")
    assert_true('print(123)' in result.calls[0].arguments["command"], "terminal command argument is preserved")

    print(json.dumps({"status": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

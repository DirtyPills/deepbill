#!/usr/bin/env python3
"""Robust tool-call extraction for web-model adapter output.

The DeepSeek web UI can leak UI labels such as "Copy", "Download", or "text"
around copied code blocks. This parser treats those labels as noise only when
they are adjacent to an explicit tool-call marker, and then extracts structured
tool calls without touching ordinary assistant text or file contents.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


UI_NOISE_WORDS = {
    "copy",
    "download",
    "text",
    "json",
    "python",
    "yaml",
    "yml",
    "plaintext",
    "plain",
}

TOOL_MARKER_RE = re.compile(
    r"(?is)(tool\s*[_\-. ]?\s*call|function\s*[_\-. ]?\s*call|tool\s*use|use\s*tool)"
)

FENCE_RE = re.compile(r"(?is)```(?P<label>[^\n`]*)\n(?P<body>.*?)\n?```")
XML_TOOL_RE = re.compile(r"(?is)<tool_call\b[^>]*>(.*?)</tool_call>")


@dataclass
class RawToolCall:
    name: str
    arguments: Any = field(default_factory=dict)


@dataclass
class ToolParseResult:
    cleaned_text: str
    calls: List[RawToolCall] = field(default_factory=list)
    ranges: List[Tuple[int, int]] = field(default_factory=list)
    explicit_marker: bool = False


def parse_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    raw = strip_leading_ui_noise(value.strip())
    if not raw:
        return ""
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def strip_leading_ui_noise(text: str) -> str:
    raw = text or ""
    cursor = 0
    while True:
        match = re.match(r"(?is)\s*([a-z_ -]+)\s*", raw[cursor:])
        if not match:
            break
        word = re.sub(r"[\s_-]+", "", match.group(1).strip().lower())
        if word not in UI_NOISE_WORDS:
            break
        cursor += match.end()
    return raw[cursor:].strip()


def _skip_noise_after_marker(text: str, pos: int) -> int:
    cursor = pos
    while cursor < len(text):
        ws = re.match(r"(?is)[\s:=>\-.,;|`]*", text[cursor:])
        if ws:
            cursor += ws.end()
        word = re.match(r"(?is)(copy|download|text|json|python|yaml|yml|plaintext|plain)\b", text[cursor:])
        if not word:
            return cursor
        cursor += word.end()


def _extend_start_over_ui_noise(text: str, start: int) -> int:
    line_start = max(text.rfind("\n", 0, start), text.rfind("\r", 0, start)) + 1
    prefix = text[line_start:start]
    if not prefix:
        return start
    tokens = re.findall(r"(?is)[a-z]+", prefix)
    if tokens and all(token.lower() in UI_NOISE_WORDS for token in tokens):
        return line_start
    return start


def find_balanced_value(text: str, start: int) -> Optional[Tuple[int, int]]:
    opener = -1
    for idx in range(start, len(text)):
        if text[idx] in "{[":
            opener = idx
            break
    if opener < 0:
        return None

    closer_for = {"{": "}", "[": "]"}
    stack: List[str] = []
    quote: Optional[str] = None
    escape = False
    for idx in range(opener, len(text)):
        char = text[idx]
        if quote is not None:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in closer_for:
            stack.append(closer_for[char])
            continue
        if stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return opener, idx + 1
    return None


def _decode_jsonish_string(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    try:
        return json.loads(f'"{raw}"')
    except Exception:
        return (
            raw.replace("\\r\\n", "\n")
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace('\\"', '"')
            .replace("\\'", "'")
        )


def _strip_outer_string(value: str) -> str:
    raw = value.strip()
    if len(raw) >= 2 and raw[0] in {'"', "'"}:
        quote = raw[0]
        inner = raw[1:].strip()
        while inner.endswith("}") or inner.endswith("]"):
            inner = inner[:-1].rstrip()
        if inner.endswith(quote):
            inner = inner[:-1]
        return _decode_jsonish_string(inner)
    return _decode_jsonish_string(raw)


def _find_next_argument_key(text: str, start: int) -> Optional[re.Match[str]]:
    pattern = re.compile(r"""(?is)(?:^|,)\s*(['"])([A-Za-z0-9_.:-]{1,128})\1\s*:""")
    return pattern.search(text, start)


def _trim_jsonish_tail(value: str) -> str:
    raw = value.strip()
    while raw.endswith(","):
        raw = raw[:-1].rstrip()
    while raw.endswith("}") or raw.endswith("]"):
        quote_count = raw.count('"') + raw.count("'")
        if quote_count >= 2 and (raw[-2:-1] in {'"', "'"} or re.search(r"""['"]\s*$""", raw[:-1])):
            raw = raw[:-1].rstrip()
            continue
        break
    return raw.strip()


def _parse_malformed_arguments(text: str) -> dict[str, Any]:
    parsed = parse_json_maybe(text)
    if isinstance(parsed, dict):
        return parsed

    args: dict[str, Any] = {}
    key_re = re.compile(r"""(?is)(['"])([A-Za-z0-9_.:-]{1,128})\1\s*:""")
    cursor = 0
    while True:
        match = key_re.search(text, cursor)
        if not match:
            break
        key = match.group(2)
        value_start = match.end()
        next_match = _find_next_argument_key(text, value_start)
        value_end = next_match.start() if next_match else len(text)
        raw_value = _trim_jsonish_tail(text[value_start:value_end])
        if raw_value:
            args[key] = _strip_outer_string(raw_value)
        cursor = value_end
    return args


def _parse_malformed_tool_object(text: str, start: int) -> Optional[Tuple[List[RawToolCall], Tuple[int, int]]]:
    opener = text.find("{", start)
    if opener < 0:
        return None

    balanced = find_balanced_value(text, start)
    if balanced:
        candidate_end = balanced[1]
    else:
        next_marker = TOOL_MARKER_RE.search(text, opener + 1)
        candidate_end = next_marker.start() if next_marker else len(text)

    candidate = text[opener:candidate_end].strip()
    name_match = re.search(r"""(?is)['"]name['"]\s*:\s*['"]([A-Za-z0-9_.:-]{1,128})['"]""", candidate)
    if not name_match:
        return None

    args_match = re.search(r"""(?is)['"](?:arguments|args|parameters|input)['"]\s*:\s*[{]""", candidate)
    if args_match:
        args_start = args_match.end() - 1
        args_text = candidate[args_start:]
        if args_text.startswith("{"):
            args_text = args_text[1:]
        arguments = _parse_malformed_arguments(args_text)
    else:
        arguments = {}

    call = RawToolCall(name_match.group(1).strip(), arguments)
    return [call], (opener, candidate_end)


class ToolCallParser:
    @staticmethod
    def has_explicit_marker(text: str) -> bool:
        raw = text or ""
        return bool(TOOL_MARKER_RE.search(raw) or XML_TOOL_RE.search(raw) or re.search(r"(?is)```(?:tool_call|tool|function)", raw))

    def parse(self, text: str, *, allow_bare_json: bool = False) -> ToolParseResult:
        raw = text or ""
        calls: List[RawToolCall] = []
        ranges: List[Tuple[int, int]] = []
        explicit_marker = self.has_explicit_marker(raw)

        occupied_ranges: List[Tuple[int, int]] = []

        for block_calls, block_range in self._parse_fences(raw):
            calls.extend(block_calls)
            ranges.append(block_range)
            occupied_ranges.append(block_range)

        for block_calls, block_range in self._parse_xml(raw):
            calls.extend(block_calls)
            ranges.append(block_range)
            occupied_ranges.append(block_range)

        for block_calls, block_range in self._parse_inline_markers(raw, occupied_ranges):
            calls.extend(block_calls)
            ranges.append(block_range)

        if allow_bare_json and not calls:
            parsed = parse_json_maybe(raw)
            calls.extend(self._calls_from_obj(parsed))
            if calls:
                ranges.append((0, len(raw)))

        if not calls:
            return ToolParseResult(raw.strip(), [], [], explicit_marker)

        merged_ranges = self._merge_ranges(ranges)
        cleaned = self._remove_ranges(raw, merged_ranges).strip()
        return ToolParseResult(cleaned, calls, merged_ranges, explicit_marker)

    def _parse_fences(self, raw: str) -> List[Tuple[List[RawToolCall], Tuple[int, int]]]:
        found: List[Tuple[List[RawToolCall], Tuple[int, int]]] = []
        for match in FENCE_RE.finditer(raw):
            label = (match.group("label") or "").strip().lower()
            body = (match.group("body") or "").strip()
            label_is_toolish = any(word in label for word in ("tool", "function", "json"))
            has_marker = self.has_explicit_marker(body)
            candidates = [body]
            if has_marker:
                for marker in TOOL_MARKER_RE.finditer(body):
                    value_range = find_balanced_value(body, _skip_noise_after_marker(body, marker.end()))
                    if value_range:
                        candidates.append(body[value_range[0] : value_range[1]])
            if not label_is_toolish and not has_marker:
                continue
            calls: List[RawToolCall] = []
            for candidate in candidates:
                calls.extend(self._calls_from_obj(parse_json_maybe(candidate)))
                if calls:
                    break
            if not calls:
                calls.extend(self._parse_legacy_block(body))
            if calls:
                found.append((calls, (match.start(), match.end())))
        return found

    def _parse_xml(self, raw: str) -> List[Tuple[List[RawToolCall], Tuple[int, int]]]:
        found: List[Tuple[List[RawToolCall], Tuple[int, int]]] = []
        for match in XML_TOOL_RE.finditer(raw):
            block = match.group(1)
            name_match = re.search(r"(?is)<(?:name|tool_name|tool)>(.*?)</(?:name|tool_name|tool)>", block)
            args_match = re.search(r"(?is)<(?:arguments|args|parameters|input)>(.*?)</(?:arguments|args|parameters|input)>", block)
            if not name_match:
                continue
            args = parse_json_maybe(args_match.group(1).strip()) if args_match else {}
            found.append(([RawToolCall(name_match.group(1).strip(), args)], (match.start(), match.end())))
        return found

    def _parse_inline_markers(
        self,
        raw: str,
        occupied_ranges: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Tuple[List[RawToolCall], Tuple[int, int]]]:
        found: List[Tuple[List[RawToolCall], Tuple[int, int]]] = []
        for marker in TOOL_MARKER_RE.finditer(raw):
            if self._position_inside_ranges(marker.start(), occupied_ranges or []):
                continue
            start_after_noise = _skip_noise_after_marker(raw, marker.end())
            value_range = find_balanced_value(raw, start_after_noise)
            if value_range is None:
                malformed = _parse_malformed_tool_object(raw, start_after_noise)
                if malformed:
                    calls, value_range = malformed
                else:
                    continue
            else:
                parsed = parse_json_maybe(raw[value_range[0] : value_range[1]])
                calls = self._calls_from_obj(parsed)
                if not calls:
                    malformed = _parse_malformed_tool_object(raw, start_after_noise)
                    if malformed:
                        calls, value_range = malformed
            if not calls:
                continue
            start = _extend_start_over_ui_noise(raw, marker.start())
            found.append((calls, (start, value_range[1])))
        return found

    @staticmethod
    def _position_inside_ranges(pos: int, ranges: List[Tuple[int, int]]) -> bool:
        return any(start <= pos < end for start, end in ranges)

    def _parse_legacy_block(self, block: str) -> List[RawToolCall]:
        name_match = re.search(r"(?im)^\s*(?:TOOL_NAME|tool_name|name)\s*:\s*(.+?)\s*$", block)
        if not name_match:
            return []
        args: dict[str, Any] = {}
        for arg_match in re.finditer(r"(?is)BEGIN_ARG\s*:\s*([^\n]+?)\s*\n(.*?)\n\s*END_ARG", block):
            args[arg_match.group(1).strip()] = parse_json_maybe(arg_match.group(2).strip())
        return [RawToolCall(name_match.group(1).strip(), args)]

    def _calls_from_obj(self, obj: Any) -> List[RawToolCall]:
        calls: List[RawToolCall] = []
        if isinstance(obj, str):
            parsed = parse_json_maybe(obj)
            if parsed is obj:
                return []
            return self._calls_from_obj(parsed)
        if isinstance(obj, list):
            for item in obj:
                calls.extend(self._calls_from_obj(item))
            return calls
        if not isinstance(obj, dict):
            return calls

        if isinstance(obj.get("tool_calls"), list):
            for item in obj["tool_calls"]:
                calls.extend(self._calls_from_obj(item))
            return calls
        if isinstance(obj.get("tools"), list):
            for item in obj["tools"]:
                calls.extend(self._calls_from_obj(item))
            return calls

        function_call = obj.get("function_call")
        if isinstance(function_call, dict):
            calls.extend(self._calls_from_obj({"function": function_call}))
            return calls

        argument_keys = ("arguments", "args", "parameters", "tool_input", "action_input", "input")

        func = obj.get("function")
        if isinstance(func, dict):
            name = func.get("name") or obj.get("name")
            arguments = next((func[key] for key in argument_keys if key in func), None)
            if arguments is None:
                arguments = {
                    key: value
                    for key, value in func.items()
                    if key not in {"name", "description", "type"} and not key.startswith("_")
                }
            call = self._make_raw_call(name, arguments)
            return [call] if call else []

        name = (
            obj.get("name")
            or obj.get("tool_name")
            or obj.get("tool")
            or obj.get("action")
            or obj.get("function")
        )
        arguments = next((obj[key] for key in argument_keys if key in obj), None)
        if arguments is None:
            arguments = {
                key: value
                for key, value in obj.items()
                if key
                not in {
                    "name",
                    "tool_name",
                    "tool",
                    "action",
                    "function",
                    "function_call",
                    "tool_calls",
                    "tools",
                }
                and not key.startswith("_")
            }
        call = self._make_raw_call(name, arguments)
        if call:
            calls.append(call)
        return calls

    def _make_raw_call(self, name: Any, arguments: Any) -> Optional[RawToolCall]:
        if not isinstance(name, str):
            return None
        clean_name = name.strip()
        if not clean_name or not re.match(r"^[A-Za-z0-9_.:-]{1,128}$", clean_name):
            return None
        parsed_args = parse_json_maybe(arguments)
        if parsed_args is None or parsed_args == "":
            parsed_args = {}
        return RawToolCall(clean_name, parsed_args)

    @staticmethod
    def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not ranges:
            return []
        merged: List[Tuple[int, int]] = []
        for start, end in sorted(ranges):
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    @staticmethod
    def _remove_ranges(raw: str, ranges: List[Tuple[int, int]]) -> str:
        parts: List[str] = []
        cursor = 0
        for start, end in ranges:
            if start > cursor:
                parts.append(raw[cursor:start])
            cursor = max(cursor, end)
        parts.append(raw[cursor:])
        return "".join(parts)

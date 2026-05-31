#!/usr/bin/env python3
"""
OpenAI-compatible adapter for the DeepSeek web client used by ScreenHelper.

The adapter intentionally exposes the old and widely supported Chat
Completions shape because Continue, Roo Code, and most OpenAI-compatible
clients still use it for agent/tool workflows.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from collections import deque
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Tuple

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from tool_call_parser import ToolCallParser, parse_json_maybe
from werkzeug.serving import make_server


logger = logging.getLogger("OpenAIAdapter")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)


DEFAULT_MODEL = os.environ.get("DEEPBILL_ADAPTER_MODEL", "deepseek-chat")
DEFAULT_TIMEOUT_SEC = int(os.environ.get("DEEPBILL_ADAPTER_TIMEOUT", "360"))
MAX_REQUEST_TIMEOUT_SEC = int(os.environ.get("DEEPBILL_ADAPTER_MAX_TIMEOUT", "1800"))
MAX_CONTEXT_BUFFER_CHARS = int(os.environ.get("DEEPBILL_ADAPTER_BUFFER_CHARS", "12000"))
STREAM_TEXT_CHARS = int(os.environ.get("DEEPBILL_ADAPTER_STREAM_CHARS", "140"))
DEEPSEEK_RETRY_ATTEMPTS = max(0, int(os.environ.get("DEEPBILL_ADAPTER_RETRIES", "1")))


class OpenAIAdapter:
    def __init__(self, browser_worker: Optional[Any] = None, port: int = 8080):
        self.browser_worker = browser_worker
        self.port = port
        self.app = Flask(__name__)
        CORS(
            self.app,
            origins="*",
            methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization", "X-Client-Request-Id"],
        )
        self._tool_call_map: Dict[str, str] = {}
        self._state_lock = threading.Lock()
        self._context_buffers: Dict[str, Deque[str]] = {}
        self._context_buffer_chars: Dict[str, int] = {}
        self.new_chat_mode = os.environ.get("DEEPBILL_ADAPTER_NEW_CHAT_MODE", "auto").strip().lower()
        if self.new_chat_mode not in {"auto", "always", "never"}:
            self.new_chat_mode = "auto"
        self._active_conversation_key: Optional[str] = None
        self.context_buffer_enabled = os.environ.get("DEEPBILL_ADAPTER_CONTEXT_BUFFER", "1") != "0"
        self.single_message_new_chat = os.environ.get("DEEPBILL_ADAPTER_SINGLE_MESSAGE_NEW_CHAT", "1") != "0"
        self._tool_parser = ToolCallParser()
        self._register_routes()
        self._server_thread: Optional[threading.Thread] = None
        self._http_server: Optional[Any] = None
        self._running = False

    # ------------------------------------------------------------------ routes
    def _register_routes(self) -> None:
        @self.app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
        @self.app.route("/chat/completions", methods=["POST", "OPTIONS"])
        def chat_completions():
            if request.method == "OPTIONS":
                return self._options_response()
            return self._handle_chat_completion()

        @self.app.route("/v1/completions", methods=["POST", "OPTIONS"])
        def completions():
            if request.method == "OPTIONS":
                return self._options_response()
            return self._handle_legacy_completion()

        @self.app.route("/v1/models", methods=["GET"])
        @self.app.route("/models", methods=["GET"])
        def list_models():
            now = int(time.time())
            return jsonify(
                {
                    "object": "list",
                    "data": [
                        {"id": DEFAULT_MODEL, "object": "model", "created": now, "owned_by": "deepbill"},
                        {"id": "deepbill", "object": "model", "created": now, "owned_by": "deepbill"},
                    ],
                }
            )

        @self.app.route("/health", methods=["GET"])
        @self.app.route("/v1/health", methods=["GET"])
        def health():
            ready = bool(self.browser_worker)
            if self.browser_worker and hasattr(self.browser_worker, "status"):
                try:
                    _started, ready, error = self.browser_worker.status()
                    payload = {"status": "ok" if ready and not error else "starting", "ready": ready, "error": error}
                    if hasattr(self.browser_worker, "diagnostics"):
                        payload["diagnostics"] = self.browser_worker.diagnostics()
                    return payload
                except Exception as exc:
                    return {"status": "error", "ready": False, "error": str(exc)}, 503
            return {"status": "ok" if ready else "error", "ready": ready}

        @self.app.route("/test", methods=["GET"])
        def test():
            return {
                "status": "ok",
                "message": "OpenAI-compatible adapter with chat, tools, buffer, and SSE streaming",
            }

    @staticmethod
    def _options_response():
        resp = jsonify({})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Client-Request-Id"
        return resp

    # --------------------------------------------------------------- formatting
    @staticmethod
    def _json(data: Dict[str, Any]) -> str:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _sse(data: Any) -> str:
        if data == "[DONE]":
            return "data: [DONE]\n\n"
        return f"data: {OpenAIAdapter._json(data)}\n\n"

    @staticmethod
    def _new_id(prefix: str = "chatcmpl") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:24]}"

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type", ""))
                if item_type in {"text", "input_text"}:
                    parts.append(str(item.get("text", "")))
                elif item_type == "image_url":
                    image = item.get("image_url", {})
                    url = image.get("url") if isinstance(image, dict) else image
                    parts.append(f"[image_url: {url}]")
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    nested = OpenAIAdapter._content_to_text(item.get("content"))
                    if nested:
                        parts.append(nested)
            return "\n".join(part for part in parts if part)
        if isinstance(content, dict):
            for key in ("text", "content", "result", "output", "stdout", "stderr", "value"):
                if key in content:
                    return OpenAIAdapter._content_to_text(content.get(key))
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    def _content_chars_by_role(self, messages: List[Dict[str, Any]], role: str) -> Tuple[int, int]:
        count = 0
        chars = 0
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != role:
                continue
            count += 1
            chars += len(self._content_to_text(msg.get("content", "")))
        return count, chars

    @staticmethod
    def _parse_json_maybe(value: str) -> Any:
        return parse_json_maybe(value)

    def _remember_tool_call(self, call_id: str, name: str) -> None:
        with self._state_lock:
            self._tool_call_map[call_id] = name

    def _tool_name_from_messages(self, messages: List[Dict[str, Any]], call_id: str) -> str:
        with self._state_lock:
            if call_id in self._tool_call_map:
                return self._tool_call_map[call_id]
        for msg in reversed(messages):
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                if str(tc.get("id", "")) != str(call_id):
                    continue
                func = tc.get("function", {})
                if isinstance(func, dict) and func.get("name"):
                    return str(func["name"])
        return "tool"

    def _append_context_buffer(self, conversation_key: Optional[str], user_text: str, assistant_text: str) -> None:
        if not self.context_buffer_enabled:
            return
        if not conversation_key:
            return
        item = f"User:\n{user_text.strip()}\n\nAssistant:\n{assistant_text.strip()}"
        if not item.strip():
            return
        with self._state_lock:
            turns = self._context_buffers.setdefault(conversation_key, deque())
            turns.append(item)
            self._context_buffer_chars[conversation_key] = self._context_buffer_chars.get(conversation_key, 0) + len(item)
            while turns and self._context_buffer_chars.get(conversation_key, 0) > MAX_CONTEXT_BUFFER_CHARS:
                removed = turns.popleft()
                self._context_buffer_chars[conversation_key] = self._context_buffer_chars.get(conversation_key, 0) - len(removed)

    def _context_buffer_text(self, conversation_key: Optional[str]) -> str:
        if not self.context_buffer_enabled:
            return ""
        if not conversation_key:
            return ""
        with self._state_lock:
            return "\n\n".join(self._context_buffers.get(conversation_key, deque()))

    # --------------------------------------------------------------- tool prompt
    def _tools_to_prompt(self, tools: List[Dict[str, Any]], tool_choice: Any = None) -> str:
        function_tools = [tool for tool in tools if isinstance(tool, dict) and tool.get("type") == "function"]
        if not function_tools:
            return ""

        specs: List[Dict[str, Any]] = []
        for tool in function_tools:
            func = tool.get("function", {})
            if not isinstance(func, dict) or not func.get("name"):
                continue
            specs.append(
                {
                    "name": func.get("name"),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {"type": "object", "properties": {}}),
                    "strict": func.get("strict", False),
                }
            )
        if not specs:
            return ""

        choice_note = ""
        if tool_choice == "none":
            choice_note = "Tool choice is none: answer directly and do not call tools."
        elif tool_choice == "required":
            choice_note = "Tool choice is required: call at least one tool before giving a final answer."
        elif isinstance(tool_choice, dict):
            name = ((tool_choice.get("function") or {}).get("name") if isinstance(tool_choice.get("function"), dict) else "")
            if name:
                choice_note = f"Tool choice forces the tool named {name!r}."

        return (
            "You have access to application-side tools. Use them when needed for file, terminal, "
            "workspace, browser, or external actions.\n"
            "If a tool is needed, do not answer normally. Return only one or more fenced tool_call "
            "blocks, with valid JSON in each block:\n"
            "```tool_call\n"
            '{"name":"tool_name","arguments":{"arg":"value"}}\n'
            "```\n"
            "When an argument contains code, escape all JSON quotes and newlines; never paste the "
            "code outside the JSON object. Do not include UI words such as Copy or Download.\n"
            "The adapter will execute the tool and send the result back to you. After receiving tool "
            "results, continue the task or call another tool if needed.\n"
            f"{choice_note}\n"
            "Available tools JSON:\n"
            f"{json.dumps(specs, ensure_ascii=False, indent=2)}"
        ).strip()

    # -------------------------------------------------------------- tool parser
    def _make_tool_call(self, name: str, arguments: Any) -> Optional[Dict[str, Any]]:
        name = (name or "").strip()
        if not name:
            return None
        if arguments is None or arguments == "":
            arguments = {}
        if isinstance(arguments, str):
            parsed = self._parse_json_maybe(arguments)
            arguments = parsed if isinstance(parsed, (dict, list)) else {"input": str(parsed)}
        if not isinstance(arguments, (dict, list)):
            arguments = {"input": str(arguments)}
        call_id = f"call_{uuid.uuid4().hex[:24]}"
        arguments_str = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        self._remember_tool_call(call_id, name)
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments_str},
        }

    def _extract_tool_calls(self, text: str) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        parsed = self._tool_parser.parse(text or "", allow_bare_json=True)
        tool_calls = [call for raw_call in parsed.calls if (call := self._make_tool_call(raw_call.name, raw_call.arguments))]
        return parsed.cleaned_text, tool_calls or None

    @staticmethod
    def _has_explicit_tool_marker(text: str) -> bool:
        return ToolCallParser.has_explicit_marker(text or "")

    def _parse_assistant_answer(self, answer: str, allow_tool_calls: bool) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        if not allow_tool_calls and not self._tool_parser.has_explicit_marker(answer or ""):
            return (answer or "").strip(), None
        parsed = self._tool_parser.parse(answer or "", allow_bare_json=allow_tool_calls)
        tool_calls = [call for raw_call in parsed.calls if (call := self._make_tool_call(raw_call.name, raw_call.arguments))]
        if tool_calls:
            return parsed.cleaned_text, tool_calls
        if allow_tool_calls:
            return parsed.cleaned_text, None
        return (answer or "").strip(), None

    # -------------------------------------------------------------- prompt build
    def _has_history(self, messages: List[Dict[str, Any]]) -> bool:
        return sum(1 for msg in messages if msg.get("role") in {"user", "assistant", "tool"}) > 1

    def _is_single_user_request(self, messages: List[Dict[str, Any]]) -> bool:
        conversation_roles = [
            str(msg.get("role", ""))
            for msg in messages
            if isinstance(msg, dict) and str(msg.get("role", "")) in {"user", "assistant", "tool"}
        ]
        return conversation_roles == ["user"]

    def _latest_user_text(self, messages: List[Dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return self._content_to_text(msg.get("content", ""))
        return ""

    def _conversation_key(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        system_parts: List[str] = []
        first_user = ""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", ""))
            if role in {"system", "developer"}:
                text = self._content_to_text(msg.get("content", "")).strip()
                if text:
                    system_parts.append(text[:500])
                continue
            if role == "user":
                first_user = self._content_to_text(msg.get("content", "")).strip()
                break
        if not first_user:
            return None
        base = "\n".join(system_parts[-2:] + [first_user[:1000]])
        return str(uuid.uuid5(uuid.NAMESPACE_URL, base))

    def _should_start_new_chat(self, messages: List[Dict[str, Any]]) -> bool:
        if self.new_chat_mode == "never":
            return False
        if self.new_chat_mode == "always":
            return True

        if self.single_message_new_chat and self._is_single_user_request(messages):
            key = self._conversation_key(messages)
            with self._state_lock:
                self._active_conversation_key = key
            return True

        key = self._conversation_key(messages)
        if key is None:
            return False
        with self._state_lock:
            if self._active_conversation_key != key:
                self._active_conversation_key = key
                return True
        return False

    def _build_prompt(self, data: Dict[str, Any]) -> Tuple[str, str]:
        messages = data.get("messages") or []
        if not isinstance(messages, list):
            messages = []

        tools = data.get("tools") or []
        tool_choice = data.get("tool_choice")
        system_parts: List[str] = []
        transcript_parts: List[str] = []
        latest_user = self._latest_user_text(messages)
        conversation_key = self._conversation_key(messages)

        if self.context_buffer_enabled and not self._has_history(messages):
            with self._state_lock:
                same_active_conversation = bool(conversation_key and conversation_key == self._active_conversation_key)
            buffered = self._context_buffer_text(conversation_key) if same_active_conversation else ""
            if buffered:
                transcript_parts.append("Recent conversation buffer:\n" + buffered)

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "user"))
            content = self._content_to_text(msg.get("content", ""))
            if role in {"system", "developer"}:
                if content:
                    system_parts.append(content)
                continue
            if role == "user":
                transcript_parts.append(f"User:\n{content}")
                continue
            if role == "assistant":
                if content:
                    transcript_parts.append(f"Assistant:\n{content}")
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    packed = []
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        func = tc.get("function", {})
                        if not isinstance(func, dict):
                            continue
                        packed.append(
                            {
                                "id": tc.get("id", ""),
                                "name": func.get("name", ""),
                                "arguments": func.get("arguments", ""),
                            }
                        )
                    if packed:
                        transcript_parts.append("Assistant tool calls:\n" + json.dumps(packed, ensure_ascii=False))
                continue
            if role == "tool":
                call_id = str(msg.get("tool_call_id", ""))
                tool_name = str(msg.get("name") or self._tool_name_from_messages(messages, call_id))
                transcript_parts.append(f"Tool result ({tool_name}, id={call_id}):\n{content}")
                continue

            if content:
                transcript_parts.append(f"{role.title()}:\n{content}")

        if not transcript_parts and latest_user:
            transcript_parts.append(f"User:\n{latest_user}")

        tool_prompt = ""
        if tool_choice != "none" and isinstance(tools, list):
            tool_prompt = self._tools_to_prompt(tools, tool_choice=tool_choice)
        if tool_prompt:
            system_parts.append(tool_prompt)

        system_text = "\n\n".join(part for part in system_parts if part).strip()
        transcript = "\n\n".join(part for part in transcript_parts if part).strip()
        prompt_parts = []
        if system_text:
            prompt_parts.append("System instructions:\n" + system_text)
        if transcript:
            prompt_parts.append("Conversation:\n" + transcript)
        prompt_parts.append(
            "Answer as the assistant. If tool use is required, output only the requested tool_call block(s)."
        )
        return "\n\n".join(prompt_parts), latest_user

    # -------------------------------------------------------------- deepseek I/O
    def _require_worker(self):
        if self.browser_worker is None:
            raise RuntimeError("Browser worker is not attached to the adapter")
        return self.browser_worker

    def _maybe_new_chat(self, messages: Optional[List[Dict[str, Any]]] = None) -> None:
        if not self._should_start_new_chat(messages or []):
            return
        self._force_new_chat("conversation_boundary")

    @staticmethod
    def _is_retryable_deepseek_error(exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        text = str(exc).lower()
        retryable_fragments = (
            "timeout",
            "таймаут",
            "target closed",
            "page closed",
            "browser has been closed",
            "execution context was destroyed",
            "not attached to the dom",
            "не принял сообщение",
            "поле ввода не очистилось",
        )
        return any(fragment in text for fragment in retryable_fragments)

    def _force_new_chat(self, reason: str) -> bool:
        worker = self._require_worker()
        if not hasattr(worker, "new_chat"):
            return False
        try:
            worker.new_chat(timeout=30)
            logger.info("stage=deepseek_new_chat reason=%s", reason)
            return True
        except Exception as exc:
            logger.warning("stage=deepseek_new_chat_failed reason=%s error=%s", reason, exc)
            return False

    def _ask_text(self, prompt: str, timeout: int, messages: Optional[List[Dict[str, Any]]] = None) -> str:
        worker = self._require_worker()
        self._maybe_new_chat(messages)
        attempts = DEEPSEEK_RETRY_ATTEMPTS + 1
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                return str(worker.ask_text(prompt=prompt, timeout=timeout) or "").strip()
            except Exception as exc:
                last_error = exc
                if attempt >= attempts or not self._is_retryable_deepseek_error(exc):
                    raise
                logger.warning(
                    "stage=deepseek_retry mode=sync attempt=%s/%s error=%s",
                    attempt + 1,
                    attempts,
                    exc,
                )
                self._force_new_chat("retry_after_error")
        raise last_error or RuntimeError("DeepSeek request failed")

    def _ask_text_stream(
        self,
        prompt: str,
        timeout: int,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Iterable[str]:
        worker = self._require_worker()
        self._maybe_new_chat(messages)
        attempts = DEEPSEEK_RETRY_ATTEMPTS + 1
        for attempt in range(1, attempts + 1):
            try:
                if hasattr(worker, "ask_text_stream"):
                    yield from worker.ask_text_stream(prompt=prompt, timeout=timeout)
                    return
                answer = str(worker.ask_text(prompt=prompt, timeout=timeout) or "")
                for chunk in self._split_text(answer):
                    yield chunk
                return
            except Exception as exc:
                if attempt >= attempts or not self._is_retryable_deepseek_error(exc):
                    raise
                logger.warning(
                    "stage=deepseek_retry mode=stream attempt=%s/%s error=%s",
                    attempt + 1,
                    attempts,
                    exc,
                )
                self._force_new_chat("retry_after_stream_error")

    # --------------------------------------------------------------- responses
    @staticmethod
    def _split_text(text: str, chunk_size: int = STREAM_TEXT_CHARS) -> Iterator[str]:
        raw = text or ""
        if not raw:
            return
        start = 0
        while start < len(raw):
            end = min(len(raw), start + chunk_size)
            if end < len(raw):
                space = raw.rfind(" ", start + max(20, chunk_size // 2), end)
                if space > start:
                    end = space + 1
            yield raw[start:end]
            start = end

    @staticmethod
    def _usage(prompt: str, answer: str) -> Dict[str, int]:
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(answer) // 4) if answer else 0
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _completion_response(
        self,
        request_id: str,
        created: int,
        model: str,
        prompt: str,
        content: Optional[str],
        tool_calls: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        message: Dict[str, Any] = {"role": "assistant", "content": content or ""}
        finish_reason = "stop"
        if tool_calls:
            message["content"] = None
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": self._usage(prompt, content or ""),
        }

    def _handle_chat_completion(self):
        data = request.get_json(silent=True) or {}
        messages = data.get("messages") or []
        model = str(data.get("model") or DEFAULT_MODEL)
        stream = bool(data.get("stream", False))
        timeout = int(data.get("timeout") or data.get("request_timeout") or DEFAULT_TIMEOUT_SEC)
        timeout = max(5, min(timeout, MAX_REQUEST_TIMEOUT_SEC))
        request_id = self._new_id()
        created = int(time.time())
        tool_message_count, tool_result_chars = self._content_chars_by_role(messages, "tool") if isinstance(messages, list) else (0, 0)

        logger.info(
            "chat.completions model=%s stream=%s messages=%s tools=%s tool_messages=%s tool_result_chars=%s",
            model,
            stream,
            len(messages),
            len(data.get("tools") or []),
            tool_message_count,
            tool_result_chars,
        )

        try:
            prompt, latest_user = self._build_prompt(data)
            conversation_key = self._conversation_key(messages) if isinstance(messages, list) else None
            logger.info("stage=prompt_built request_id=%s prompt_chars=%s latest_user_chars=%s", request_id, len(prompt), len(latest_user))
        except Exception as exc:
            logger.exception("Invalid chat completion request")
            return jsonify({"error": {"message": str(exc), "type": "invalid_request_error"}}), 400

        if not latest_user and not any(msg.get("role") == "tool" for msg in messages if isinstance(msg, dict)):
            return jsonify({"error": {"message": "No user or tool message", "type": "invalid_request_error"}}), 400

        if stream:
            return self._stream_chat_completion(data, prompt, latest_user, conversation_key, request_id, created, model, timeout)

        try:
            logger.info("stage=deepseek_request request_id=%s mode=sync timeout=%s", request_id, timeout)
            answer = self._ask_text(prompt, timeout=timeout, messages=messages if isinstance(messages, list) else [])
            logger.info("stage=deepseek_response request_id=%s answer_chars=%s preview=%r", request_id, len(answer), answer[:240])
            has_tools = bool(data.get("tools")) and data.get("tool_choice") != "none"
            cleaned, tool_calls = self._parse_assistant_answer(answer, allow_tool_calls=has_tools)
            logger.info(
                "stage=parse_response request_id=%s has_tools=%s parsed_tool_calls=%s cleaned_chars=%s",
                request_id,
                has_tools,
                len(tool_calls or []),
                len(cleaned or ""),
            )
            if not tool_calls:
                self._append_context_buffer(conversation_key, latest_user, cleaned)
            return jsonify(self._completion_response(request_id, created, model, prompt, cleaned, tool_calls))
        except Exception as exc:
            logger.exception("DeepSeek request failed")
            return jsonify({"error": {"message": str(exc), "type": "server_error"}}), 500

    def _stream_chat_completion(
        self,
        data: Dict[str, Any],
        prompt: str,
        latest_user: str,
        conversation_key: Optional[str],
        request_id: str,
        created: int,
        model: str,
        timeout: int,
    ):
        has_tools = bool(data.get("tools")) and data.get("tool_choice") != "none"

        def role_chunk() -> Dict[str, Any]:
            return {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }

        def content_chunk(content: str) -> Dict[str, Any]:
            return {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }

        def finish_chunk(reason: str) -> Dict[str, Any]:
            return {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
            }

        def tool_delta(index: int, delta: Dict[str, Any], finish: Optional[str] = None) -> Dict[str, Any]:
            return {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": index, **delta}]},
                        "finish_reason": finish,
                    }
                ],
            }

        def generate() -> Iterator[str]:
            yield self._sse(role_chunk())
            try:
                mode = "stream_tools_buffered" if has_tools else "stream_parse_buffered"
                logger.info("stage=deepseek_request request_id=%s mode=%s timeout=%s", request_id, mode, timeout)
                collected = self._ask_text(
                    prompt,
                    timeout=timeout,
                    messages=data.get("messages") if isinstance(data.get("messages"), list) else [],
                ).strip()
                logger.info("stage=deepseek_response request_id=%s answer_chars=%s preview=%r", request_id, len(collected), collected[:240])
                cleaned, tool_calls = self._parse_assistant_answer(collected, allow_tool_calls=has_tools)
                logger.info(
                    "stage=parse_response request_id=%s has_tools=%s parsed_tool_calls=%s cleaned_chars=%s",
                    request_id,
                    has_tools,
                    len(tool_calls or []),
                    len(cleaned or ""),
                )
                if tool_calls:
                    for index, call in enumerate(tool_calls):
                        func = call.get("function", {})
                        yield self._sse(
                            tool_delta(
                                index,
                                {
                                    "id": call.get("id"),
                                    "type": "function",
                                    "function": {"name": func.get("name", ""), "arguments": ""},
                                },
                            )
                        )
                        for arg_delta in self._split_text(str(func.get("arguments", "")), chunk_size=96):
                            yield self._sse(tool_delta(index, {"function": {"arguments": arg_delta}}))
                    yield self._sse(finish_chunk("tool_calls"))
                    yield self._sse("[DONE]")
                    return

                for text_delta in self._split_text(cleaned):
                    yield self._sse(content_chunk(text_delta))
                self._append_context_buffer(conversation_key, latest_user, cleaned)
                yield self._sse(finish_chunk("stop"))
                yield self._sse("[DONE]")
            except Exception as exc:
                logger.exception("Streaming DeepSeek request failed")
                error_payload = {"error": {"message": str(exc), "type": "server_error"}}
                yield self._sse(error_payload)
                yield self._sse("[DONE]")

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
            },
        )

    def _handle_legacy_completion(self):
        data = request.get_json(silent=True) or {}
        model = str(data.get("model") or DEFAULT_MODEL)
        stream = bool(data.get("stream", False))
        timeout = int(data.get("timeout") or DEFAULT_TIMEOUT_SEC)
        timeout = max(5, min(timeout, MAX_REQUEST_TIMEOUT_SEC))
        prompt_value = data.get("prompt", "")
        if isinstance(prompt_value, list):
            prompt = "\n".join(str(item) for item in prompt_value)
        else:
            prompt = str(prompt_value or "")
        request_id = self._new_id("cmpl")
        created = int(time.time())

        if not prompt.strip():
            return jsonify({"error": {"message": "No prompt", "type": "invalid_request_error"}}), 400

        if stream:
            def generate() -> Iterator[str]:
                try:
                    for delta in self._ask_text_stream(prompt, timeout=timeout, messages=[]):
                        payload = {
                            "id": request_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "text": delta, "finish_reason": None}],
                        }
                        yield self._sse(payload)
                    yield self._sse(
                        {
                            "id": request_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                        }
                    )
                    yield self._sse("[DONE]")
                except Exception as exc:
                    yield self._sse({"error": {"message": str(exc), "type": "server_error"}})
                    yield self._sse("[DONE]")

            return Response(stream_with_context(generate()), mimetype="text/event-stream")

        try:
            answer = self._ask_text(prompt, timeout=timeout, messages=[])
        except Exception as exc:
            return jsonify({"error": {"message": str(exc), "type": "server_error"}}), 500
        return jsonify(
            {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "text": answer, "finish_reason": "stop"}],
                "usage": self._usage(prompt, answer),
            }
        )

    # --------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()
        logger.info("Adapter started on %s", self.get_url())

    def _run_server(self) -> None:
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.WARNING)
        server = make_server("0.0.0.0", self.port, self.app, threaded=True)
        self._http_server = server
        try:
            server.serve_forever()
        finally:
            server.server_close()
            self._http_server = None

    def stop(self) -> None:
        server = self._http_server
        self._running = False
        if server is not None:
            server.shutdown()
        thread = self._server_thread
        if thread is not None and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=5)
        logger.info("Adapter stopped")

    @property
    def is_running(self):
        return self._running and self._server_thread and self._server_thread.is_alive()

    def get_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

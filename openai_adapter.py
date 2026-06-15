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
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Tuple

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
REASONING_MODEL = os.environ.get("DEEPBILL_ADAPTER_REASONING_MODEL", "deepbill-deepthink")
AUTO_MODEL = os.environ.get("DEEPBILL_ADAPTER_AUTO_MODEL", "deepbill-auto")
DEFAULT_TIMEOUT_SEC = int(os.environ.get("DEEPBILL_ADAPTER_TIMEOUT", "360"))
MAX_REQUEST_TIMEOUT_SEC = int(os.environ.get("DEEPBILL_ADAPTER_MAX_TIMEOUT", "1800"))
REASONING_TIMEOUT_SEC = int(os.environ.get("DEEPBILL_REASONING_TIMEOUT", str(DEFAULT_TIMEOUT_SEC)))
REASONING_MAX_TIMEOUT_SEC = int(os.environ.get("DEEPBILL_REASONING_MAX_TIMEOUT", str(MAX_REQUEST_TIMEOUT_SEC)))
NORMAL_CONTEXT_WINDOW_TOKENS = int(os.environ.get("DEEPBILL_CONTEXT_WINDOW_TOKENS", "64000"))
REASONING_CONTEXT_WINDOW_TOKENS = int(os.environ.get("DEEPBILL_REASONING_CONTEXT_WINDOW_TOKENS", "128000"))
REASONING_CONTEXT_SOFT_LIMIT = int(os.environ.get("DEEPBILL_REASONING_CONTEXT_SOFT_LIMIT", "24000"))
MAX_CONTEXT_BUFFER_CHARS = int(os.environ.get("DEEPBILL_ADAPTER_BUFFER_CHARS", "12000"))
STREAM_TEXT_CHARS = int(os.environ.get("DEEPBILL_ADAPTER_STREAM_CHARS", "140"))
DEEPSEEK_RETRY_ATTEMPTS = max(0, int(os.environ.get("DEEPBILL_ADAPTER_RETRIES", "1")))
BROWSER_BUSY_TIMEOUT_SEC = float(os.environ.get("DEEPBILL_ADAPTER_BUSY_TIMEOUT", str(MAX_REQUEST_TIMEOUT_SEC)))
ADAPTER_QUEUE_LIMIT = max(0, int(os.environ.get("DEEPBILL_ADAPTER_QUEUE_LIMIT", "16")))
ADAPTER_CIRCUIT_FAILURE_THRESHOLD = max(0, int(os.environ.get("DEEPBILL_ADAPTER_CIRCUIT_FAILURE_THRESHOLD", "3")))
ADAPTER_CIRCUIT_COOLDOWN_SEC = max(1.0, float(os.environ.get("DEEPBILL_ADAPTER_CIRCUIT_COOLDOWN_SEC", "60")))
TRAFFIC_LOG_DIR = os.environ.get("DEEPBILL_TRAFFIC_LOG_DIR", "logs")
TRAFFIC_LOG_FILE = os.environ.get("DEEPBILL_TRAFFIC_LOG_FILE", "adapter_traffic.jsonl")
TRAFFIC_LOG_DETAILED = os.environ.get("DEEPBILL_TRAFFIC_LOG_DETAILED", "0") != "0"
TRAFFIC_HISTORY_LIMIT = max(10, int(os.environ.get("DEEPBILL_TRAFFIC_HISTORY_LIMIT", "300")))
REASONING_FORCE_MODEL_FRAGMENTS = (
    "deepthink",
    "reasoner",
    "reasoning",
    "r1",
)
REASONING_AUTO_MODEL_FRAGMENTS = (
    "auto",
)


def normalize_reasoning_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "off": "off",
        "disabled": "off",
        "нет": "off",
        "выкл": "off",
        "auto": "auto",
        "авто": "auto",
        "1": "on",
        "true": "on",
        "yes": "on",
        "on": "on",
        "enabled": "on",
        "да": "on",
        "вкл": "on",
    }
    return aliases.get(text, "off")


class BrowserBusyError(RuntimeError):
    """Raised when another request is already using the single DeepSeek browser."""


class AdapterCircuitOpenError(RuntimeError):
    """Raised when recent DeepSeek/browser failures temporarily block new work."""

    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = max(1.0, float(retry_after or 1.0))


class AdapterTrafficJournal:
    """Append-only adapter traffic journal with compact GUI events.

    Basic records are always written. Full request/response payloads are only
    written when detailed logging is enabled because they can be very large.
    """

    def __init__(
        self,
        log_dir: str = TRAFFIC_LOG_DIR,
        log_file: str = TRAFFIC_LOG_FILE,
        detailed: bool = TRAFFIC_LOG_DETAILED,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.path = Path(log_dir).expanduser().resolve() / log_file
        self.detailed = bool(detailed)
        self.event_sink = event_sink
        self._lock = threading.Lock()
        self._history: Deque[Dict[str, Any]] = deque(maxlen=TRAFFIC_HISTORY_LIMIT)

    def set_detailed(self, enabled: bool) -> None:
        self.detailed = bool(enabled)
        self.event(
            "traffic_log.mode",
            level="info",
            summary=f"detailed={'on' if self.detailed else 'off'}",
            data={"detailed": self.detailed, "path": str(self.path)},
        )

    def set_event_sink(self, event_sink: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        self.event_sink = event_sink

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "detailed": self.detailed,
            "tail": list(self._history)[-20:],
        }

    def event(
        self,
        stage: str,
        *,
        request_id: str = "",
        level: str = "info",
        summary: str = "",
        data: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.time()
        record: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "epoch": round(now, 3),
            "thread": threading.current_thread().name,
            "level": level,
            "stage": stage,
            "request_id": request_id,
            "summary": summary,
            "data": self._safe_json(data or {}, detailed=False),
        }
        if self.detailed and details:
            record["details"] = self._safe_json(details, detailed=True)

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                self._history.append(self._compact_record(record))
        except Exception as exc:
            logger.warning("stage=traffic_log_write_failed error=%s", exc)

        sink = self.event_sink
        if sink is not None:
            try:
                sink(self._compact_record(record))
            except Exception as exc:
                logger.debug("Adapter traffic event sink failed: %s", exc)

    @classmethod
    def _safe_json(cls, value: Any, *, detailed: bool) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            if detailed:
                return value
            return value if len(value) <= 1200 else value[:1200] + f"...<truncated {len(value) - 1200} chars>"
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace")
            return cls._safe_json(text, detailed=detailed)
        if isinstance(value, dict):
            return {str(key): cls._safe_json(item, detailed=detailed) for key, item in value.items()}
        if isinstance(value, (list, tuple, deque)):
            items = list(value)
            if not detailed and len(items) > 40:
                items = items[:40] + [f"...<truncated {len(value) - 40} items>"]
            return [cls._safe_json(item, detailed=detailed) for item in items]
        return cls._safe_json(str(value), detailed=detailed)

    @staticmethod
    def _compact_record(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ts": record.get("ts", ""),
            "level": record.get("level", ""),
            "stage": record.get("stage", ""),
            "request_id": record.get("request_id", ""),
            "summary": record.get("summary", ""),
            "data": record.get("data", {}),
        }


class OpenAIAdapter:
    def __init__(
        self,
        browser_worker: Optional[Any] = None,
        port: int = 8080,
        *,
        detailed_logging: Optional[bool] = None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        traffic_journal: Optional[AdapterTrafficJournal] = None,
    ):
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
        self._browser_request_lock = threading.Lock()
        self._admission_lock = threading.Lock()
        self._request_local = threading.local()
        self._request_states: Dict[str, Dict[str, Any]] = {}
        self._request_state_history: Deque[Dict[str, Any]] = deque(maxlen=TRAFFIC_HISTORY_LIMIT)
        self.queue_limit = ADAPTER_QUEUE_LIMIT
        self.circuit_failure_threshold = ADAPTER_CIRCUIT_FAILURE_THRESHOLD
        self.circuit_cooldown_sec = ADAPTER_CIRCUIT_COOLDOWN_SEC
        self._waiting_browser_requests = 0
        self._active_adapter_request_id = ""
        self._active_adapter_since: Optional[float] = None
        self._accepted_requests = 0
        self._completed_requests = 0
        self._rejected_backpressure = 0
        self._rejected_busy_timeout = 0
        self._circuit_state = "closed"
        self._circuit_failures = 0
        self._circuit_open_until = 0.0
        self._circuit_open_count = 0
        self._last_circuit_error = ""
        self._last_circuit_opened_at = 0.0
        self._context_buffers: Dict[str, Deque[str]] = {}
        self._context_buffer_chars: Dict[str, int] = {}
        self.new_chat_mode = os.environ.get("DEEPBILL_ADAPTER_NEW_CHAT_MODE", "auto").strip().lower()
        if self.new_chat_mode not in {"auto", "always", "never"}:
            self.new_chat_mode = "auto"
        self.default_reasoning_mode = normalize_reasoning_mode(os.environ.get("DEEPBILL_ADAPTER_REASONING_MODE", "off"))
        self._active_conversation_key: Optional[str] = None
        self.context_buffer_enabled = os.environ.get("DEEPBILL_ADAPTER_CONTEXT_BUFFER", "1") != "0"
        self.single_message_new_chat = os.environ.get("DEEPBILL_ADAPTER_SINGLE_MESSAGE_NEW_CHAT", "1") != "0"
        self._tool_parser = ToolCallParser()
        if traffic_journal is None:
            traffic_journal = AdapterTrafficJournal(
                detailed=TRAFFIC_LOG_DETAILED if detailed_logging is None else bool(detailed_logging),
                event_sink=event_sink,
            )
        else:
            traffic_journal.set_event_sink(event_sink)
            if detailed_logging is not None:
                traffic_journal.set_detailed(bool(detailed_logging))
        self.traffic_journal = traffic_journal
        self._register_routes()
        self._server_thread: Optional[threading.Thread] = None
        self._http_server: Optional[Any] = None
        self._running = False

    def set_detailed_logging(self, enabled: bool) -> None:
        self.traffic_journal.set_detailed(bool(enabled))

    def set_event_sink(self, event_sink: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        self.traffic_journal.set_event_sink(event_sink)

    def get_traffic_log_path(self) -> str:
        return str(self.traffic_journal.path)

    def _traffic_event(
        self,
        stage: str,
        *,
        request_id: str = "",
        level: str = "info",
        summary: str = "",
        data: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.traffic_journal.event(
            stage,
            request_id=request_id,
            level=level,
            summary=summary,
            data=data,
            details=details,
        )

    def _set_request_state(self, request_id: str, state: str, **data: Any) -> None:
        if not request_id:
            return
        now = time.time()
        with self._admission_lock:
            item = self._request_states.setdefault(
                request_id,
                {"request_id": request_id, "created_at": now, "history": []},
            )
            item.update({"state": state, "updated_at": now, **data})
            item["history"].append({"state": state, "at": round(now, 3), **data})
            snapshot = {
                "request_id": request_id,
                "state": state,
                "updated_at": round(now, 3),
                **data,
            }
            self._request_state_history.append(snapshot)
        self._traffic_event(
            "request.state",
            request_id=request_id,
            summary=state,
            data={"state": state, **data},
        )

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
                        {
                            "id": DEFAULT_MODEL,
                            "object": "model",
                            "created": now,
                            "owned_by": "deepbill",
                            "context_length": NORMAL_CONTEXT_WINDOW_TOKENS,
                            "max_context_tokens": NORMAL_CONTEXT_WINDOW_TOKENS,
                        },
                        {
                            "id": "deepbill",
                            "object": "model",
                            "created": now,
                            "owned_by": "deepbill",
                            "context_length": NORMAL_CONTEXT_WINDOW_TOKENS,
                            "max_context_tokens": NORMAL_CONTEXT_WINDOW_TOKENS,
                        },
                        {
                            "id": AUTO_MODEL,
                            "object": "model",
                            "created": now,
                            "owned_by": "deepbill",
                            "context_length": REASONING_CONTEXT_WINDOW_TOKENS,
                            "max_context_tokens": REASONING_CONTEXT_WINDOW_TOKENS,
                        },
                        {
                            "id": REASONING_MODEL,
                            "object": "model",
                            "created": now,
                            "owned_by": "deepbill",
                            "context_length": REASONING_CONTEXT_WINDOW_TOKENS,
                            "max_context_tokens": REASONING_CONTEXT_WINDOW_TOKENS,
                        },
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
                    diagnostics = payload.setdefault("diagnostics", {})
                    if isinstance(diagnostics, dict):
                        diagnostics["adapter"] = self._adapter_diagnostics()
                        diagnostics["traffic_log"] = self.traffic_journal.diagnostics()
                    return payload
                except Exception as exc:
                    return {"status": "error", "ready": False, "error": str(exc)}, 503
            return {
                "status": "ok" if ready else "error",
                "ready": ready,
                "diagnostics": {
                    "adapter": self._adapter_diagnostics(),
                    "traffic_log": self.traffic_journal.diagnostics(),
                },
            }

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

    @staticmethod
    def _error_response(
        message: str,
        error_type: str,
        status: int,
        retry_after: Optional[float] = None,
    ):
        resp = jsonify({"error": {"message": message, "type": error_type}})
        if retry_after is not None:
            resp.headers["Retry-After"] = str(max(1, int(float(retry_after) + 0.999)))
        return resp, status

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

    def _adapter_diagnostics(self) -> Dict[str, Any]:
        with self._admission_lock:
            active_for = (
                time.monotonic() - self._active_adapter_since
                if self._active_adapter_since is not None
                else 0.0
            )
            retry_after = max(0.0, self._circuit_open_until - time.monotonic())
            return {
                "queue_limit": self.queue_limit,
                "busy_timeout_sec": BROWSER_BUSY_TIMEOUT_SEC,
                "waiting_requests": self._waiting_browser_requests,
                "active_request_id": self._active_adapter_request_id,
                "active_for_sec": round(active_for, 1),
                "accepted_requests": self._accepted_requests,
                "completed_requests": self._completed_requests,
                "rejected_backpressure": self._rejected_backpressure,
                "rejected_busy_timeout": self._rejected_busy_timeout,
                "circuit_state": self._circuit_state,
                "circuit_failures": self._circuit_failures,
                "circuit_failure_threshold": self.circuit_failure_threshold,
                "circuit_cooldown_sec": self.circuit_cooldown_sec,
                "circuit_retry_after_sec": round(retry_after, 1),
                "circuit_open_count": self._circuit_open_count,
                "last_circuit_error": self._last_circuit_error,
                "last_circuit_opened_at": round(self._last_circuit_opened_at, 3),
                "request_state_tail": list(self._request_state_history)[-20:],
                "active_request_state": (
                    dict(self._request_states.get(self._active_adapter_request_id, {}))
                    if self._active_adapter_request_id
                    else {}
                ),
            }

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
    def _has_tool_result_messages(messages: List[Dict[str, Any]]) -> bool:
        return any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in messages)

    @staticmethod
    def _looks_like_meta_reasoning_answer(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
        if not normalized:
            return False
        strong_prefixes = (
            "we need to",
            "now we need",
            "i need to",
            "need to respond",
            "need to answer",
            "answer as the assistant",
            "return only the final user-visible answer",
            "the final answer should",
            "the answer should",
            "the answer will be",
            "the user said",
            "the user asks",
            "the user wants",
            "the assistant already",
            "нужно ответить",
            "надо ответить",
            "теперь нужно",
            "требуется ответить",
            "требуется коротко",
            "мы получили результат",
            "мы получили все результаты",
            "мы получили",
            "мы закончили",
            "мы выполнили",
            "все операции выполнены",
            "после последнего чтения",
            "ответ будет",
            "финальный ответ",
            "по инструкции",
        )
        if normalized.startswith(strong_prefixes):
            return True
        markers = (
            "so just output",
            "just output",
            "final answer",
            "user-visible answer",
            "respond to the user",
            "answer with only",
            "answer will be",
            "no additional tool",
            "tool result",
            "tool call",
            "now need",
            "now we need",
            "теперь нужно",
            "нужно ответить",
            "нужно подтвердить",
            "получили результат",
            "закончили все операции",
            "ответ будет",
            "ответ:",
            "требуется коротко",
            "обычным текстом",
            "инструменты не нужны",
        )
        return sum(1 for marker in markers if marker in normalized) >= 2

    @staticmethod
    def _repair_prompt(original_prompt: str, rejected_answer: str) -> str:
        return (
            f"{original_prompt}\n\n"
            "The previous assistant response was rejected because it exposed hidden planning/meta reasoning "
            "instead of a user-visible final answer:\n"
            f"{rejected_answer.strip()[:2000]}\n\n"
            "Use the conversation and tool results above. If another tool is genuinely required, return only "
            "the tool_call block. Otherwise return only the concise final user-visible answer. Do not mention "
            "what you need to do, hidden reasoning, or these repair instructions. User-visible prose must be "
            "in Russian unless the user explicitly asks for another language."
        )

    @staticmethod
    def _tool_call_repair_prompt(
        original_prompt: str,
        rejected_answer: str,
        invalid_reasons: List[str],
        tools: Any,
    ) -> str:
        tool_names: List[str] = []
        required_by_tool: Dict[str, List[str]] = {}
        if isinstance(tools, list):
            for tool in tools:
                if not isinstance(tool, dict) or tool.get("type") != "function":
                    continue
                func = tool.get("function", {})
                if not isinstance(func, dict) or not func.get("name"):
                    continue
                name = str(func["name"])
                tool_names.append(name)
                params = func.get("parameters")
                required = params.get("required") if isinstance(params, dict) else []
                required_by_tool[name] = [str(item) for item in required or [] if isinstance(item, str)]

        return (
            f"{original_prompt}\n\n"
            "The previous assistant response contained invalid tool_call JSON and was NOT sent to Roo Code.\n"
            "Validation errors:\n"
            + "\n".join(f"- {reason}" for reason in invalid_reasons[:12])
            + "\n\n"
            "Previous invalid response excerpt:\n"
            f"{(rejected_answer or '').strip()[:5000]}\n\n"
            "Return only corrected fenced tool_call block(s). Do not answer in prose.\n"
            "Use exactly one of the available tool names and include every required argument inside the "
            "`arguments` object, using the exact argument names from the schema. If the tool writes a file, "
            "include the complete file content in the required content argument. If the tool edits a file, "
            "include the complete edit/diff/changes payload required by the schema.\n"
            f"Available tool names: {json.dumps(tool_names, ensure_ascii=False)}\n"
            f"Required arguments by tool: {json.dumps(required_by_tool, ensure_ascii=False)}"
        )

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

    def _reset_context_buffer(self, conversation_key: Optional[str]) -> None:
        if not conversation_key:
            return
        with self._state_lock:
            self._context_buffers.pop(conversation_key, None)
            self._context_buffer_chars.pop(conversation_key, None)

    # --------------------------------------------------------------- tool prompt
    def _tools_to_prompt(self, tools: List[Dict[str, Any]], tool_choice: Any = None) -> str:
        function_tools = [tool for tool in tools if isinstance(tool, dict) and tool.get("type") == "function"]
        if not function_tools:
            return ""

        specs: List[Dict[str, Any]] = []
        example_calls: List[Dict[str, Any]] = []
        for tool in function_tools:
            func = tool.get("function", {})
            if not isinstance(func, dict) or not func.get("name"):
                continue
            parameters = func.get("parameters", {"type": "object", "properties": {}})
            required = []
            properties = {}
            if isinstance(parameters, dict):
                raw_required = parameters.get("required") or []
                required = [str(item) for item in raw_required if isinstance(item, str)]
                raw_properties = parameters.get("properties") or {}
                properties = raw_properties if isinstance(raw_properties, dict) else {}
            specs.append(
                {
                    "name": func.get("name"),
                    "description": func.get("description", ""),
                    "parameters": parameters,
                    "strict": func.get("strict", False),
                }
            )
            if required:
                example_args = {
                    key: self._placeholder_for_tool_argument(str(func.get("name")), key, properties.get(key))
                    for key in required[:8]
                }
                example_calls.append({"name": func.get("name"), "arguments": example_args})
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

        examples_text = ""
        if example_calls:
            examples_text = (
                "\nValid examples using the exact required argument names from the schemas:\n"
                f"{json.dumps(example_calls[:8], ensure_ascii=False, indent=2)}\n"
            )

        return (
            "You have access to application-side tools. Use them when needed for file, terminal, "
            "workspace, browser, or external actions.\n"
            "This client uses native OpenAI/Roo Code tool calls through the adapter. The adapter will "
            "convert your fenced tool_call block into native tool_calls. Roo Code rejects missing "
            "native arguments.\n"
            "If a tool is needed, do not answer normally. Return only one or more fenced tool_call "
            "blocks, with valid JSON in each block:\n"
            "```tool_call\n"
            '{"name":"tool_name","arguments":{"arg":"value"}}\n'
            "```\n"
            "Use exactly one of the tool names listed below. Put every tool parameter inside the "
            "`arguments` object. Use the exact argument names from each tool schema, especially every "
            "name listed in `required`; do not rename `path` to `filepath` or `content` to `contents` "
            "unless the schema itself uses those names.\n"
            "When writing or editing a file, include the full file content or full edit payload inside "
            "the required JSON argument. Never put generated code outside the JSON object, never say "
            "that you will call a tool, and never omit required arguments such as `path`, `content`, "
            "`line_count`, `diff`, `changes`, or `command`.\n"
            "When an argument contains code, escape all JSON quotes and newlines; never paste the "
            "code outside the JSON object. Do not include UI words such as Copy or Download.\n"
            "The adapter will execute the tool and send the result back to you. After receiving tool "
            "results, continue the task or call another tool if needed.\n"
            f"{choice_note}\n"
            f"{examples_text}"
            "Available tools JSON:\n"
            f"{json.dumps(specs, ensure_ascii=False, indent=2)}"
        ).strip()

    @staticmethod
    def _placeholder_for_tool_argument(tool_name: str, key: str, schema: Any = None) -> Any:
        lower_key = key.lower()
        lower_tool = (tool_name or "").lower()
        if "path" in lower_key or lower_key in {"file", "filename"}:
            return "relative/path.ext"
        if lower_key in {"content", "contents", "text", "body"}:
            return "<complete file content>" if "file" in lower_tool or "write" in lower_tool else "<content>"
        if lower_key in {"diff", "patch", "changes"}:
            return "<complete edit payload>"
        if "command" in lower_key or lower_key == "cmd":
            return "python -m pytest"
        if isinstance(schema, dict):
            schema_type = schema.get("type")
            if schema_type == "integer":
                return 0
            if schema_type == "number":
                return 0
            if schema_type == "boolean":
                return False
            if schema_type == "array":
                return []
            if schema_type == "object":
                return {}
        return f"<{key}>"

    @staticmethod
    def _tool_specs_by_name(tools: Any) -> Dict[str, Dict[str, Any]]:
        specs: Dict[str, Dict[str, Any]] = {}
        if not isinstance(tools, list):
            return specs
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                continue
            func = tool.get("function", {})
            if not isinstance(func, dict) or not func.get("name"):
                continue
            specs[str(func["name"])] = func
        return specs

    @staticmethod
    def _required_tool_args(spec: Optional[Dict[str, Any]]) -> List[str]:
        if not isinstance(spec, dict):
            return []
        params = spec.get("parameters")
        if not isinstance(params, dict):
            return []
        required = params.get("required") or []
        return [str(item) for item in required if isinstance(item, str)]

    @staticmethod
    def _tool_arg_properties(spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(spec, dict):
            return {}
        params = spec.get("parameters")
        if not isinstance(params, dict):
            return {}
        properties = params.get("properties") or {}
        return properties if isinstance(properties, dict) else {}

    @staticmethod
    def _argument_is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        return False

    @classmethod
    def _normalize_tool_arguments_for_schema(cls, name: str, arguments: Any, spec: Optional[Dict[str, Any]]) -> Any:
        if isinstance(arguments, str):
            parsed = cls._parse_json_maybe(arguments)
            arguments = parsed if isinstance(parsed, (dict, list)) else {"input": str(parsed)}
        if not isinstance(arguments, dict):
            return arguments

        normalized = dict(arguments)
        properties = cls._tool_arg_properties(spec)
        required = set(cls._required_tool_args(spec))
        relevant_keys = set(properties) | required
        alias_groups = {
            "path": ("filepath", "file_path", "filename", "file"),
            "filepath": ("path", "file_path", "filename", "file"),
            "content": ("contents", "file_content", "body", "text", "code", "html"),
            "contents": ("content", "file_content", "body", "text", "code", "html"),
            "changes": ("diff", "patch", "replacement", "content", "contents"),
            "diff": ("changes", "patch", "replacement"),
            "patch": ("diff", "changes"),
            "command": ("cmd", "shell_command", "terminal_command"),
            "cmd": ("command", "shell_command", "terminal_command"),
        }
        for key in relevant_keys:
            if key in normalized and not cls._argument_is_missing(normalized.get(key)):
                continue
            for alias in alias_groups.get(key, ()):
                if alias in normalized and not cls._argument_is_missing(normalized.get(alias)):
                    normalized[key] = normalized[alias]
                    break
        if (
            "line_count" in relevant_keys
            and cls._argument_is_missing(normalized.get("line_count"))
            and not cls._argument_is_missing(normalized.get("content"))
        ):
            content_text = str(normalized.get("content") or "")
            normalized["line_count"] = 0 if content_text == "" else content_text.count("\n") + 1
        return normalized

    def _validate_raw_tool_call(
        self,
        raw_call: Any,
        tools: Any,
        *,
        require_known_tool: bool,
    ) -> Tuple[Optional[str], Any, Optional[str]]:
        name = str(getattr(raw_call, "name", "") or "").strip()
        if not name:
            return None, None, "tool call has no name"

        specs = self._tool_specs_by_name(tools)
        spec = specs.get(name)
        if require_known_tool and specs and spec is None:
            return name, None, f"tool {name!r} is not in the request tool list"

        arguments = self._normalize_tool_arguments_for_schema(name, getattr(raw_call, "arguments", {}), spec)
        if spec is not None:
            params = spec.get("parameters")
            params_type = params.get("type") if isinstance(params, dict) else None
            if params_type in {None, "object"} and not isinstance(arguments, dict):
                return name, arguments, f"tool {name!r} arguments must be a JSON object"
            if isinstance(arguments, dict):
                missing = [
                    key
                    for key in self._required_tool_args(spec)
                    if key not in arguments or self._argument_is_missing(arguments.get(key))
                ]
                if missing:
                    received = ", ".join(sorted(arguments.keys())) or "none"
                    return (
                        name,
                        arguments,
                        f"tool {name!r} is missing required argument(s): {', '.join(missing)}; received keys: {received}",
                    )
        return name, arguments, None

    def _raw_calls_to_tool_calls(
        self,
        raw_calls: List[Any],
        *,
        tools: Any = None,
        require_known_tool: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        tool_calls: List[Dict[str, Any]] = []
        invalid_reasons: List[str] = []
        for raw_call in raw_calls:
            name, arguments, error = self._validate_raw_tool_call(
                raw_call,
                tools,
                require_known_tool=require_known_tool,
            )
            if error:
                invalid_reasons.append(error)
                continue
            call = self._make_tool_call(str(name), arguments)
            if call:
                tool_calls.append(call)
        deduped = self._dedupe_tool_calls(tool_calls) or []
        return deduped, invalid_reasons

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

    def _dedupe_tool_calls(self, tool_calls: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        if not tool_calls:
            return None
        seen: set[tuple[str, str]] = set()
        unique: List[Dict[str, Any]] = []
        for call in tool_calls:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            if not isinstance(function, dict):
                continue
            key = (str(function.get("name", "")), str(function.get("arguments", "")))
            if key in seen:
                continue
            seen.add(key)
            unique.append(call)
        return unique or None

    def _extract_tool_calls(self, text: str) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        parsed = self._tool_parser.parse(text or "", allow_bare_json=True)
        tool_calls, _invalid_reasons = self._raw_calls_to_tool_calls(parsed.calls)
        return parsed.cleaned_text, tool_calls or None

    @staticmethod
    def _has_explicit_tool_marker(text: str) -> bool:
        return ToolCallParser.has_explicit_marker(text or "")

    def _parse_assistant_answer_with_validation(
        self,
        answer: str,
        allow_tool_calls: bool,
        tools: Any = None,
    ) -> Tuple[str, Optional[List[Dict[str, Any]]], List[str]]:
        if not allow_tool_calls and not self._tool_parser.has_explicit_marker(answer or ""):
            return (answer or "").strip(), None, []
        parsed = self._tool_parser.parse(answer or "", allow_bare_json=allow_tool_calls)
        require_known_tool = bool(tools) and allow_tool_calls
        tool_calls, invalid_reasons = self._raw_calls_to_tool_calls(
            parsed.calls,
            tools=tools,
            require_known_tool=require_known_tool,
        )
        if tool_calls:
            return parsed.cleaned_text, tool_calls, invalid_reasons
        if allow_tool_calls:
            if parsed.explicit_marker and not invalid_reasons:
                invalid_reasons.append("explicit tool_call marker did not contain a complete valid tool call")
            if parsed.explicit_marker and parsed.cleaned_text and self._looks_like_meta_reasoning_answer(parsed.cleaned_text):
                invalid_reasons.append("explicit tool_call marker was mixed with hidden/meta reasoning text")
            return parsed.cleaned_text, None, invalid_reasons
        return (answer or "").strip(), None, invalid_reasons

    def _parse_assistant_answer(
        self,
        answer: str,
        allow_tool_calls: bool,
        tools: Any = None,
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        cleaned, tool_calls, _invalid_reasons = self._parse_assistant_answer_with_validation(
            answer,
            allow_tool_calls=allow_tool_calls,
            tools=tools,
        )
        return cleaned, tool_calls

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

        if self.context_buffer_enabled and not self._has_history(messages) and not self._is_single_user_request(messages):
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
            "Answer as the assistant. Return only the final user-visible answer. Do not include analysis, "
            "reasoning, DeepThink text, hidden thoughts, or meta-commentary about how to answer. "
            "Write user-visible prose only in Russian unless the user explicitly asks for another language. "
            "Never write phrases like 'we need to answer', 'we received the tool result', 'the answer will be', "
            "'мы получили результат', 'мы закончили все операции', 'требуется ответить', 'теперь нужно', "
            "'ответ будет', or 'ответ:'. If tool use is required, output only the requested tool_call block(s)."
        )
        return "\n\n".join(prompt_parts), latest_user

    def _worker_reasoning_mode(self) -> str:
        worker = self.browser_worker
        if worker is not None:
            getter = getattr(worker, "get_reasoning_mode", None)
            if callable(getter):
                try:
                    return normalize_reasoning_mode(getter())
                except Exception:
                    pass
            if hasattr(worker, "reasoning_mode"):
                try:
                    return normalize_reasoning_mode(getattr(worker, "reasoning_mode"))
                except Exception:
                    pass
        return self.default_reasoning_mode

    @staticmethod
    def _model_requests_reasoning(model: str) -> bool:
        text = (model or "").strip().lower()
        return any(fragment in text for fragment in REASONING_FORCE_MODEL_FRAGMENTS)

    @staticmethod
    def _model_requests_auto(model: str) -> bool:
        text = (model or "").strip().lower()
        return any(fragment in text for fragment in REASONING_AUTO_MODEL_FRAGMENTS)

    def _auto_reasoning_decision(
        self,
        data: Dict[str, Any],
        prompt: str,
        latest_user: str,
    ) -> Tuple[bool, str]:
        messages = data.get("messages") or []
        message_count = len(messages) if isinstance(messages, list) else 0
        tools = data.get("tools") or []
        tools_count = len(tools) if isinstance(tools, list) else 0
        tool_message_count, tool_result_chars = (
            self._content_chars_by_role(messages, "tool") if isinstance(messages, list) else (0, 0)
        )
        prompt_tokens = self._prompt_usage_tokens(prompt, request_data=data)
        if prompt_tokens >= REASONING_CONTEXT_SOFT_LIMIT:
            return True, f"prompt_tokens>={REASONING_CONTEXT_SOFT_LIMIT}"

        complex_re = re.compile(
            r"(сложн|архитектур|рефактор|debug|traceback|тест|test|план|проанализ|анализ|"
            r"исправ|найди|причин|implement|fix|bug|error|ошиб|код|roo|agent|агент|"
            r"редакт|tool|инструмент|стабильн|завис|лог)",
            flags=re.IGNORECASE,
        )
        complex_task = bool(complex_re.search(latest_user or prompt[:4000]))
        if tools_count and complex_task:
            return True, "tools+complex_task"
        if message_count >= 6 and (tool_result_chars >= 1200 or tool_message_count >= 2):
            return True, "long_tool_history"
        if len(latest_user or "") >= 3000 and complex_task:
            return True, "large_complex_user_request"
        return False, "simple_request"

    def _decide_use_reasoning(
        self,
        data: Dict[str, Any],
        prompt: str,
        latest_user: str,
        model: str,
    ) -> Tuple[bool, str, str]:
        worker_mode = self._worker_reasoning_mode()
        if self._model_requests_reasoning(model):
            return True, "model", "forced_by_model"
        mode = "auto" if self._model_requests_auto(model) else worker_mode
        if mode == "on":
            return True, mode, "mode_on"
        if mode == "off":
            return False, mode, "mode_off"
        use_reasoning, reason = self._auto_reasoning_decision(data, prompt, latest_user)
        return use_reasoning, mode, reason

    @staticmethod
    def _reasoning_timeout(timeout: int, use_reasoning: bool) -> int:
        if not use_reasoning:
            return max(5, min(timeout, MAX_REQUEST_TIMEOUT_SEC))
        target = max(timeout, REASONING_TIMEOUT_SEC)
        return max(5, min(target, REASONING_MAX_TIMEOUT_SEC))

    # -------------------------------------------------------------- deepseek I/O
    def _require_worker(self):
        if self.browser_worker is None:
            raise RuntimeError("Browser worker is not attached to the adapter")
        return self.browser_worker

    def _raise_if_circuit_open_locked(self) -> None:
        if self.circuit_failure_threshold <= 0:
            self._circuit_state = "disabled"
            return
        now = time.monotonic()
        if self._circuit_open_until > now:
            retry_after = self._circuit_open_until - now
            raise AdapterCircuitOpenError(
                f"DeepSeek adapter circuit is open after repeated failures; retry in {retry_after:.1f}s",
                retry_after=retry_after,
            )
        if self._circuit_state == "open":
            self._circuit_state = "half_open"

    def _mark_request_active_locked(self, request_id: Optional[str]) -> None:
        self._active_adapter_request_id = str(request_id or "")
        self._active_adapter_since = time.monotonic()
        self._accepted_requests += 1

    def _acquire_browser_request(self, request_id: Optional[str] = None) -> None:
        depth = int(getattr(self._request_local, "browser_lock_depth", 0) or 0)
        if depth > 0:
            self._request_local.browser_lock_depth = depth + 1
            self._traffic_event(
                "queue.reentrant",
                request_id=str(request_id or ""),
                summary=f"depth={depth + 1}",
                data={"depth": depth + 1},
            )
            return

        timeout = max(0.0, BROWSER_BUSY_TIMEOUT_SEC)
        with self._admission_lock:
            self._raise_if_circuit_open_locked()

        acquired = self._browser_request_lock.acquire(blocking=False)
        if acquired:
            with self._admission_lock:
                try:
                    self._raise_if_circuit_open_locked()
                except Exception:
                    self._browser_request_lock.release()
                    raise
                self._mark_request_active_locked(request_id)
            self._request_local.browser_lock_depth = 1
            self._traffic_event(
                "queue.acquired",
                request_id=str(request_id or ""),
                summary="acquired immediately",
                data={"waiting": 0, "queue_limit": self.queue_limit},
            )
            return

        with self._admission_lock:
            self._raise_if_circuit_open_locked()
            if self._waiting_browser_requests >= self.queue_limit:
                self._rejected_backpressure += 1
                active_for = (
                    time.monotonic() - self._active_adapter_since
                    if self._active_adapter_since is not None
                    else 0.0
                )
                raise BrowserBusyError(
                    "DeepSeek browser request queue is full "
                    f"(active_for={active_for:.1f}s, waiting={self._waiting_browser_requests}, "
                    f"queue_limit={self.queue_limit})"
                )
            self._waiting_browser_requests += 1
            waiting_now = self._waiting_browser_requests

        self._traffic_event(
            "queue.wait",
            request_id=str(request_id or ""),
            summary=f"waiting active={self._active_adapter_request_id or 'unknown'}",
            data={
                "waiting": waiting_now,
                "queue_limit": self.queue_limit,
                "active_request_id": self._active_adapter_request_id,
                "timeout_sec": timeout,
            },
        )

        try:
            if timeout:
                acquired = self._browser_request_lock.acquire(timeout=timeout)
            else:
                acquired = self._browser_request_lock.acquire(blocking=False)
        finally:
            with self._admission_lock:
                self._waiting_browser_requests = max(0, self._waiting_browser_requests - 1)

        if not acquired:
            with self._admission_lock:
                self._rejected_busy_timeout += 1
            self._traffic_event(
                "queue.timeout",
                request_id=str(request_id or ""),
                level="warning",
                summary=f"busy for more than {timeout:g}s",
                data={"timeout_sec": timeout},
            )
            raise BrowserBusyError(
                f"DeepSeek browser is busy with another request for more than {timeout:g} seconds"
            )
        with self._admission_lock:
            try:
                self._raise_if_circuit_open_locked()
            except Exception:
                self._browser_request_lock.release()
                raise
            self._mark_request_active_locked(request_id)
        self._request_local.browser_lock_depth = 1
        self._traffic_event(
            "queue.acquired",
            request_id=str(request_id or ""),
            summary="acquired after wait",
            data={"queue_limit": self.queue_limit},
        )

    def _release_browser_request(self) -> None:
        depth = int(getattr(self._request_local, "browser_lock_depth", 0) or 0)
        if depth > 1:
            self._request_local.browser_lock_depth = depth - 1
            self._traffic_event(
                "queue.reentrant_release",
                request_id=str(getattr(self._request_local, "request_id", "") or ""),
                summary=f"depth={depth - 1}",
                data={"depth": depth - 1},
            )
            return
        self._request_local.browser_lock_depth = 0
        with self._admission_lock:
            request_id = self._active_adapter_request_id
            self._active_adapter_request_id = ""
            self._active_adapter_since = None
        self._browser_request_lock.release()
        self._traffic_event(
            "queue.released",
            request_id=request_id,
            summary="released",
        )

    def _record_adapter_success(self) -> None:
        with self._admission_lock:
            self._completed_requests += 1
            if self.circuit_failure_threshold > 0:
                self._circuit_state = "closed"
                self._circuit_failures = 0
                self._circuit_open_until = 0.0
                self._last_circuit_error = ""

    def _record_adapter_failure(self, exc: Exception) -> None:
        if self.circuit_failure_threshold <= 0:
            return
        with self._admission_lock:
            self._circuit_failures += 1
            self._last_circuit_error = str(exc)
            if self._circuit_failures >= self.circuit_failure_threshold:
                self._circuit_state = "open"
                self._circuit_open_until = time.monotonic() + self.circuit_cooldown_sec
                self._circuit_open_count += 1
                self._last_circuit_opened_at = time.time()
                self._traffic_event(
                    "adapter.circuit_open",
                    level="warning",
                    summary=str(exc),
                    data={
                        "failures": self._circuit_failures,
                        "cooldown_sec": self.circuit_cooldown_sec,
                    },
                )
                logger.warning(
                    "stage=adapter_circuit_open failures=%s cooldown=%ss error=%s",
                    self._circuit_failures,
                    self.circuit_cooldown_sec,
                    exc,
                )

    def _maybe_new_chat(self, messages: Optional[List[Dict[str, Any]]] = None) -> None:
        messages = messages or []
        if not self._should_start_new_chat(messages):
            return
        self._reset_context_buffer(self._conversation_key(messages))
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
            "chat length limit",
            "length limit reached",
            "не принял сообщение",
            "поле ввода не очистилось",
        )
        return any(fragment in text for fragment in retryable_fragments)

    @staticmethod
    def _is_browser_busy_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "browser is busy" in text or "browser worker is recovering after timeout" in text

    def _force_new_chat(self, reason: str) -> bool:
        worker = self._require_worker()
        if not hasattr(worker, "new_chat"):
            return False
        try:
            worker.new_chat(timeout=30)
            self._traffic_event(
                "browser.new_chat",
                request_id=str(getattr(self._request_local, "request_id", "") or ""),
                summary=reason,
                data={"reason": reason},
            )
            logger.info("stage=deepseek_new_chat reason=%s", reason)
            return True
        except Exception as exc:
            self._traffic_event(
                "browser.new_chat_failed",
                request_id=str(getattr(self._request_local, "request_id", "") or ""),
                level="warning",
                summary=str(exc),
                data={"reason": reason, "error": str(exc)},
            )
            logger.warning("stage=deepseek_new_chat_failed reason=%s error=%s", reason, exc)
            return False

    @staticmethod
    def _worker_ask_text(
        worker: Any,
        prompt: str,
        timeout: int,
        use_reasoning: bool,
        request_id: Optional[str] = None,
    ) -> str:
        try:
            return str(
                worker.ask_text(
                    prompt=prompt,
                    timeout=timeout,
                    use_reasoning=use_reasoning,
                    request_id=request_id,
                )
                or ""
            ).strip()
        except TypeError as exc:
            if "request_id" not in str(exc) and "use_reasoning" not in str(exc):
                raise
        try:
            return str(worker.ask_text(prompt=prompt, timeout=timeout, use_reasoning=use_reasoning) or "").strip()
        except TypeError as exc:
            if "use_reasoning" not in str(exc):
                raise
            return str(worker.ask_text(prompt=prompt, timeout=timeout) or "").strip()

    @staticmethod
    def _worker_ask_text_stream(
        worker: Any,
        prompt: str,
        timeout: int,
        use_reasoning: bool,
        request_id: Optional[str] = None,
    ) -> Iterable[str]:
        if hasattr(worker, "ask_text_stream"):
            try:
                yield from worker.ask_text_stream(
                    prompt=prompt,
                    timeout=timeout,
                    use_reasoning=use_reasoning,
                    request_id=request_id,
                )
                return
            except TypeError as exc:
                if "request_id" not in str(exc) and "use_reasoning" not in str(exc):
                    raise
            try:
                yield from worker.ask_text_stream(prompt=prompt, timeout=timeout, use_reasoning=use_reasoning)
                return
            except TypeError as exc:
                if "use_reasoning" not in str(exc):
                    raise
                yield from worker.ask_text_stream(prompt=prompt, timeout=timeout)
                return
        yield OpenAIAdapter._worker_ask_text(worker, prompt, timeout, use_reasoning, request_id=request_id)

    def _ask_text(
        self,
        prompt: str,
        timeout: int,
        messages: Optional[List[Dict[str, Any]]] = None,
        use_reasoning: bool = False,
        request_id: Optional[str] = None,
    ) -> str:
        worker = self._require_worker()
        self._acquire_browser_request(request_id=request_id)
        try:
            self._maybe_new_chat(messages)
            attempts = DEEPSEEK_RETRY_ATTEMPTS + 1
            last_error: Optional[Exception] = None
            for attempt in range(1, attempts + 1):
                try:
                    self._traffic_event(
                        "browser.request",
                        request_id=str(request_id or ""),
                        summary=f"sync attempt={attempt}/{attempts}",
                        data={
                            "attempt": attempt,
                            "attempts": attempts,
                            "timeout": timeout,
                            "use_reasoning": bool(use_reasoning),
                            "prompt_chars": len(prompt or ""),
                        },
                        details={"prompt": prompt},
                    )
                    answer = self._worker_ask_text(worker, prompt, timeout, use_reasoning, request_id=request_id)
                    self._traffic_event(
                        "browser.response",
                        request_id=str(request_id or ""),
                        summary=f"answer_chars={len(answer or '')}",
                        data={"answer_chars": len(answer or ""), "attempt": attempt},
                        details={
                            "answer": answer,
                            "worker_diagnostics": self._worker_diagnostics_snapshot(worker),
                        },
                    )
                    self._record_adapter_success()
                    return answer
                except Exception as exc:
                    if self._is_browser_busy_error(exc):
                        raise BrowserBusyError(str(exc)) from exc
                    last_error = exc
                    if attempt >= attempts or not self._is_retryable_deepseek_error(exc):
                        raise
                    self._traffic_event(
                        "browser.retry",
                        request_id=str(request_id or ""),
                        level="warning",
                        summary=str(exc),
                        data={"next_attempt": attempt + 1, "attempts": attempts, "error": str(exc)},
                    )
                    logger.warning(
                        "stage=deepseek_retry mode=sync attempt=%s/%s error=%s",
                        attempt + 1,
                        attempts,
                        exc,
                    )
                    self._force_new_chat("retry_after_error")
            raise last_error or RuntimeError("DeepSeek request failed")
        except BrowserBusyError:
            raise
        except Exception as exc:
            self._record_adapter_failure(exc)
            raise
        finally:
            self._release_browser_request()

    def _ask_text_stream(
        self,
        prompt: str,
        timeout: int,
        messages: Optional[List[Dict[str, Any]]] = None,
        use_reasoning: bool = False,
        request_id: Optional[str] = None,
    ) -> Iterable[str]:
        worker = self._require_worker()
        self._acquire_browser_request(request_id=request_id)
        try:
            self._maybe_new_chat(messages)
            attempts = DEEPSEEK_RETRY_ATTEMPTS + 1
            for attempt in range(1, attempts + 1):
                try:
                    self._traffic_event(
                        "browser.request",
                        request_id=str(request_id or ""),
                        summary=f"stream attempt={attempt}/{attempts}",
                        data={
                            "attempt": attempt,
                            "attempts": attempts,
                            "timeout": timeout,
                            "use_reasoning": bool(use_reasoning),
                            "prompt_chars": len(prompt or ""),
                        },
                        details={"prompt": prompt},
                    )
                    if hasattr(worker, "ask_text_stream"):
                        yield from self._worker_ask_text_stream(
                            worker,
                            prompt,
                            timeout,
                            use_reasoning,
                            request_id=request_id,
                        )
                        self._record_adapter_success()
                        return
                    answer = self._worker_ask_text(worker, prompt, timeout, use_reasoning, request_id=request_id)
                    self._traffic_event(
                        "browser.response",
                        request_id=str(request_id or ""),
                        summary=f"answer_chars={len(answer or '')}",
                        data={"answer_chars": len(answer or ""), "attempt": attempt},
                        details={
                            "answer": answer,
                            "worker_diagnostics": self._worker_diagnostics_snapshot(worker),
                        },
                    )
                    for chunk in self._split_text(answer):
                        yield chunk
                    self._record_adapter_success()
                    return
                except Exception as exc:
                    if self._is_browser_busy_error(exc):
                        raise BrowserBusyError(str(exc)) from exc
                    if attempt >= attempts or not self._is_retryable_deepseek_error(exc):
                        raise
                    self._traffic_event(
                        "browser.retry",
                        request_id=str(request_id or ""),
                        level="warning",
                        summary=str(exc),
                        data={"next_attempt": attempt + 1, "attempts": attempts, "error": str(exc)},
                    )
                    logger.warning(
                        "stage=deepseek_retry mode=stream attempt=%s/%s error=%s",
                        attempt + 1,
                        attempts,
                        exc,
                    )
                    self._force_new_chat("retry_after_stream_error")
        except BrowserBusyError:
            raise
        except Exception as exc:
            self._record_adapter_failure(exc)
            raise
        finally:
            self._release_browser_request()

    @staticmethod
    def _worker_diagnostics_snapshot(worker: Any) -> Dict[str, Any]:
        diagnostics = getattr(worker, "diagnostics", None)
        if not callable(diagnostics):
            return {}
        try:
            value = diagnostics()
        except Exception as exc:
            return {"error": str(exc)}
        return value if isinstance(value, dict) else {"value": value}

    def _maybe_repair_meta_answer(
        self,
        cleaned: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        *,
        has_tools: bool,
        prompt: str,
        timeout: int,
        messages: List[Dict[str, Any]],
        use_reasoning: bool,
        request_id: str,
        tools: Any = None,
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        if tool_calls:
            return cleaned, tool_calls
        if not self._looks_like_meta_reasoning_answer(cleaned):
            return cleaned, tool_calls

        repair_prompt = self._repair_prompt(prompt, cleaned)
        repair_request_id = f"{request_id}-repair"
        logger.warning(
            "stage=repair_meta_answer request_id=%s cleaned_chars=%s",
            request_id,
            len(cleaned or ""),
        )
        self._traffic_event(
            "repair.meta.start",
            request_id=request_id,
            level="warning",
            summary="hidden/meta reasoning was rejected",
            data={"cleaned_chars": len(cleaned or ""), "has_tools": bool(has_tools)},
            details={"rejected_answer": cleaned, "repair_prompt": repair_prompt},
        )
        try:
            repaired = self._ask_text(
                repair_prompt,
                timeout=timeout,
                messages=messages,
                use_reasoning=use_reasoning,
                request_id=repair_request_id,
            )
            repaired_cleaned, repaired_tool_calls = self._parse_assistant_answer(
                repaired,
                allow_tool_calls=has_tools,
                tools=tools,
            )
            if repaired_tool_calls or (
                repaired_cleaned and not self._looks_like_meta_reasoning_answer(repaired_cleaned)
            ):
                logger.info(
                    "stage=repair_meta_answer_done request_id=%s repaired_chars=%s repaired_tool_calls=%s",
                    request_id,
                    len(repaired_cleaned or ""),
                    len(repaired_tool_calls or []),
                )
                self._traffic_event(
                    "repair.meta.done",
                    request_id=request_id,
                    summary=f"tool_calls={len(repaired_tool_calls or [])} chars={len(repaired_cleaned or '')}",
                    data={
                        "repaired_chars": len(repaired_cleaned or ""),
                        "tool_calls": len(repaired_tool_calls or []),
                    },
                    details={"repaired_answer": repaired},
                )
                return repaired_cleaned, repaired_tool_calls
            logger.warning(
                "stage=repair_meta_answer_rejected request_id=%s repaired_preview=%r",
                request_id,
                (repaired_cleaned or repaired)[:240],
            )
            self._traffic_event(
                "repair.meta.rejected",
                request_id=request_id,
                level="warning",
                summary="repair still looked invalid",
                data={"repaired_chars": len(repaired_cleaned or repaired or "")},
                details={"repaired_answer": repaired},
            )
        except Exception as exc:
            self._traffic_event(
                "repair.meta.failed",
                request_id=request_id,
                level="warning",
                summary=str(exc),
                data={"error": str(exc)},
            )
            logger.warning("stage=repair_meta_answer_failed request_id=%s error=%s", request_id, exc)
        return cleaned, tool_calls

    def _maybe_repair_invalid_tool_calls(
        self,
        answer: str,
        cleaned: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        invalid_reasons: List[str],
        *,
        has_tools: bool,
        prompt: str,
        timeout: int,
        messages: List[Dict[str, Any]],
        use_reasoning: bool,
        request_id: str,
        tools: Any = None,
    ) -> Tuple[str, Optional[List[Dict[str, Any]]], List[str]]:
        if not invalid_reasons:
            return cleaned, tool_calls, invalid_reasons
        if not has_tools:
            return cleaned, tool_calls, invalid_reasons

        repair_prompt = self._tool_call_repair_prompt(prompt, answer, invalid_reasons, tools)
        repair_request_id = f"{request_id}-tool-repair"
        logger.warning(
            "stage=repair_invalid_tool_calls request_id=%s invalid=%s",
            request_id,
            "; ".join(invalid_reasons[:4]),
        )
        self._traffic_event(
            "repair.tool_call.start",
            request_id=request_id,
            level="warning",
            summary="; ".join(invalid_reasons[:4]),
            data={"invalid_reasons": invalid_reasons[:20]},
            details={"rejected_answer": answer, "repair_prompt": repair_prompt},
        )
        try:
            repaired = self._ask_text(
                repair_prompt,
                timeout=timeout,
                messages=messages,
                use_reasoning=use_reasoning,
                request_id=repair_request_id,
            )
            repaired_cleaned, repaired_tool_calls, repaired_invalid = self._parse_assistant_answer_with_validation(
                repaired,
                allow_tool_calls=has_tools,
                tools=tools,
            )
            logger.info(
                "stage=repair_invalid_tool_calls_done request_id=%s parsed_tool_calls=%s invalid_after=%s",
                request_id,
                len(repaired_tool_calls or []),
                len(repaired_invalid or []),
            )
            self._traffic_event(
                "repair.tool_call.done",
                request_id=request_id,
                summary=f"tool_calls={len(repaired_tool_calls or [])} invalid={len(repaired_invalid or [])}",
                data={
                    "tool_calls": len(repaired_tool_calls or []),
                    "invalid_after": repaired_invalid,
                    "cleaned_chars": len(repaired_cleaned or ""),
                },
                details={"repaired_answer": repaired},
            )
            if repaired_tool_calls and not repaired_invalid:
                return repaired_cleaned, repaired_tool_calls, []
            if repaired_tool_calls:
                return repaired_cleaned, repaired_tool_calls, repaired_invalid
            if repaired_cleaned and not self._looks_like_meta_reasoning_answer(repaired_cleaned):
                return repaired_cleaned, None, repaired_invalid
            return cleaned, tool_calls, repaired_invalid or invalid_reasons
        except Exception as exc:
            self._traffic_event(
                "repair.tool_call.failed",
                request_id=request_id,
                level="warning",
                summary=str(exc),
                data={"error": str(exc), "invalid_reasons": invalid_reasons[:20]},
            )
            logger.warning("stage=repair_invalid_tool_calls_failed request_id=%s error=%s", request_id, exc)
            return cleaned, tool_calls, invalid_reasons

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
    def _estimate_tokens(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            text = str(value)
        if not text:
            return 0
        tokens = 0
        for part in re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE):
            if len(part) == 1 and not part.isalnum():
                tokens += 1
                continue
            tokens += max(1, (len(part) + 3) // 4)
        return tokens

    @classmethod
    def _prompt_usage_tokens(cls, prompt: str, request_data: Optional[Dict[str, Any]] = None) -> int:
        prompt_tokens = cls._estimate_tokens(prompt)
        if not request_data:
            return max(1, prompt_tokens)

        payload_tokens = 0
        messages = request_data.get("messages") or []
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                payload_tokens += 4
                payload_tokens += cls._estimate_tokens(msg)
        tools = request_data.get("tools") or []
        if isinstance(tools, list):
            for tool in tools:
                payload_tokens += 8
                payload_tokens += cls._estimate_tokens(tool)
        if request_data.get("tool_choice") is not None:
            payload_tokens += cls._estimate_tokens({"tool_choice": request_data.get("tool_choice")})
        return max(1, prompt_tokens, payload_tokens)

    @classmethod
    def _usage(
        cls,
        prompt: str,
        answer: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        request_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        prompt_tokens = cls._prompt_usage_tokens(prompt, request_data=request_data)
        completion_source: Any = tool_calls if tool_calls else answer
        completion_tokens = cls._estimate_tokens(completion_source) if completion_source else 0
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
        request_data: Optional[Dict[str, Any]] = None,
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
            "usage": self._usage(prompt, content or "", tool_calls=tool_calls, request_data=request_data),
        }

    def _handle_chat_completion(self):
        data = request.get_json(silent=True) or {}
        messages = data.get("messages") or []
        model = str(data.get("model") or DEFAULT_MODEL)
        stream = bool(data.get("stream", False))
        requested_timeout = int(data.get("timeout") or data.get("request_timeout") or DEFAULT_TIMEOUT_SEC)
        request_id = self._new_id()
        created = int(time.time())
        tool_message_count, tool_result_chars = self._content_chars_by_role(messages, "tool") if isinstance(messages, list) else (0, 0)
        client_request_id = str(request.headers.get("X-Client-Request-Id") or "")
        self._set_request_state(
            request_id,
            "received",
            model=model,
            stream=stream,
            client_request_id=client_request_id,
        )
        self._traffic_event(
            "http.chat.received",
            request_id=request_id,
            summary=f"model={model} stream={stream} messages={len(messages) if isinstance(messages, list) else 0}",
            data={
                "model": model,
                "stream": stream,
                "messages": len(messages) if isinstance(messages, list) else 0,
                "tools": len(data.get("tools") or []),
                "tool_messages": tool_message_count,
                "tool_result_chars": tool_result_chars,
                "client_request_id": client_request_id,
            },
            details={"request": data},
        )

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
            use_reasoning, reasoning_mode, reasoning_reason = self._decide_use_reasoning(data, prompt, latest_user, model)
            timeout = self._reasoning_timeout(requested_timeout, use_reasoning)
            logger.info(
                "stage=prompt_built request_id=%s prompt_chars=%s latest_user_chars=%s",
                request_id,
                len(prompt),
                len(latest_user),
            )
            self._set_request_state(
                request_id,
                "prompt_built",
                prompt_chars=len(prompt),
                latest_user_chars=len(latest_user),
                conversation_key=conversation_key or "",
            )
            self._traffic_event(
                "prompt.built",
                request_id=request_id,
                summary=f"prompt_chars={len(prompt)} latest_user_chars={len(latest_user)}",
                data={
                    "prompt_chars": len(prompt),
                    "latest_user_chars": len(latest_user),
                    "conversation_key": conversation_key or "",
                },
                details={"prompt": prompt},
            )
            logger.info(
                "stage=reasoning_route request_id=%s mode=%s use_reasoning=%s reason=%s timeout=%s",
                request_id,
                reasoning_mode,
                use_reasoning,
                reasoning_reason,
                timeout,
            )
            self._traffic_event(
                "reasoning.route",
                request_id=request_id,
                summary=f"use_reasoning={use_reasoning} reason={reasoning_reason}",
                data={
                    "mode": reasoning_mode,
                    "use_reasoning": bool(use_reasoning),
                    "reason": reasoning_reason,
                    "timeout": timeout,
                },
            )
        except Exception as exc:
            logger.exception("Invalid chat completion request")
            self._set_request_state(request_id, "failed", error=str(exc))
            self._traffic_event(
                "http.chat.invalid",
                request_id=request_id,
                level="error",
                summary=str(exc),
                data={"error": str(exc)},
            )
            return jsonify({"error": {"message": str(exc), "type": "invalid_request_error"}}), 400

        if not latest_user and not any(msg.get("role") == "tool" for msg in messages if isinstance(msg, dict)):
            self._set_request_state(request_id, "failed", error="No user or tool message")
            self._traffic_event(
                "http.chat.invalid",
                request_id=request_id,
                level="error",
                summary="No user or tool message",
            )
            return jsonify({"error": {"message": "No user or tool message", "type": "invalid_request_error"}}), 400

        if stream:
            return self._stream_chat_completion(
                data,
                prompt,
                latest_user,
                conversation_key,
                request_id,
                created,
                model,
                timeout,
                use_reasoning,
            )

        try:
            self._request_local.request_id = request_id
            self._set_request_state(request_id, "queued")
            self._acquire_browser_request(request_id=request_id)
            self._set_request_state(request_id, "active")
            logger.info(
                "stage=deepseek_request request_id=%s mode=sync timeout=%s reasoning=%s",
                request_id,
                timeout,
                use_reasoning,
            )
            answer = self._ask_text(
                prompt,
                timeout=timeout,
                messages=messages if isinstance(messages, list) else [],
                use_reasoning=use_reasoning,
                request_id=request_id,
            )
            logger.info("stage=deepseek_response request_id=%s answer_chars=%s preview=%r", request_id, len(answer), answer[:240])
            self._set_request_state(request_id, "validating", answer_chars=len(answer or ""))
            has_tools = bool(data.get("tools")) and data.get("tool_choice") != "none"
            tools = data.get("tools") or []
            cleaned, tool_calls, invalid_tool_calls = self._parse_assistant_answer_with_validation(
                answer,
                allow_tool_calls=has_tools,
                tools=tools,
            )
            cleaned, tool_calls, invalid_tool_calls = self._maybe_repair_invalid_tool_calls(
                answer,
                cleaned,
                tool_calls,
                invalid_tool_calls,
                has_tools=has_tools,
                prompt=prompt,
                timeout=timeout,
                messages=messages if isinstance(messages, list) else [],
                use_reasoning=use_reasoning,
                request_id=request_id,
                tools=tools,
            )
            cleaned, tool_calls = self._maybe_repair_meta_answer(
                cleaned,
                tool_calls,
                has_tools=has_tools,
                prompt=prompt,
                timeout=timeout,
                messages=messages if isinstance(messages, list) else [],
                use_reasoning=use_reasoning,
                request_id=request_id,
                tools=tools,
            )
            logger.info(
                "stage=parse_response request_id=%s has_tools=%s parsed_tool_calls=%s invalid_tool_calls=%s cleaned_chars=%s",
                request_id,
                has_tools,
                len(tool_calls or []),
                len(invalid_tool_calls or []),
                len(cleaned or ""),
            )
            response_payload = self._completion_response(request_id, created, model, prompt, cleaned, tool_calls, data)
            finish_reason = "tool_calls" if tool_calls else "stop"
            self._set_request_state(
                request_id,
                "completed",
                finish_reason=finish_reason,
                tool_calls=len(tool_calls or []),
                cleaned_chars=len(cleaned or ""),
                invalid_tool_calls=len(invalid_tool_calls or []),
            )
            self._traffic_event(
                "http.chat.completed",
                request_id=request_id,
                summary=f"finish={finish_reason} tool_calls={len(tool_calls or [])} chars={len(cleaned or '')}",
                data={
                    "finish_reason": finish_reason,
                    "tool_calls": len(tool_calls or []),
                    "cleaned_chars": len(cleaned or ""),
                    "invalid_tool_calls": invalid_tool_calls,
                },
                details={
                    "raw_answer": answer,
                    "cleaned": cleaned,
                    "tool_calls": tool_calls or [],
                    "response": response_payload,
                },
            )
            if not tool_calls:
                self._append_context_buffer(conversation_key, latest_user, cleaned)
            return jsonify(response_payload)
        except AdapterCircuitOpenError as exc:
            logger.warning("DeepSeek adapter circuit is open")
            self._set_request_state(request_id, "failed", error=str(exc), error_type="server_unavailable")
            self._traffic_event(
                "http.chat.failed",
                request_id=request_id,
                level="warning",
                summary=str(exc),
                data={"error_type": "server_unavailable", "error": str(exc)},
            )
            return self._error_response(str(exc), "server_unavailable", 503, retry_after=exc.retry_after)
        except BrowserBusyError as exc:
            logger.warning("DeepSeek browser is busy")
            self._set_request_state(request_id, "failed", error=str(exc), error_type="server_busy")
            self._traffic_event(
                "http.chat.failed",
                request_id=request_id,
                level="warning",
                summary=str(exc),
                data={"error_type": "server_busy", "error": str(exc)},
            )
            return self._error_response(str(exc), "server_busy", 429)
        except Exception as exc:
            logger.exception("DeepSeek request failed")
            self._set_request_state(request_id, "failed", error=str(exc), error_type="server_error")
            self._traffic_event(
                "http.chat.failed",
                request_id=request_id,
                level="error",
                summary=str(exc),
                data={"error_type": "server_error", "error": str(exc)},
            )
            return self._error_response(str(exc), "server_error", 500)
        finally:
            if int(getattr(self._request_local, "browser_lock_depth", 0) or 0) > 0:
                self._release_browser_request()
            self._request_local.request_id = ""

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
        use_reasoning: bool,
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

        def finish_chunk(reason: str, usage: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
            payload = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
            }
            if usage is not None:
                payload["usage"] = usage
            return payload

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
            self._request_local.request_id = request_id
            locked = False
            try:
                self._set_request_state(request_id, "queued")
                self._acquire_browser_request(request_id=request_id)
                locked = True
                self._set_request_state(request_id, "active_stream")
                yield self._sse(role_chunk())
                mode = "stream_tools_buffered" if has_tools else "stream_parse_buffered"
                logger.info(
                    "stage=deepseek_request request_id=%s mode=%s timeout=%s reasoning=%s",
                    request_id,
                    mode,
                    timeout,
                    use_reasoning,
                )
                collected = self._ask_text(
                    prompt,
                    timeout=timeout,
                    messages=data.get("messages") if isinstance(data.get("messages"), list) else [],
                    use_reasoning=use_reasoning,
                    request_id=request_id,
                ).strip()
                logger.info("stage=deepseek_response request_id=%s answer_chars=%s preview=%r", request_id, len(collected), collected[:240])
                self._set_request_state(request_id, "validating_stream", answer_chars=len(collected or ""))
                tools = data.get("tools") or []
                cleaned, tool_calls, invalid_tool_calls = self._parse_assistant_answer_with_validation(
                    collected,
                    allow_tool_calls=has_tools,
                    tools=tools,
                )
                cleaned, tool_calls, invalid_tool_calls = self._maybe_repair_invalid_tool_calls(
                    collected,
                    cleaned,
                    tool_calls,
                    invalid_tool_calls,
                    has_tools=has_tools,
                    prompt=prompt,
                    timeout=timeout,
                    messages=data.get("messages") if isinstance(data.get("messages"), list) else [],
                    use_reasoning=use_reasoning,
                    request_id=request_id,
                    tools=tools,
                )
                cleaned, tool_calls = self._maybe_repair_meta_answer(
                    cleaned,
                    tool_calls,
                    has_tools=has_tools,
                    prompt=prompt,
                    timeout=timeout,
                    messages=data.get("messages") if isinstance(data.get("messages"), list) else [],
                    use_reasoning=use_reasoning,
                    request_id=request_id,
                    tools=tools,
                )
                logger.info(
                    "stage=parse_response request_id=%s has_tools=%s parsed_tool_calls=%s invalid_tool_calls=%s cleaned_chars=%s",
                    request_id,
                    has_tools,
                    len(tool_calls or []),
                    len(invalid_tool_calls or []),
                    len(cleaned or ""),
                )
                if tool_calls:
                    self._set_request_state(
                        request_id,
                        "streaming_tool_calls",
                        tool_calls=len(tool_calls or []),
                        invalid_tool_calls=len(invalid_tool_calls or []),
                    )
                    self._traffic_event(
                        "http.chat.stream.completed",
                        request_id=request_id,
                        summary=f"finish=tool_calls tool_calls={len(tool_calls or [])}",
                        data={
                            "finish_reason": "tool_calls",
                            "tool_calls": len(tool_calls or []),
                            "invalid_tool_calls": invalid_tool_calls,
                        },
                        details={
                            "raw_answer": collected,
                            "cleaned": cleaned,
                            "tool_calls": tool_calls or [],
                        },
                    )
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
                    yield self._sse(finish_chunk("tool_calls", self._usage(prompt, "", tool_calls=tool_calls, request_data=data)))
                    yield self._sse("[DONE]")
                    self._set_request_state(request_id, "completed", finish_reason="tool_calls", tool_calls=len(tool_calls or []))
                    return

                self._set_request_state(request_id, "streaming_text", cleaned_chars=len(cleaned or ""))
                self._traffic_event(
                    "http.chat.stream.completed",
                    request_id=request_id,
                    summary=f"finish=stop chars={len(cleaned or '')}",
                    data={
                        "finish_reason": "stop",
                        "cleaned_chars": len(cleaned or ""),
                        "invalid_tool_calls": invalid_tool_calls,
                    },
                    details={"raw_answer": collected, "cleaned": cleaned},
                )
                for text_delta in self._split_text(cleaned):
                    yield self._sse(content_chunk(text_delta))
                self._append_context_buffer(conversation_key, latest_user, cleaned)
                yield self._sse(finish_chunk("stop", self._usage(prompt, cleaned, request_data=data)))
                yield self._sse("[DONE]")
                self._set_request_state(request_id, "completed", finish_reason="stop", cleaned_chars=len(cleaned or ""))
            except Exception as exc:
                logger.exception("Streaming DeepSeek request failed")
                if isinstance(exc, AdapterCircuitOpenError):
                    error_type = "server_unavailable"
                elif isinstance(exc, BrowserBusyError):
                    error_type = "server_busy"
                else:
                    error_type = "server_error"
                self._set_request_state(request_id, "failed", error=str(exc), error_type=error_type)
                self._traffic_event(
                    "http.chat.stream.failed",
                    request_id=request_id,
                    level="error" if error_type == "server_error" else "warning",
                    summary=str(exc),
                    data={"error_type": error_type, "error": str(exc)},
                )
                error_payload = {"error": {"message": str(exc), "type": error_type}}
                yield self._sse(error_payload)
                yield self._sse("[DONE]")
            finally:
                if locked and int(getattr(self._request_local, "browser_lock_depth", 0) or 0) > 0:
                    self._release_browser_request()
                self._request_local.request_id = ""

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
        requested_timeout = int(data.get("timeout") or DEFAULT_TIMEOUT_SEC)
        prompt_value = data.get("prompt", "")
        if isinstance(prompt_value, list):
            prompt = "\n".join(str(item) for item in prompt_value)
        else:
            prompt = str(prompt_value or "")
        request_id = self._new_id("cmpl")
        created = int(time.time())
        use_reasoning, _reasoning_mode, _reasoning_reason = self._decide_use_reasoning(
            {"messages": [{"role": "user", "content": prompt}]},
            prompt,
            prompt,
            model,
        )
        timeout = self._reasoning_timeout(requested_timeout, use_reasoning)

        if not prompt.strip():
            return jsonify({"error": {"message": "No prompt", "type": "invalid_request_error"}}), 400

        if stream:
            def generate() -> Iterator[str]:
                try:
                    for delta in self._ask_text_stream(
                        prompt,
                        timeout=timeout,
                        messages=[],
                        use_reasoning=use_reasoning,
                        request_id=request_id,
                    ):
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
                            "usage": self._usage(prompt, ""),
                        }
                    )
                    yield self._sse("[DONE]")
                except Exception as exc:
                    if isinstance(exc, AdapterCircuitOpenError):
                        error_type = "server_unavailable"
                    elif isinstance(exc, BrowserBusyError):
                        error_type = "server_busy"
                    else:
                        error_type = "server_error"
                    yield self._sse({"error": {"message": str(exc), "type": error_type}})
                    yield self._sse("[DONE]")

            return Response(stream_with_context(generate()), mimetype="text/event-stream")

        try:
            answer = self._ask_text(
                prompt,
                timeout=timeout,
                messages=[],
                use_reasoning=use_reasoning,
                request_id=request_id,
            )
        except AdapterCircuitOpenError as exc:
            return self._error_response(str(exc), "server_unavailable", 503, retry_after=exc.retry_after)
        except BrowserBusyError as exc:
            return self._error_response(str(exc), "server_busy", 429)
        except Exception as exc:
            return self._error_response(str(exc), "server_error", 500)
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

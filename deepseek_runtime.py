#!/usr/bin/env python3
"""Text-only DeepSeek browser runtime used by the DBill OpenAI adapter GUI."""

from __future__ import annotations

import logging
import json
import os
import queue
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Optional

from playwright.sync_api import BrowserContext, Page, sync_playwright
from tool_call_parser import TOOL_MARKER_RE, ToolCallParser, _skip_noise_after_marker, find_balanced_value


DEEPSEEK_URL = "https://chat.deepseek.com/"
DEFAULT_TIMEOUT_SEC = 360
DEFAULT_ANSWER_STABLE_SEC = 2.5
MIN_ANSWER_STABLE_SEC = 0.5
TOOL_CALL_REASONING_GRACE_SEC = 8.0
GENERATION_ACTIVE_GRACE_SEC = 20.0
WORKER_RESPONSE_GRACE_SEC = 60
DEFAULT_USER_DATA_DIR = "./deepseek_profile"
REASONING_MODES = {"off", "auto", "on"}
DEBUG_ARTIFACT_DIR = os.environ.get("DEEPBILL_DEBUG_ARTIFACT_DIR", "deepbill_debug")
DEBUG_ARTIFACT_LIMIT = max(1, int(os.environ.get("DEEPBILL_DEBUG_ARTIFACT_LIMIT", "20")))
REQUEST_JOURNAL_LIMIT = max(1, int(os.environ.get("DEEPBILL_REQUEST_JOURNAL_LIMIT", "20")))
WATCHDOG_HANG_LIMIT = max(1, int(os.environ.get("DEEPBILL_WATCHDOG_HANG_LIMIT", "3")))


class DeepSeekChatLengthLimitReached(RuntimeError):
    """Raised when DeepSeek asks the user to continue in a fresh web chat."""


def clamp_answer_stable_sec(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_ANSWER_STABLE_SEC
    return max(MIN_ANSWER_STABLE_SEC, min(parsed, 30.0))


def clamp_tool_call_reasoning_grace_sec(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = TOOL_CALL_REASONING_GRACE_SEC
    return max(MIN_ANSWER_STABLE_SEC, min(parsed, 120.0))


def clamp_generation_active_grace_sec(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = GENERATION_ACTIVE_GRACE_SEC
    return max(5.0, min(parsed, 180.0))


TOOL_CALL_REASONING_GRACE_SEC = clamp_tool_call_reasoning_grace_sec(
    os.environ.get("DEEPBILL_TOOL_CALL_REASONING_GRACE_SEC", TOOL_CALL_REASONING_GRACE_SEC)
)
GENERATION_ACTIVE_GRACE_SEC = clamp_generation_active_grace_sec(
    os.environ.get("DEEPBILL_GENERATION_ACTIVE_GRACE_SEC", GENERATION_ACTIVE_GRACE_SEC)
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
        "выключено": "off",
        "auto": "auto",
        "авто": "auto",
        "automatic": "auto",
        "1": "on",
        "true": "on",
        "yes": "on",
        "on": "on",
        "enabled": "on",
        "да": "on",
        "вкл": "on",
        "включено": "on",
    }
    return aliases.get(text, "off")


@dataclass
class AnswerSnapshot:
    final_text: str = ""
    reasoning_text: str = ""
    reasoning_active: bool = False
    web_search_active: bool = False
    web_search_seen: bool = False
    generation_active: bool = False
    raw_texts: list[str] = field(default_factory=list)


@dataclass
class RequestJournal:
    request_id: str
    prompt_chars: int
    timeout_sec: int
    use_reasoning: bool
    started_at: float = field(default_factory=time.monotonic)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"
    error: str = ""
    answer_chars: int = 0
    elapsed_sec: float = 0.0

    def add_event(self, stage: str, **details: Any) -> None:
        now = time.monotonic()
        self.events.append(
            {
                "stage": stage,
                "elapsed_sec": round(now - self.started_at, 3),
                **details,
            }
        )

    def finish(self, status: str, answer_chars: int = 0, error: str = "") -> None:
        self.status = status
        self.answer_chars = int(answer_chars or 0)
        self.error = str(error or "")
        self.elapsed_sec = round(time.monotonic() - self.started_at, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt_chars": self.prompt_chars,
            "timeout_sec": self.timeout_sec,
            "use_reasoning": self.use_reasoning,
            "status": self.status,
            "error": self.error,
            "answer_chars": self.answer_chars,
            "elapsed_sec": self.elapsed_sec or round(time.monotonic() - self.started_at, 3),
            "events": list(self.events[-80:]),
        }


class DeepSeekWebClient:
    INPUT_SELECTORS = [
        "textarea",
        "input[type='text']",
        "[role='textbox']",
        "[contenteditable='true']",
        "textarea[placeholder*='Message']",
        "textarea[placeholder*='Send']",
        "textarea[placeholder*='Введите']",
        "[aria-label*='Message']",
        "[aria-label*='Ask']",
        "[aria-label*='Введите']",
        "div[contenteditable='true']",
        "#chat-input",
        "[data-testid='chat-input']",
        "[data-testid*='composer']",
        "[class*='composer']",
    ]
    SEND_BUTTON_SELECTORS = [
        "button[type='submit']",
        "button:has-text('Send')",
        "button:has-text('Отправить')",
        "button[aria-label*='Send']",
        "button[aria-label*='Отправить']",
    ]
    ASSISTANT_MESSAGE_SELECTORS = [
        "[data-testid*='assistant']",
        "[data-role='assistant']",
        "[class*='assistant']",
        "[class*='assistant'] [class*='markdown']",
        "[class*='assistant'] [class*='prose']",
        "[class*='markdown']",
        "[class*='prose']",
        "article",
        "div.markdown-body",
    ]
    USER_MESSAGE_SELECTORS = [
        "[data-testid*='user']",
        "[data-role='user']",
        "[class*='user']",
        "[class*='human']",
        "[class*='question']",
        "[class*='message']",
        "article",
    ]
    NEW_CHAT_SELECTORS = [
        "a:has-text('New Chat')",
        "button:has-text('New Chat')",
        "a:has-text('Новый чат')",
        "button:has-text('Новый чат')",
        "button[aria-label*='New Chat' i]",
        "button[aria-label*='Новый чат' i]",
        "button[title*='New Chat' i]",
        "button[title*='Новый чат' i]",
        "a[aria-label*='New Chat' i]",
        "a[aria-label*='Новый чат' i]",
        "a[title*='New Chat' i]",
        "a[title*='Новый чат' i]",
        "[role='button'][aria-label*='New Chat' i]",
        "[role='button'][aria-label*='Новый чат' i]",
        "[role='button'][title*='New Chat' i]",
        "[role='button'][title*='Новый чат' i]",
    ]
    CONTINUE_BUTTON_SELECTORS = [
        "button:has-text('Continue')",
        "button:has-text('Продолжить')",
        "[role='button']:has-text('Continue')",
        "[role='button']:has-text('Продолжить')",
    ]
    REASONING_TOGGLE_SELECTORS = [
        "button:has-text('DeepThink')",
        "button:has-text('Deep Think')",
        "button:has-text('R1')",
        "button:has-text('Reason')",
        "button:has-text('Reasoning')",
        "button:has-text('Глубокое мышление')",
        "button:has-text('Рассуждения')",
        "[role='button']:has-text('DeepThink')",
        "[role='button']:has-text('R1')",
        "[aria-label*='DeepThink' i]",
        "[aria-label*='Reason' i]",
        "[aria-label*='Рассужд' i]",
        "[title*='DeepThink' i]",
        "[title*='Reason' i]",
        "[title*='Рассужд' i]",
    ]
    WEB_SEARCH_STATUS_SELECTORS = [
        "[aria-label*='Search' i]",
        "[aria-label*='Поиск' i]",
        "[title*='Search' i]",
        "[title*='Поиск' i]",
        "[class*='search' i]",
        "[class*='web' i]",
    ]

    def __init__(
        self,
        headless: bool = False,
        user_data_dir: str = DEFAULT_USER_DATA_DIR,
        answer_stable_sec: float = DEFAULT_ANSWER_STABLE_SEC,
    ):
        self.headless = headless
        self.user_data_dir = user_data_dir
        self.answer_stable_sec = clamp_answer_stable_sec(answer_stable_sec)
        self.last_continue_clicks = 0
        self.total_continue_clicks = 0
        self.last_use_reasoning = False
        self.last_reasoning_chars = 0
        self.last_reasoning_active = False
        self.last_reasoning_wait_sec = 0.0
        self.last_web_search_seen = False
        self.last_web_search_active = False
        self.last_web_search_wait_sec = 0.0
        self.last_reasoning_toggle_found = False
        self.last_debug_artifact_dir = ""
        self.last_debug_artifact_error = ""
        self.last_request_journal: dict[str, Any] = {}
        self.request_journal_history: Deque[dict[str, Any]] = deque(maxlen=REQUEST_JOURNAL_LIMIT)
        self._active_journal: Optional[RequestJournal] = None
        self._last_answer_state: Optional[tuple[bool, bool, bool, int]] = None
        self._cancel_event: Optional[threading.Event] = None
        self._request_lock = threading.Lock()
        self._playwright = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    def set_answer_stable_sec(self, value: Any) -> float:
        self.answer_stable_sec = clamp_answer_stable_sec(value)
        return self.answer_stable_sec

    def _start_request_journal(
        self,
        request_id: Optional[str],
        prompt: str,
        timeout_sec: int,
        use_reasoning: bool,
    ) -> None:
        rid = str(request_id or "").strip() or f"browser-{uuid.uuid4().hex[:16]}"
        self._active_journal = RequestJournal(
            request_id=rid,
            prompt_chars=len(prompt or ""),
            timeout_sec=int(timeout_sec or DEFAULT_TIMEOUT_SEC),
            use_reasoning=bool(use_reasoning),
        )
        self._last_answer_state = None
        self._journal_event("start")

    def _journal_event(self, stage: str, **details: Any) -> None:
        journal = self._active_journal
        if journal is None:
            return
        safe_details: dict[str, Any] = {}
        for key, value in details.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe_details[key] = value
            elif isinstance(value, (list, tuple)):
                safe_details[key] = list(value[:20])
            elif isinstance(value, dict):
                safe_details[key] = {str(k): v for k, v in list(value.items())[:20]}
            else:
                safe_details[key] = str(value)
        journal.add_event(stage, **safe_details)

    def _finish_request_journal(self, status: str, answer_chars: int = 0, error: str = "") -> None:
        journal = self._active_journal
        if journal is None:
            return
        journal.finish(status=status, answer_chars=answer_chars, error=error)
        payload = journal.to_dict()
        self.last_request_journal = payload
        self.request_journal_history.append(payload)
        self._active_journal = None
        self._last_answer_state = None

    def start(self) -> None:
        self._playwright = sync_playwright().start()
        self._context = self._playwright.chromium.launch_persistent_context(
            user_data_dir=self.user_data_dir,
            headless=self.headless,
            viewport={"width": 1365, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
        )
        pages = self._context.pages
        self._page = pages[0] if pages else self._context.new_page()
        self.open_chat_page()
        try:
            self._wait_for_input_ready(timeout=120)
        except Exception:
            logging.warning("DeepSeek input is not ready yet. Manual login may be required.")

    def close(self) -> None:
        try:
            if self._context is not None:
                self._context.close()
        finally:
            if self._playwright is not None:
                self._playwright.stop()

    def cancel_active_request(self) -> None:
        cancel_event = self._cancel_event
        if cancel_event is not None:
            cancel_event.set()

    def _raise_if_cancelled(self) -> None:
        cancel_event = self._cancel_event
        if cancel_event is not None and cancel_event.is_set():
            raise TimeoutError("DeepSeek browser request was cancelled")

    def open_chat_page(self) -> None:
        if self._page is None:
            raise RuntimeError("DeepSeek page is not initialized")
        current = str(self._page.url or "")
        if "chat.deepseek.com" not in current:
            self._page.goto(DEEPSEEK_URL, wait_until="domcontentloaded", timeout=60000)
            time.sleep(1.0)
        try:
            self._page.bring_to_front()
        except Exception:
            pass

    def send_text_prompt(
        self,
        prompt: str,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        use_reasoning: bool = False,
        request_id: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        message = (prompt or "").strip()
        if not message:
            raise ValueError("Empty text prompt")

        with self._request_lock:
            self._cancel_event = cancel_event
            self._reset_answer_diagnostics(use_reasoning=use_reasoning)
            self._start_request_journal(request_id, message, timeout_sec, use_reasoning)
            try:
                for attempt in range(2):
                    self._raise_if_cancelled()
                    self._journal_event("attempt_start", attempt=attempt + 1)
                    if self._page_has_chat_length_limit_notice():
                        logging.info("DeepSeek chat length limit was already visible; opening a fresh chat.")
                        self._journal_event("length_limit_notice_before_send")
                        self._start_new_chat_locked()

                    self._raise_if_cancelled()
                    self._wait_for_chat_idle_before_send(timeout=min(60, max(10, timeout_sec)))
                    self._set_reasoning_enabled(bool(use_reasoning))
                    input_handle = self._wait_for_input_ready(timeout=60)
                    self._wait_for_chat_idle_before_send(timeout=min(60, max(10, timeout_sec)))
                    self._raise_if_cancelled()
                    self._journal_event("input_ready")
                    previous_messages = self._get_assistant_messages_texts()
                    previous_user_messages = self._get_user_messages_texts()
                    self._journal_event("previous_messages_read", count=len(previous_messages))
                    self._focus_and_fill_input(input_handle, message)
                    self._raise_if_cancelled()
                    self._journal_event("prompt_filled")
                    self._send_current_message(
                        input_handle,
                        previous_messages=previous_messages,
                        question=message,
                        previous_user_messages=previous_user_messages,
                    )
                    self._raise_if_cancelled()
                    self._journal_event("message_sent")
                    try:
                        self._journal_event("answer_wait_start")
                        answer = self._wait_for_new_answer(previous_messages, message, timeout_sec)
                    except DeepSeekChatLengthLimitReached:
                        if attempt:
                            raise
                        logging.info("DeepSeek chat reached the length limit; retrying in a fresh chat.")
                        self._journal_event("length_limit_retry")
                        self._start_new_chat_locked()
                        continue
                    if not self._is_chat_length_limit_text(answer):
                        self._journal_event("answer_ready", answer_chars=len(answer))
                        self._finish_request_journal("ok", answer_chars=len(answer))
                        return answer
                    if attempt:
                        raise DeepSeekChatLengthLimitReached("DeepSeek chat length limit reached after retry")
                    logging.info("DeepSeek returned its chat length limit text; retrying in a fresh chat.")
                    self._journal_event("length_limit_text_retry")
                    self._start_new_chat_locked()
                raise DeepSeekChatLengthLimitReached("DeepSeek chat length limit reached")
            except Exception as exc:
                self._journal_event("error", error=str(exc))
                self._finish_request_journal("error", error=str(exc))
                raise
            finally:
                self._cancel_event = None

    def _reset_answer_diagnostics(self, use_reasoning: bool) -> None:
        self.last_use_reasoning = bool(use_reasoning)
        self.last_reasoning_chars = 0
        self.last_reasoning_active = False
        self.last_reasoning_wait_sec = 0.0
        self.last_web_search_seen = False
        self.last_web_search_active = False
        self.last_web_search_wait_sec = 0.0

    def _set_reasoning_enabled(self, enabled: bool) -> None:
        if self._page is None:
            return

        result = None
        try:
            result = self._page.evaluate(
                """
                ({enabled, selectors}) => {
                  const keywords = [
                    'deepthink', 'deep think', 'r1', 'reason', 'reasoning',
                    'глубокое мышление', 'рассужд'
                  ];
                  const activeValues = new Set(['true', 'on', 'active', 'checked', 'selected', 'pressed']);
                  const inactiveValues = new Set(['false', 'off', 'inactive', 'unchecked', 'unselected']);
                  const visible = (el) => {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return style.display !== 'none' && style.visibility !== 'hidden'
                      && rect.width > 0 && rect.height > 0;
                  };
                  const textOf = (el) => [
                    el.innerText,
                    el.textContent,
                    el.getAttribute('aria-label'),
                    el.getAttribute('title'),
                    el.getAttribute('data-testid'),
                    el.getAttribute('class'),
                  ].join(' ').replace(/\\s+/g, ' ').trim().toLowerCase();
                  const matches = (el) => {
                    const text = textOf(el);
                    return text && keywords.some((sample) => text.includes(sample));
                  };
                  const stateOf = (el) => {
                    const attrs = [
                      el.getAttribute('aria-pressed'),
                      el.getAttribute('aria-selected'),
                      el.getAttribute('aria-checked'),
                      el.getAttribute('data-state'),
                      el.getAttribute('data-active'),
                      el.getAttribute('data-selected'),
                      el.getAttribute('data-checked'),
                    ].map((value) => (value || '').trim().toLowerCase()).filter(Boolean);
                    for (const value of attrs) {
                      if (activeValues.has(value)) return true;
                      if (inactiveValues.has(value)) return false;
                    }
                    const className = String(el.getAttribute('class') || '').toLowerCase();
                    if (/\\b(active|selected|checked|pressed)\\b/.test(className)) return true;
                    if (el.querySelector('[aria-pressed="true"],[aria-selected="true"],[aria-checked="true"]')) {
                      return true;
                    }
                    if (el.querySelector('[aria-pressed="false"],[aria-selected="false"],[aria-checked="false"]')) {
                      return false;
                    }
                    return null;
                  };
                  const candidateSelector = [
                    'button',
                    '[role="button"]',
                    'label',
                    '[aria-pressed]',
                    '[aria-selected]',
                    '[aria-checked]',
                    '[data-state]',
                    '[data-active]',
                    '[data-selected]',
                    '[data-checked]',
                    '[class*="toggle" i]',
                    '[class*="ds-toggle-button" i]'
                  ].join(',');
                  const candidates = [];
                  for (const selector of selectors) {
                    try {
                      for (const node of document.querySelectorAll(selector)) {
                        const target = node.closest(candidateSelector) || node;
                        if (visible(target) && matches(target) && !candidates.includes(target)) {
                          candidates.push(target);
                        }
                      }
                    } catch (_error) {}
                  }
                  for (const node of document.querySelectorAll(candidateSelector)) {
                    if (visible(node) && matches(node) && !candidates.includes(node)) {
                      candidates.push(node);
                    }
                  }
                  candidates.sort((left, right) => {
                    const l = left.getBoundingClientRect();
                    const r = right.getBoundingClientRect();
                    return (r.top - l.top) || (textOf(left).length - textOf(right).length);
                  });
                  for (const candidate of candidates) {
                    const state = stateOf(candidate);
                    if (state === enabled) {
                      return {found: true, changed: false, state};
                    }
                    if (state === null && !enabled) {
                      return {found: true, changed: false, state};
                    }
                    candidate.click();
                    return {found: true, changed: true, state};
                  }
                  return {found: false, changed: false, state: null};
                }
                """,
                {"enabled": bool(enabled), "selectors": self.REASONING_TOGGLE_SELECTORS},
            )
        except Exception as exc:
            logging.debug("DeepSeek reasoning toggle probe failed: %s", exc)

        found = bool(isinstance(result, dict) and result.get("found"))
        self.last_reasoning_toggle_found = found
        self._journal_event(
            "reasoning_toggle",
            requested=bool(enabled),
            found=found,
            changed=bool(isinstance(result, dict) and result.get("changed")),
        )
        if not found and enabled:
            logging.warning("DeepSeek reasoning toggle was not found; continuing without toggling DeepThink.")
        elif isinstance(result, dict) and result.get("changed"):
            time.sleep(0.25)

    def start_new_chat(self) -> None:
        if self._page is None:
            raise RuntimeError("DeepSeek page is not initialized")
        with self._request_lock:
            self._start_new_chat_locked()

    def _start_new_chat_locked(self) -> None:
        if self._page is None:
            raise RuntimeError("DeepSeek page is not initialized")
        require_limit_notice_cleared = self._page_has_chat_length_limit_notice()
        if self._click_new_chat_control():
            time.sleep(0.8)
            try:
                self._wait_for_fresh_chat_ready(require_limit_notice_cleared, timeout=4)
                return
            except TimeoutError:
                logging.warning("DeepSeek new-chat control did not leave the exhausted chat; navigating to a fresh chat.")

        self._page.goto(DEEPSEEK_URL, wait_until="domcontentloaded", timeout=60000)
        time.sleep(1.0)
        self._wait_for_fresh_chat_ready(require_limit_notice_cleared, timeout=30)

    def _click_new_chat_control(self) -> bool:
        if self._page is None:
            raise RuntimeError("DeepSeek page is not initialized")
        for selector in self.NEW_CHAT_SELECTORS:
            try:
                button = self._page.query_selector(selector)
                if button is None or not button.is_visible():
                    continue
                button.click()
                return True
            except Exception:
                continue
        return bool(
            self._page.evaluate(
                """
                () => {
                  const variants = ['new chat', 'новый чат'];
                  const visible = (el) => {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return style.display !== 'none' && style.visibility !== 'hidden'
                      && rect.width > 0 && rect.height > 0;
                  };
                  const matches = (item) => {
                    const values = [
                      item.innerText,
                      item.textContent,
                      item.getAttribute('aria-label'),
                      item.getAttribute('title'),
                    ];
                    return values.some((value) => {
                      const normalized = (value || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                      return normalized && variants.some((sample) => normalized.includes(sample));
                    });
                  };
                  for (const item of document.querySelectorAll('a,button,[role="button"]')) {
                    if (!visible(item) || !matches(item)) continue;
                    item.click();
                    return true;
                  }
                  const divs = Array.from(document.querySelectorAll('div'))
                    .filter((item) => visible(item) && matches(item))
                    .sort((left, right) => {
                      const leftText = (left.innerText || left.textContent || '').trim();
                      const rightText = (right.innerText || right.textContent || '').trim();
                      return leftText.length - rightText.length;
                    });
                  for (const item of divs) {
                    const target = item.closest('a,button,[role="button"]') || item;
                    target.click();
                    return true;
                  }
                  return false;
                }
                """
            )
        )

    def _wait_for_fresh_chat_ready(self, require_limit_notice_cleared: bool, timeout: int) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._raise_if_cancelled()
            if require_limit_notice_cleared and self._page_has_chat_length_limit_notice():
                time.sleep(0.25)
                continue
            try:
                self._wait_for_input_ready(timeout=1)
            except TimeoutError:
                time.sleep(0.25)
                continue
            if not require_limit_notice_cleared or not self._page_has_chat_length_limit_notice():
                return
            time.sleep(0.25)
        if require_limit_notice_cleared:
            raise TimeoutError("DeepSeek fresh chat still shows the previous length-limit notice")
        raise TimeoutError("DeepSeek fresh chat input not found")

    def _wait_for_input_ready(self, timeout: int = 60):
        if self._page is None:
            raise RuntimeError("DeepSeek page is not initialized")
        deadline = time.monotonic() + timeout
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            self._raise_if_cancelled()
            for selector in self.INPUT_SELECTORS:
                try:
                    element = self._page.query_selector(selector)
                    if element is None:
                        continue
                    candidate = self._editable_input_from_element(element)
                    if candidate is not None and candidate.is_visible():
                        return candidate
                except Exception as exc:
                    last_error = exc
            time.sleep(0.35)
        if last_error is not None:
            raise TimeoutError(f"DeepSeek input not found: {last_error}")
        raise TimeoutError("DeepSeek input not found")

    def _wait_for_chat_idle_before_send(self, timeout: int = 60) -> None:
        if self._page is None:
            raise RuntimeError("DeepSeek page is not initialized")
        deadline = time.monotonic() + max(1, timeout)
        last_state: Optional[tuple[bool, bool, bool]] = None
        last_busy: dict[str, Any] = {}
        while time.monotonic() < deadline:
            self._raise_if_cancelled()
            dom = self._get_answer_dom_snapshot()
            generation_active = bool(dom.get("generation_active")) if isinstance(dom, dict) else False
            reasoning_active = bool(dom.get("reasoning_active")) if isinstance(dom, dict) else False
            web_search_active = bool(dom.get("web_search_active")) if isinstance(dom, dict) else False
            state = (generation_active, reasoning_active, web_search_active)
            if not any(state):
                if last_state and any(last_state):
                    self._journal_event("chat_idle_before_send")
                return
            if state != last_state:
                last_busy = {
                    "generation_active": generation_active,
                    "reasoning_active": reasoning_active,
                    "web_search_active": web_search_active,
                }
                self._journal_event("chat_busy_before_send", **last_busy)
                last_state = state
            time.sleep(0.35)
        raise RuntimeError(
            "DeepSeek chat is still busy before sending a new prompt: "
            + ", ".join(f"{key}={value}" for key, value in last_busy.items())
        )

    @staticmethod
    def _is_editable_input(element) -> bool:
        try:
            return bool(
                element.evaluate(
                    """
                    (node) => {
                      const tag = (node.tagName || '').toLowerCase();
                      if (tag === 'textarea' || tag === 'input') return !node.disabled && !node.readOnly;
                      if (node.isContentEditable) return true;
                      if ((node.getAttribute('contenteditable') || '').toLowerCase() === 'true') return true;
                      return (node.getAttribute('role') || '').toLowerCase() === 'textbox';
                    }
                    """
                )
            )
        except Exception:
            return False

    def _editable_input_from_element(self, element):
        if self._is_editable_input(element):
            return element
        try:
            inner = element.query_selector("textarea,input,[contenteditable='true'],[role='textbox']")
        except Exception:
            return None
        if inner is not None and self._is_editable_input(inner):
            return inner
        return None

    def _focus_and_fill_input(self, input_handle, text: str) -> None:
        if self._page is None:
            raise RuntimeError("DeepSeek page is not initialized")
        input_handle.click()
        self._page.keyboard.press("Control+A")
        self._page.keyboard.press("Backspace")
        methods = (
            self._fill_input_with_clipboard,
            self._fill_input_with_insert_text,
            self._fill_input_with_type,
            self._fill_input_fast,
            self._fill_input_with_dom,
        )
        last_error: Optional[Exception] = None
        for method in methods:
            try:
                method(input_handle, text)
                if self._input_matches_text(input_handle, text):
                    return
            except Exception as exc:
                last_error = exc
            self._clear_input(input_handle)
        raise RuntimeError(f"Unable to fill DeepSeek input. last_error={last_error!r}")

    def _clear_input(self, input_handle) -> None:
        try:
            input_handle.evaluate(
                """
                (el) => {
                  el.focus();
                  const tag = (el.tagName || '').toLowerCase();
                  if (tag === 'textarea' || tag === 'input') {
                    el.value = '';
                  } else {
                    el.textContent = '';
                    el.innerHTML = '';
                  }
                  el.dispatchEvent(new InputEvent('input', {bubbles:true, inputType:'deleteContentBackward'}));
                  el.dispatchEvent(new Event('change', {bubbles:true}));
                }
                """
            )
        except Exception:
            pass
        input_handle.click()
        self._page.keyboard.press("Control+A")
        self._page.keyboard.press("Backspace")

    def _fill_input_fast(self, input_handle, text: str) -> None:
        input_handle.fill(text, timeout=5000)

    def _fill_input_with_clipboard(self, input_handle, text: str) -> None:
        if self._page is None:
            raise RuntimeError("DeepSeek page is not initialized")
        try:
            if self._context is not None:
                self._context.grant_permissions(["clipboard-read", "clipboard-write"], origin=DEEPSEEK_URL)
        except Exception:
            pass
        self._page.evaluate("(value) => navigator.clipboard.writeText(value)", text)
        input_handle.click()
        self._page.keyboard.press("Control+V")
        time.sleep(0.2)

    def _fill_input_with_insert_text(self, input_handle, text: str) -> None:
        input_handle.click()
        for start in range(0, len(text), 8000):
            self._page.keyboard.insert_text(text[start : start + 8000])
            time.sleep(0.02)

    def _fill_input_with_type(self, input_handle, text: str) -> None:
        input_handle.click()
        for start in range(0, len(text), 4000):
            self._page.keyboard.type(text[start : start + 4000], delay=0)
            time.sleep(0.02)

    def _fill_input_with_dom(self, input_handle, text: str) -> None:
        input_handle.evaluate(
            """
            (el, value) => {
              el.focus();
              const tag = (el.tagName || '').toLowerCase();
              if (tag === 'textarea' || tag === 'input') {
                el.value = value;
              } else {
                el.textContent = value;
              }
              el.dispatchEvent(new InputEvent('input', {bubbles:true, inputType:'insertText', data:value}));
              el.dispatchEvent(new Event('change', {bubbles:true}));
            }
            """,
            text,
        )

    def _input_matches_text(self, input_handle, expected: str) -> bool:
        actual = self._get_input_text(input_handle)
        wanted = (expected or "").strip().replace("\r\n", "\n")
        got = (actual or "").strip().replace("\r\n", "\n")
        if got == wanted:
            return True
        if len(wanted) <= 500:
            return wanted in got or got in wanted
        return wanted[:160] in got and wanted[-160:] in got

    def _send_current_message(
        self,
        input_handle,
        previous_messages: Optional[list[str]] = None,
        question: str = "",
        previous_user_messages: Optional[list[str]] = None,
    ) -> None:
        if self._page is None:
            raise RuntimeError("DeepSeek page is not initialized")
        self._wait_for_send_enabled(timeout=5)
        for selector in self.SEND_BUTTON_SELECTORS:
            button = self._page.query_selector(selector)
            if button is None:
                continue
            try:
                ready = button.is_enabled() and button.is_visible()
            except Exception:
                continue
            if ready:
                button.click()
                time.sleep(0.5)
                self._ensure_message_submitted(input_handle, previous_messages, question, previous_user_messages)
                return
        input_handle.press("Enter")
        time.sleep(0.5)
        self._ensure_message_submitted(input_handle, previous_messages, question, previous_user_messages)

    def _wait_for_send_enabled(self, timeout: int = 5) -> bool:
        if self._page is None:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._raise_if_cancelled()
            for selector in self.SEND_BUTTON_SELECTORS:
                try:
                    button = self._page.query_selector(selector)
                    if button is not None and button.is_visible() and button.is_enabled():
                        return True
                except Exception:
                    continue
            time.sleep(0.25)
        return False

    def _ensure_message_submitted(
        self,
        input_handle,
        previous_messages: Optional[list[str]] = None,
        question: str = "",
        previous_user_messages: Optional[list[str]] = None,
        timeout: int = 5,
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._raise_if_cancelled()
            if self._submission_observed(input_handle, previous_messages or [], question, previous_user_messages or []):
                return
            time.sleep(0.25)
        for key in ("Enter", "Control+Enter"):
            try:
                input_handle.press(key)
            except Exception:
                pass
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if self._submission_observed(input_handle, previous_messages or [], question, previous_user_messages or []):
                    return
                time.sleep(0.25)
        raise RuntimeError("DeepSeek did not accept the message")

    def _submission_observed(
        self,
        input_handle,
        previous_messages: list[str],
        question: str,
        previous_user_messages: list[str],
    ) -> bool:
        if not self._get_input_text(input_handle):
            return True
        if self._user_message_observed(previous_user_messages, question):
            logging.info("DeepSeek submission confirmed by a new user message while composer was still populated.")
            return True
        return False

    def _user_message_observed(self, previous_user_messages: list[str], question: str) -> bool:
        user_messages = self._get_user_messages_texts()
        if not user_messages:
            return False
        previous_count = len(previous_user_messages or [])
        if len(user_messages) <= previous_count:
            return False
        latest = user_messages[-1]
        return self._text_matches_prompt(latest, question)

    @classmethod
    def _text_matches_prompt(cls, actual: str, expected: str) -> bool:
        got = (actual or "").strip().replace("\r\n", "\n")
        wanted = (expected or "").strip().replace("\r\n", "\n")
        if not got or not wanted:
            return False
        if cls._normalize_compare_text(got) == cls._normalize_compare_text(wanted):
            return True
        if len(wanted) <= 500:
            return wanted in got or got in wanted
        return wanted[:180] in got and wanted[-180:] in got

    @staticmethod
    def _get_input_text(input_handle) -> str:
        try:
            tag_name = input_handle.evaluate("el => el.tagName.toLowerCase()")
            if tag_name in {"textarea", "input"}:
                return str(input_handle.input_value() or "").strip()
            return str(input_handle.evaluate("el => (el.innerText || el.textContent || '').trim()") or "").strip()
        except Exception:
            return ""

    def _wait_for_new_answer(self, previous_messages: list[str], question: str, timeout: int) -> str:
        deadline = time.monotonic() + timeout
        candidate = ""
        last_seen = ""
        last_snapshot = AnswerSnapshot()
        last_change_ts = time.monotonic()
        continuation_prefix = ""
        continue_clicks = 0
        reasoning_active_since: Optional[float] = None
        web_search_active_since: Optional[float] = None
        tool_call_reasoning_ready_since: Optional[float] = None
        tool_call_reasoning_candidate = ""
        generation_active_ready_since: Optional[float] = None
        generation_active_candidate = ""
        reasoning_wait_total = 0.0
        web_search_wait_total = 0.0
        self.last_continue_clicks = 0

        while time.monotonic() < deadline:
            self._raise_if_cancelled()
            if self._page_has_chat_length_limit_notice():
                raise DeepSeekChatLengthLimitReached("DeepSeek chat length limit notice is visible")
            now = time.monotonic()
            snapshot = self._get_answer_snapshot(question, previous_messages)
            last_snapshot = snapshot
            text = snapshot.final_text
            if snapshot.reasoning_text:
                self.last_reasoning_chars = max(self.last_reasoning_chars, len(snapshot.reasoning_text))
            self.last_reasoning_active = bool(snapshot.reasoning_active)
            self.last_web_search_active = bool(snapshot.web_search_active)
            self.last_web_search_seen = self.last_web_search_seen or snapshot.web_search_seen or snapshot.web_search_active
            state = (
                bool(snapshot.reasoning_active),
                bool(snapshot.web_search_active),
                bool(snapshot.web_search_seen),
                bool(snapshot.generation_active),
                len(text or ""),
            )
            if state != self._last_answer_state:
                self._journal_event(
                    "answer_state",
                    final_chars=len(text or ""),
                    reasoning_active=bool(snapshot.reasoning_active),
                    reasoning_chars=len(snapshot.reasoning_text or ""),
                    web_search_active=bool(snapshot.web_search_active),
                    web_search_seen=bool(snapshot.web_search_seen),
                    generation_active=bool(snapshot.generation_active),
                    raw_count=len(snapshot.raw_texts or []),
                )
                self._last_answer_state = state

            if snapshot.reasoning_active:
                if reasoning_active_since is None:
                    reasoning_active_since = now
                self.last_reasoning_wait_sec = reasoning_wait_total + (now - reasoning_active_since)
            elif reasoning_active_since is not None:
                reasoning_wait_total += now - reasoning_active_since
                self.last_reasoning_wait_sec = reasoning_wait_total
                reasoning_active_since = None

            if snapshot.web_search_active:
                if web_search_active_since is None:
                    web_search_active_since = now
                self.last_web_search_wait_sec = web_search_wait_total + (now - web_search_active_since)
            elif web_search_active_since is not None:
                web_search_wait_total += now - web_search_active_since
                self.last_web_search_wait_sec = web_search_wait_total
                web_search_active_since = None

            if self._is_chat_length_limit_text(text):
                raise DeepSeekChatLengthLimitReached("DeepSeek chat length limit response was returned")
            if continuation_prefix and text:
                text = self._merge_continued_answer(continuation_prefix, text)
            if text:
                last_seen = text
                if text != candidate:
                    candidate = text
                    last_change_ts = now
                    tool_call_reasoning_ready_since = None
                    tool_call_reasoning_candidate = ""
                    generation_active_ready_since = None
                    generation_active_candidate = ""
                stable_for_answer = now - last_change_ts >= self.answer_stable_sec
                complete_tool_call = self._looks_like_complete_tool_call_answer(text)
                if (
                    not snapshot.web_search_active
                    and not snapshot.generation_active
                    and stable_for_answer
                    and not snapshot.reasoning_active
                ):
                    if self._click_continue_if_available():
                        continue_clicks += 1
                        self.last_continue_clicks = continue_clicks
                        self.total_continue_clicks += 1
                        continuation_prefix = candidate
                        last_change_ts = time.monotonic()
                        self._journal_event("continue_clicked", click=continue_clicks, answer_chars=len(candidate))
                        logging.info(
                            "DeepSeek continuation accepted: click=%s answer_chars=%s",
                            continue_clicks,
                            len(candidate),
                        )
                        time.sleep(0.5)
                        continue
                    return text
                if (
                    not snapshot.web_search_active
                    and not snapshot.reasoning_active
                    and snapshot.generation_active
                    and stable_for_answer
                ):
                    if generation_active_candidate != text or generation_active_ready_since is None:
                        generation_active_candidate = text
                        generation_active_ready_since = now
                        self._journal_event(
                            "stable_answer_waiting_for_generation_flag",
                            answer_chars=len(text),
                            grace_sec=GENERATION_ACTIVE_GRACE_SEC,
                        )
                    elif now - generation_active_ready_since >= GENERATION_ACTIVE_GRACE_SEC:
                        self._journal_event(
                            "stable_answer_returned_after_generation_grace",
                            answer_chars=len(text),
                            waited_sec=round(now - generation_active_ready_since, 3),
                        )
                        if self._click_continue_if_available():
                            continue_clicks += 1
                            self.last_continue_clicks = continue_clicks
                            self.total_continue_clicks += 1
                            continuation_prefix = candidate
                            last_change_ts = time.monotonic()
                            generation_active_ready_since = None
                            generation_active_candidate = ""
                            self._journal_event("continue_clicked", click=continue_clicks, answer_chars=len(candidate))
                            logging.info(
                                "DeepSeek continuation accepted after generation grace: click=%s answer_chars=%s",
                                continue_clicks,
                                len(candidate),
                            )
                            time.sleep(0.5)
                            continue
                        return text
                if (
                    not snapshot.web_search_active
                    and not snapshot.generation_active
                    and snapshot.reasoning_active
                    and stable_for_answer
                    and complete_tool_call
                ):
                    if tool_call_reasoning_candidate != text or tool_call_reasoning_ready_since is None:
                        tool_call_reasoning_candidate = text
                        tool_call_reasoning_ready_since = now
                        self._journal_event(
                            "complete_tool_call_waiting_for_reasoning_flag",
                            answer_chars=len(text),
                            grace_sec=TOOL_CALL_REASONING_GRACE_SEC,
                        )
                    elif now - tool_call_reasoning_ready_since >= TOOL_CALL_REASONING_GRACE_SEC:
                        self._journal_event(
                            "complete_tool_call_returned_after_reasoning_grace",
                            answer_chars=len(text),
                            waited_sec=round(now - tool_call_reasoning_ready_since, 3),
                        )
                        if self._click_continue_if_available():
                            continue_clicks += 1
                            self.last_continue_clicks = continue_clicks
                            self.total_continue_clicks += 1
                            continuation_prefix = candidate
                            last_change_ts = time.monotonic()
                            tool_call_reasoning_ready_since = None
                            tool_call_reasoning_candidate = ""
                            self._journal_event("continue_clicked", click=continue_clicks, answer_chars=len(candidate))
                            logging.info(
                                "DeepSeek continuation accepted after reasoning grace: click=%s answer_chars=%s",
                                continue_clicks,
                                len(candidate),
                            )
                            time.sleep(0.5)
                            continue
                        return text
            time.sleep(0.5)
        status = []
        if self.last_reasoning_active:
            status.append("reasoning_active")
        if self.last_web_search_active:
            status.append("web_search_active")
        if last_snapshot.generation_active:
            status.append("generation_active")
        suffix = f" status={','.join(status)}" if status else ""
        artifact = self._capture_timeout_artifacts(
            reason="answer_timeout",
            metadata={
                "timeout_sec": timeout,
                "status": status,
                "last_seen": last_seen[:2000],
                "candidate_chars": len(candidate or ""),
                "last_snapshot": {
                    "final_chars": len(last_snapshot.final_text or ""),
                    "reasoning_chars": len(last_snapshot.reasoning_text or ""),
                    "reasoning_active": last_snapshot.reasoning_active,
                    "web_search_active": last_snapshot.web_search_active,
                    "web_search_seen": last_snapshot.web_search_seen,
                    "generation_active": last_snapshot.generation_active,
                    "raw_count": len(last_snapshot.raw_texts or []),
                },
            },
        )
        if artifact:
            self._journal_event("debug_artifacts_saved", **artifact)
        raise TimeoutError(f"DeepSeek answer timed out{suffix}. Last fragment: {last_seen[:250]!r}")

    def _capture_timeout_artifacts(self, reason: str, metadata: dict[str, Any]) -> dict[str, Any]:
        request_id = self._active_journal.request_id if self._active_journal is not None else "unknown"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe_request_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", request_id)[:80]
        stem = f"{stamp}_{safe_request_id}_{uuid.uuid4().hex[:8]}_{reason}"
        artifact_dir = Path(DEBUG_ARTIFACT_DIR).expanduser().resolve()
        result: dict[str, Any] = {"debug_dir": str(artifact_dir)}
        try:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "reason": reason,
                "request_id": request_id,
                "url": str(getattr(self._page, "url", "") or ""),
                "diagnostics": {
                    "last_use_reasoning": self.last_use_reasoning,
                    "last_reasoning_chars": self.last_reasoning_chars,
                    "last_reasoning_active": self.last_reasoning_active,
                    "last_reasoning_wait_sec": round(self.last_reasoning_wait_sec, 3),
                    "last_web_search_seen": self.last_web_search_seen,
                    "last_web_search_active": self.last_web_search_active,
                    "last_web_search_wait_sec": round(self.last_web_search_wait_sec, 3),
                    "last_continue_clicks": self.last_continue_clicks,
                    "total_continue_clicks": self.total_continue_clicks,
                },
                "metadata": metadata,
                "journal": self._active_journal.to_dict() if self._active_journal is not None else {},
            }
            json_path = artifact_dir / f"{stem}.json"
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            result["metadata_path"] = str(json_path)

            if self._page is not None:
                try:
                    html_path = artifact_dir / f"{stem}.html"
                    html_path.write_text(self._page.content(), encoding="utf-8", errors="replace")
                    result["html_path"] = str(html_path)
                except Exception as exc:
                    result["html_error"] = str(exc)
                try:
                    screenshot_path = artifact_dir / f"{stem}.png"
                    self._page.screenshot(path=str(screenshot_path), full_page=True)
                    result["screenshot_path"] = str(screenshot_path)
                except Exception as exc:
                    result["screenshot_error"] = str(exc)
            self.last_debug_artifact_dir = str(artifact_dir)
            self.last_debug_artifact_error = ""
            self._prune_debug_artifacts(artifact_dir)
            logging.warning("DeepSeek timeout artifacts saved in %s", artifact_dir)
        except Exception as exc:
            self.last_debug_artifact_error = str(exc)
            result["debug_error"] = str(exc)
            logging.warning("DeepSeek timeout artifact capture failed: %s", exc)
        return result

    @staticmethod
    def _prune_debug_artifacts(artifact_dir: Path) -> None:
        try:
            json_files = sorted(artifact_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        except Exception:
            return
        for json_path in json_files[DEBUG_ARTIFACT_LIMIT:]:
            stem = json_path.stem
            for path in artifact_dir.glob(f"{stem}.*"):
                try:
                    path.unlink()
                except Exception:
                    pass

    def _get_answer_snapshot(self, question: str, previous_messages: list[str]) -> AnswerSnapshot:
        dom = self._get_answer_dom_snapshot()
        raw_texts = dom.get("final_texts") if isinstance(dom, dict) else None
        if not isinstance(raw_texts, list) or not raw_texts:
            raw_texts = self._get_assistant_messages_texts()
        raw_texts = [str(item or "").strip() for item in raw_texts if str(item or "").strip()]
        previous_answer = self._latest_assistant_message_text(previous_messages)
        previous = "" if len(raw_texts) > len(previous_messages) else previous_answer
        final_text = self._select_assistant_answer_text(raw_texts, question, previous)
        reasoning_text = ""
        reasoning_active = False
        web_search_active = False
        web_search_seen = False
        generation_active = False
        if isinstance(dom, dict):
            reasoning_text = str(dom.get("reasoning_text") or "").strip()
            reasoning_active = bool(dom.get("reasoning_active"))
            web_search_active = bool(dom.get("web_search_active"))
            web_search_seen = bool(dom.get("web_search_seen"))
            generation_active = bool(dom.get("generation_active"))
        return AnswerSnapshot(
            final_text=final_text,
            reasoning_text=reasoning_text,
            reasoning_active=reasoning_active,
            web_search_active=web_search_active,
            web_search_seen=web_search_seen,
            generation_active=generation_active,
            raw_texts=raw_texts,
        )

    def _get_answer_dom_snapshot(self) -> dict[str, Any]:
        if self._page is None:
            return {}
        js = """
        (options) => {
          const selectors = Array.isArray(options) ? options : (options.selectors || []);
          const inputSelectors = Array.isArray(options) ? [] : (options.input_selectors || []);
          const root = document.querySelector('main') || document.body;
          const clean = (value) => (value || '')
            .replace(/\\u00a0/g, ' ')
            .replace(/[ \\t]+\\n/g, '\\n')
            .replace(/\\n{3,}/g, '\\n\\n')
            .trim();
          const visible = (el) => {
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style.display !== 'none' && style.visibility !== 'hidden'
              && rect.width > 0 && rect.height > 0;
          };
          const textOf = (el) => clean([
            el.innerText,
            el.textContent,
            el.getAttribute && el.getAttribute('aria-label'),
            el.getAttribute && el.getAttribute('title'),
            el.getAttribute && el.getAttribute('data-testid'),
            el.getAttribute && el.getAttribute('class'),
          ].join(' ')).toLowerCase();
          const labelTextOf = (el) => clean([
            el.innerText,
            el.textContent,
            el.getAttribute && el.getAttribute('aria-label'),
            el.getAttribute && el.getAttribute('title'),
          ].join(' ')).toLowerCase();
          const hasAny = (text, samples) => samples.some((sample) => text.includes(sample));
          const reasoningTerms = [
            'reasoning', 'thinking', 'deepthink', 'deep think', 'thought process',
            'chain of thought', 'рассужд', 'размышл', 'думаю', 'мысл'
          ];
          const reasoningActiveTerms = [
            'thinking', 'generating', 'loading', 'working', 'думает',
            'идет', 'идёт', '...'
          ];
          const searchTerms = [
            'web search', 'searching', 'search web', 'internet', 'browsing',
            'sources', 'поиск', 'ищу', 'ищет', 'интернет', 'источник'
          ];
          const searchActiveTerms = [
            'searching', 'searching web', 'browsing', 'looking up', 'ищу',
            'ищет', 'поиск в интернете', 'поиск по интернету'
          ];
          const generationStopTerms = [
            'stop generating', 'stop response', 'stop answering', 'cancel response',
            'stop', 'cancel', 'остановить', 'прервать', 'стоп'
          ];
          const controlSelector = [
            'button', '[role="button"]', 'svg', 'canvas', 'input', 'textarea',
            'select', 'script', 'style', '[contenteditable="true"]',
            '[aria-pressed]', '[aria-selected]', '[aria-checked]',
            '[class*="toggle" i]', '[class*="ds-toggle-button" i]',
            '[aria-label*="copy" i]', '[aria-label*="download" i]',
            '[title*="copy" i]', '[title*="download" i]',
            '[aria-label*="копировать" i]', '[title*="копировать" i]'
          ].join(',');
          const interactiveSelector = [
            'button', '[role="button"]', 'label', 'input', 'textarea', 'select',
            '[contenteditable="true"]', '[aria-pressed]', '[aria-selected]',
            '[aria-checked]', '[class*="toggle" i]', '[class*="ds-toggle-button" i]'
          ].join(',');
          const isInteractiveControl = (node) => {
            try {
              return Boolean(node.closest && node.closest(interactiveSelector));
            } catch (_error) {
              return false;
            }
          };
          const isReasoningNode = (node) => {
            if (isInteractiveControl(node)) return false;
            const text = textOf(node);
            if (!text || !hasAny(text, reasoningTerms)) return false;
            const visibleText = clean(node.innerText || node.textContent || '');
            const classHint = String(node.getAttribute && node.getAttribute('class') || '').toLowerCase();
            const labelHint = [
              node.getAttribute && node.getAttribute('aria-label'),
              node.getAttribute && node.getAttribute('title'),
              node.getAttribute && node.getAttribute('data-testid'),
            ].join(' ').toLowerCase();
            if (hasAny(classHint + ' ' + labelHint, reasoningTerms)) return true;
            return visibleText.length <= 6000 && hasAny(visibleText.slice(0, 300).toLowerCase(), reasoningTerms);
          };
          const hasReasoningStructuralHint = (node) => {
            if (!node || isInteractiveControl(node)) return false;
            const classHint = String(node.getAttribute && node.getAttribute('class') || '').toLowerCase();
            const labelHint = [
              node.getAttribute && node.getAttribute('aria-label'),
              node.getAttribute && node.getAttribute('title'),
              node.getAttribute && node.getAttribute('data-testid'),
            ].join(' ').toLowerCase();
            return hasAny(classHint + ' ' + labelHint, reasoningTerms);
          };
          const isInsideReasoningNode = (node) => {
            let current = node;
            while (current && current !== root && current !== document.body) {
              if (current === node ? isReasoningNode(current) : hasReasoningStructuralHint(current)) return true;
              current = current.parentElement;
            }
            return false;
          };
          const isActiveIndicator = (node, terms) => {
            if (isInteractiveControl(node)) return false;
            const text = textOf(node);
            if (!hasAny(text, terms)) return false;
            const visibleText = clean(node.innerText || node.textContent || '');
            if (visibleText.length > 320) return false;
            if (node.matches && node.matches('[role="progressbar"],[aria-busy="true"]')) return true;
            const className = String(node.getAttribute && node.getAttribute('class') || '').toLowerCase();
            if (/loading|spinner|progress|pending|active|animate/.test(className)) return true;
            return hasAny(visibleText.toLowerCase(), terms);
          };
          const readText = (node) => {
            const clone = node.cloneNode(true);
            for (const item of Array.from(clone.querySelectorAll('*'))) {
              if (isReasoningNode(item)) item.remove();
            }
            for (const control of clone.querySelectorAll(controlSelector)) control.remove();
            const sandbox = document.createElement('div');
            sandbox.style.cssText = 'position:fixed;left:-100000px;top:0;width:1200px;white-space:normal;';
            sandbox.appendChild(clone);
            document.body.appendChild(sandbox);
            const value = clone.innerText || clone.textContent || '';
            sandbox.remove();
            return clean(value);
          };
          const candidates = [];
          for (const selector of selectors) {
            try {
              for (const node of root.querySelectorAll(selector)) {
                if (!node || !visible(node)) continue;
                if (isInsideReasoningNode(node)) continue;
                const text = readText(node);
                if (!text) continue;
                const rect = node.getBoundingClientRect();
                candidates.push({node, text, top: rect.top || 0, bottom: rect.bottom || 0});
              }
            } catch (_error) {}
          }
          const complete = candidates.filter((item) => !candidates.some((outer) =>
            outer !== item
            && outer.node !== item.node
            && outer.node.contains(item.node)
            && outer.text.length >= item.text.length
          ));
          complete.sort((a, b) => (a.bottom - b.bottom) || (a.top - b.top));
          const finalTexts = [];
          const seen = [];
          for (const item of complete) {
            const duplicate = seen.some((prev) =>
              prev.text === item.text
              && Math.abs(prev.top - item.top) < 3
              && Math.abs(prev.bottom - item.bottom) < 3
            );
            if (duplicate) continue;
            seen.push(item);
            finalTexts.push(item.text);
          }

          const reasoningTexts = [];
          let reasoningActive = false;
          let webSearchActive = false;
          let webSearchSeen = false;
          let generationActive = false;
          for (const node of root.querySelectorAll('div,section,article,p,span,[role="status"],[role="progressbar"]')) {
            if (!visible(node)) continue;
            if (isInteractiveControl(node)) continue;
            const text = textOf(node);
            if (!text) continue;
            if (hasAny(text, reasoningTerms)) {
              const visibleText = clean(node.innerText || node.textContent || '');
              if (visibleText && visibleText.length <= 6000) reasoningTexts.push(visibleText);
              if (isActiveIndicator(node, reasoningActiveTerms)) reasoningActive = true;
            }
            if (hasAny(text, searchTerms)) {
              webSearchSeen = true;
              if (isActiveIndicator(node, searchActiveTerms)) webSearchActive = true;
            }
          }
          for (const spinner of root.querySelectorAll('[aria-busy="true"],[role="progressbar"],[class*="loading" i],[class*="spinner" i]')) {
            if (!visible(spinner)) continue;
            if (isInteractiveControl(spinner)) continue;
            const text = textOf(spinner);
            if (hasAny(text, reasoningTerms)) reasoningActive = true;
            if (hasAny(text, searchTerms)) {
              webSearchSeen = true;
              webSearchActive = true;
            }
            const className = String(spinner.getAttribute && spinner.getAttribute('class') || '').toLowerCase();
            if (spinner.matches && spinner.matches('[aria-busy="true"],[role="progressbar"]')) generationActive = true;
            if (/loading|spinner|progress|pending|animate/.test(className)) generationActive = true;
          }
          for (const control of root.querySelectorAll('button,[role="button"],[aria-label],[title]')) {
            if (!visible(control)) continue;
            const text = labelTextOf(control);
            if (text && hasAny(text, generationStopTerms)) {
              generationActive = true;
              break;
            }
          }
          for (const selector of inputSelectors) {
            try {
              for (const input of document.querySelectorAll(selector)) {
                if (!input || !visible(input)) continue;
                const tag = String(input.tagName || '').toLowerCase();
                const disabled = Boolean(input.disabled || input.readOnly)
                  || String(input.getAttribute && input.getAttribute('aria-disabled') || '').toLowerCase() === 'true'
                  || Boolean(input.closest && input.closest('[aria-disabled="true"],[data-disabled="true"]'));
                const editable = tag === 'textarea' || tag === 'input' || input.isContentEditable
                  || String(input.getAttribute && input.getAttribute('role') || '').toLowerCase() === 'textbox';
                if (editable && disabled) generationActive = true;
              }
            } catch (_error) {}
          }
          return {
            final_texts: finalTexts,
            reasoning_text: clean(reasoningTexts.join('\\n\\n')),
            reasoning_active: reasoningActive,
            web_search_active: webSearchActive,
            web_search_seen: webSearchSeen,
            generation_active: generationActive,
          };
        }
        """
        try:
            value = self._page.evaluate(
                js,
                {
                    "selectors": self.ASSISTANT_MESSAGE_SELECTORS,
                    "input_selectors": self.INPUT_SELECTORS,
                },
            )
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    @classmethod
    def _looks_like_tool_call_answer(cls, text: str) -> bool:
        raw = text or ""
        return bool(
            re.search(
                r"(?is)(?:^|\n)\s*(?:```)?\s*(?:tool\s*[_\-. ]?\s*call|function\s*[_\-. ]?\s*call|tool\s*use|use\s*tool)\b",
                raw,
            )
        )

    @classmethod
    def _looks_like_complete_tool_call_answer(cls, text: str) -> bool:
        raw = text or ""
        if not cls._looks_like_tool_call_answer(raw):
            return False
        try:
            parsed = ToolCallParser().parse(raw, allow_bare_json=True)
        except Exception:
            return False
        if not parsed.calls:
            return False
        parsed_ranges = parsed.ranges or []
        for marker in TOOL_MARKER_RE.finditer(raw):
            start_after_noise = _skip_noise_after_marker(raw, marker.end())
            if find_balanced_value(raw, start_after_noise) is not None:
                continue
            if any(start <= marker.start() < end for start, end in parsed_ranges):
                continue
            return False
        return True

    @classmethod
    def _strip_reasoning_text(cls, text: str) -> str:
        answer = (text or "").strip()
        if not answer:
            return ""
        lower_head = answer[:2500].lower()
        reasoning_terms = ("reasoning", "thinking", "deepthink", "thought", "рассужд", "размышл", "думаю", "мысл")
        if not any(term in lower_head for term in reasoning_terms):
            return answer

        final_markers = (
            r"final answer\s*:",
            r"answer\s*:",
            r"итог\s*:",
            r"ответ\s*:",
            r"финальный ответ\s*:",
        )
        for marker in final_markers:
            matches = list(re.finditer(rf"(?im)(?:^|\n)\s*{marker}\s*", answer))
            if matches:
                return answer[matches[-1].end() :].strip()

        paragraphs = re.split(r"\n\s*\n", answer, maxsplit=1)
        if len(paragraphs) == 2 and any(term in paragraphs[0].lower() for term in reasoning_terms):
            if len(paragraphs[0]) <= 2500:
                return paragraphs[1].strip()
        return answer

    @classmethod
    def _select_assistant_answer_text(cls, messages: list[str], question: str, previous_answer: str) -> str:
        candidates: list[tuple[str, bool]] = []
        for item in messages:
            stripped = cls._strip_reasoning_text(str(item or ""))
            text = cls._sanitize_answer_text(stripped, question, previous_answer)
            if text:
                candidates.append((text, cls._looks_like_leaked_reasoning(stripped)))
        if not candidates:
            return ""

        if not any(leaked for _text, leaked in candidates):
            return cls._latest_assistant_message_text([text for text, _leaked in candidates])

        non_leaked = [text for text, leaked in candidates if not leaked]
        if non_leaked:
            return cls._latest_assistant_message_text(non_leaked)

        return candidates[-1][0]

    @staticmethod
    def _normalize_compare_text(value: str) -> str:
        return " ".join((value or "").strip().lower().split())

    @classmethod
    def _is_chat_length_limit_text(cls, text: str) -> bool:
        normalized = cls._normalize_compare_text(text)
        return "length limit reached" in normalized and "please start a new chat" in normalized

    def _page_has_chat_length_limit_notice(self) -> bool:
        if self._page is None:
            return False
        try:
            return bool(
                self._page.evaluate(
                    """
                    () => {
                      const visible = (el) => {
                        const style = window.getComputedStyle(el);
                        const rect = el.getBoundingClientRect();
                        return style.display !== 'none' && style.visibility !== 'hidden'
                          && rect.width > 0 && rect.height > 0;
                      };
                      for (const el of document.querySelectorAll('div,p,span,[role="alert"]')) {
                        if (!visible(el)) continue;
                        const text = (el.innerText || el.textContent || '')
                          .replace(/\\s+/g, ' ')
                          .trim()
                          .toLowerCase();
                        if (text.length > 220) continue;
                        if (text.includes('length limit reached')
                            && text.includes('please start a new chat')) {
                          return true;
                        }
                      }
                      return false;
                    }
                    """
                )
            )
        except Exception:
            return False

    @classmethod
    def _latest_assistant_message_text(cls, messages: list[str]) -> str:
        texts = [str(item or "").strip() for item in messages if str(item or "").strip()]
        if not texts:
            return ""

        latest = texts[-1]
        latest_normalized = cls._normalize_compare_text(latest)
        best = latest
        for item in reversed(texts[:-1]):
            normalized = cls._normalize_compare_text(item)
            if not latest_normalized or latest_normalized not in normalized:
                continue
            if len(item) > len(best):
                best = item
        return best

    @classmethod
    def _sanitize_answer_text(cls, text: str, question: str, previous_answer: str) -> str:
        answer = (text or "").strip()
        if not answer or cls._looks_like_ui_action_text(answer):
            return ""
        normalized = cls._normalize_compare_text(answer)
        if normalized in {
            cls._normalize_compare_text(question),
            cls._normalize_compare_text(previous_answer),
        }:
            return ""
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        if lines and cls._normalize_compare_text(lines[0]) == cls._normalize_compare_text(question):
            answer = "\n".join(lines[1:]).strip()
        extracted = cls._extract_final_from_leaked_reasoning(answer)
        if extracted:
            answer = extracted
        elif cls._looks_like_leaked_reasoning(answer):
            return ""
        return answer

    @classmethod
    def _looks_like_leaked_reasoning(cls, text: str) -> bool:
        normalized = cls._normalize_compare_text(text)
        if not normalized:
            return False
        strong_prefixes = (
            "we need to respond",
            "we need to answer",
            "we need to output",
            "we need to say",
            "need to respond",
            "need to answer",
            "the user said",
            "the user asks",
            "the user wants",
            "i need to respond",
            "i need to answer",
            "the assistant already",
            "нужно ответить",
            "надо ответить",
            "мы получили результат",
            "теперь нужно",
            "нужно подтвердить",
        )
        if normalized.startswith(strong_prefixes):
            return True

        markers = (
            "user instruction",
            "as per user instruction",
            "respond to the user",
            "answer with only",
            "just output",
            "just say",
            "something like",
            "command printed",
            "output exactly",
            "exact string",
            "no tool use",
            "no additional tool",
            "the assistant already",
            "tool call",
            "final answer",
            "as the final answer",
            "return final answer",
            "return the final answer",
            "return only final",
            "return only the final",
            "marker",
            "по инструкции",
            "инструменты не нужны",
            "обычным текстом",
            "получили результат",
            "нужно подтвердить",
            "ответ:",
        )
        return sum(1 for marker in markers if marker in normalized) >= 2

    @classmethod
    def _extract_final_from_leaked_reasoning(cls, text: str) -> str:
        answer = (text or "").strip()
        if not cls._looks_like_leaked_reasoning(answer):
            return ""

        marker_patterns = (
            r"(?im)(?:^|\n)\s*(?:final answer|answer|итог|ответ|финальный ответ)\s*:\s*([^\n]+)",
            r"(?is)(?:result|результат)\s*:\s*([^\"“”«»\n]+)",
        )
        for pattern in marker_patterns:
            matches = [match.group(1).strip() for match in re.finditer(pattern, answer)]
            if matches:
                cleaned = cls._clean_extracted_final(matches[-1])
                if cleaned:
                    return cleaned

        quote_matches = re.findall(r'"([^"\n]{1,300})"|“([^”\n]{1,300})”|«([^»\n]{1,300})»', answer)
        quotes = [next(part for part in match if part).strip() for match in quote_matches if any(match)]
        quotes = [quote for quote in quotes if quote and not cls._looks_like_leaked_reasoning(quote)]
        prompt_quote_prefixes = (
            "answer with",
            "respond with",
            "reply with",
            "ответь",
            "ответить",
            "вызови",
            "call ",
        )
        quotes = [
            quote
            for quote in quotes
            if not cls._normalize_compare_text(quote).startswith(prompt_quote_prefixes)
        ]
        if quotes:
            exact_quotes = [
                quote
                for quote in quotes
                if re.fullmatch(r"[\w .,:;!?+=/@#%()\\-]{1,120}", quote, flags=re.UNICODE)
            ]
            return cls._clean_extracted_final((exact_quotes or quotes)[-1])

        return ""

    @staticmethod
    def _clean_extracted_final(text: str) -> str:
        cleaned = (text or "").strip().strip("`'\"“”«»")
        cleaned = re.sub(r"\s+(?:or|или)\s+.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip().strip("`'\"“”«»")
        return cleaned if 0 < len(cleaned) <= 500 else ""

    @classmethod
    def _looks_like_ui_action_text(cls, text: str) -> bool:
        tokens = cls._normalize_compare_text(text).split()
        if not tokens or len(tokens) > 8:
            return False
        allowed = {
            "copy",
            "download",
            "regenerate",
            "like",
            "dislike",
            "share",
            "копировать",
            "скачать",
            "повтор",
            "нравится",
            "поделиться",
        }
        cleaned = ["".join(ch for ch in token if ch.isalnum()) for token in tokens]
        cleaned = [token for token in cleaned if token]
        return bool(cleaned) and all(token in allowed for token in cleaned)

    @classmethod
    def _merge_continued_answer(cls, previous: str, current: str) -> str:
        earlier = (previous or "").strip()
        latest = (current or "").strip()
        if not earlier:
            return latest
        if not latest:
            return earlier
        if latest.startswith(earlier) or earlier in latest:
            return latest
        if earlier.startswith(latest) or latest in earlier:
            return earlier
        for size in range(min(len(earlier), len(latest), 4096), 15, -1):
            if earlier[-size:] == latest[:size]:
                return earlier + latest[size:]
        return earlier.rstrip() + "\n\n" + latest.lstrip()

    def _click_continue_if_available(self) -> bool:
        if self._page is None:
            return False
        for selector in self.CONTINUE_BUTTON_SELECTORS:
            try:
                button = self._page.query_selector(selector)
                if button is not None and button.is_visible() and button.is_enabled():
                    button.click()
                    return True
            except Exception:
                continue
        try:
            return bool(
                self._page.evaluate(
                    """
                    () => {
                      const variants = new Set(['continue', 'продолжить']);
                      const visible = (el) => {
                        const style = window.getComputedStyle(el);
                        const rect = el.getBoundingClientRect();
                        return style.display !== 'none' && style.visibility !== 'hidden'
                          && rect.width > 0 && rect.height > 0;
                      };
                      for (const el of document.querySelectorAll('button,[role="button"]')) {
                        const text = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                        if (!variants.has(text) || !visible(el) || el.disabled) continue;
                        el.click();
                        return true;
                      }
                      return false;
                    }
                    """
                )
            )
        except Exception:
            return False

    def _get_assistant_messages_texts(self) -> list[str]:
        if self._page is None:
            return []
        js = """
        (selectors) => {
          const root = document.querySelector('main') || document.body;
          const clean = (value) => (value || '')
            .replace(/\\u00a0/g, ' ')
            .replace(/[ \\t]+\\n/g, '\\n')
            .replace(/\\n{3,}/g, '\\n\\n')
            .trim();
          const isVisible = (el) => {
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style.display !== 'none' && style.visibility !== 'hidden'
              && rect.width > 0 && rect.height > 0;
          };
          const badTexts = new Set(['новый чат', 'deepseek', 'копировать', 'скачать', 'copy', 'download']);
          const controlSelector = [
            'button', '[role="button"]', 'svg', 'canvas', 'input', 'textarea',
            'select', 'script', 'style', '[contenteditable="true"]',
            '[aria-pressed]', '[aria-selected]', '[aria-checked]',
            '[class*="toggle" i]', '[class*="ds-toggle-button" i]',
            '[aria-label*="copy" i]', '[aria-label*="download" i]',
            '[title*="copy" i]', '[title*="download" i]'
          ].join(',');
          const readText = (node) => {
            const clone = node.cloneNode(true);
            for (const control of clone.querySelectorAll(controlSelector)) control.remove();
            const sandbox = document.createElement('div');
            sandbox.style.cssText = 'position:fixed;left:-100000px;top:0;width:1200px;white-space:normal;';
            sandbox.appendChild(clone);
            document.body.appendChild(sandbox);
            const value = clone.innerText || clone.textContent || '';
            sandbox.remove();
            return clean(value);
          };
          const candidates = [];
          for (const selector of selectors) {
            for (const node of root.querySelectorAll(selector)) {
              if (!node || !isVisible(node)) continue;
              const text = readText(node);
              if (!text || badTexts.has(text.toLowerCase())) continue;
              const rect = node.getBoundingClientRect();
              candidates.push({node, text, top:rect.top || 0, bottom:rect.bottom || 0});
            }
          }
          const complete = candidates.filter((item) => !candidates.some((outer) =>
            outer !== item
            && outer.node !== item.node
            && outer.node.contains(item.node)
            && outer.text.length >= item.text.length
          ));
          complete.sort((a, b) => (a.bottom - b.bottom) || (a.top - b.top));
          const output = [];
          const seen = [];
          for (const item of complete) {
            const duplicate = seen.some((prev) =>
              prev.text === item.text
              && Math.abs(prev.top - item.top) < 3
              && Math.abs(prev.bottom - item.bottom) < 3
            );
            if (duplicate) continue;
            seen.push(item);
            output.push(item.text);
          }
          return output;
        }
        """
        try:
            raw = self._page.evaluate(js, self.ASSISTANT_MESSAGE_SELECTORS)
        except Exception:
            return []
        if not isinstance(raw, list):
            return []
        return [str(item or "").strip() for item in raw if str(item or "").strip()]

    def _get_user_messages_texts(self) -> list[str]:
        if self._page is None:
            return []
        js = """
        (selectors) => {
          const root = document.querySelector('main') || document.body;
          const clean = (value) => (value || '')
            .replace(/\\u00a0/g, ' ')
            .replace(/[ \\t]+\\n/g, '\\n')
            .replace(/\\n{3,}/g, '\\n\\n')
            .trim();
          const isVisible = (el) => {
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style.display !== 'none' && style.visibility !== 'hidden'
              && rect.width > 0 && rect.height > 0;
          };
          const controlSelector = [
            'button', '[role="button"]', 'svg', 'canvas', 'input', 'textarea',
            'select', 'script', 'style', '[contenteditable="true"]',
            '[aria-pressed]', '[aria-selected]', '[aria-checked]',
            '[class*="toggle" i]', '[class*="ds-toggle-button" i]',
            '[aria-label*="copy" i]', '[aria-label*="download" i]',
            '[title*="copy" i]', '[title*="download" i]'
          ].join(',');
          const readText = (node) => {
            const clone = node.cloneNode(true);
            for (const control of clone.querySelectorAll(controlSelector)) control.remove();
            const sandbox = document.createElement('div');
            sandbox.style.cssText = 'position:fixed;left:-100000px;top:0;width:1200px;white-space:normal;';
            sandbox.appendChild(clone);
            document.body.appendChild(sandbox);
            const value = clone.innerText || clone.textContent || '';
            sandbox.remove();
            return clean(value);
          };
          const candidateSelector = selectors.join(',');
          const candidates = [];
          for (const node of root.querySelectorAll(candidateSelector)) {
            if (!node || !isVisible(node)) continue;
            const text = readText(node);
            if (!text) continue;
            const hint = [
              node.getAttribute && node.getAttribute('data-testid'),
              node.getAttribute && node.getAttribute('data-role'),
              node.getAttribute && node.getAttribute('class'),
              node.getAttribute && node.getAttribute('aria-label'),
            ].join(' ').toLowerCase();
            const looksUser = /user|human|question|request|prompt|用户|человек|пользователь/.test(hint);
            const looksAssistant = /assistant|bot|model|answer|response|markdown|prose|ассист|ответ/.test(hint);
            if (looksAssistant && !looksUser) continue;
            const rect = node.getBoundingClientRect();
            candidates.push({node, text, top:rect.top || 0, bottom:rect.bottom || 0, userHint: looksUser});
          }
          const explicitUserCandidates = candidates.filter((item) => item.userHint);
          const complete = explicitUserCandidates.filter((item) => !explicitUserCandidates.some((outer) =>
            outer !== item
            && outer.node !== item.node
            && outer.node.contains(item.node)
            && outer.text.length >= item.text.length
          ));
          complete.sort((a, b) => (a.bottom - b.bottom) || (a.top - b.top));
          const output = [];
          const seen = [];
          for (const item of complete) {
            const duplicate = seen.some((prev) =>
              prev.text === item.text
              && Math.abs(prev.top - item.top) < 3
              && Math.abs(prev.bottom - item.bottom) < 3
            );
            if (duplicate) continue;
            seen.push(item);
            output.push(item.text);
          }
          return output;
        }
        """
        try:
            raw = self._page.evaluate(js, self.USER_MESSAGE_SELECTORS)
        except Exception:
            return []
        if not isinstance(raw, list):
            return []
        return [str(item or "").strip() for item in raw if str(item or "").strip()]


@dataclass
class WorkerCommand:
    kind: str
    payload: dict[str, Any]
    response: "queue.Queue[Any]"
    created_at: float = field(default_factory=time.monotonic)
    cancelled: threading.Event = field(default_factory=threading.Event)


class BrowserWorker:
    def __init__(
        self,
        headless: bool = False,
        user_data_dir: str = DEFAULT_USER_DATA_DIR,
        answer_stable_sec: float = DEFAULT_ANSWER_STABLE_SEC,
        reasoning_mode: str = "off",
    ):
        self.headless = headless
        self.user_data_dir = user_data_dir
        self.answer_stable_sec = clamp_answer_stable_sec(answer_stable_sec)
        self.reasoning_mode = normalize_reasoning_mode(reasoning_mode)
        self._commands: "queue.Queue[WorkerCommand]" = queue.Queue()
        self._request_gate = threading.Lock()
        self._thread = threading.Thread(target=self._thread_main, name="DBillBrowser", daemon=True)
        self._started_event = threading.Event()
        self._status_lock = threading.Lock()
        self._ready = False
        self._startup_error: Optional[Exception] = None
        self._client: Optional[DeepSeekWebClient] = None
        self._active_command: Optional[WorkerCommand] = None
        self._active_kind: Optional[str] = None
        self._active_since: Optional[float] = None
        self._recovering_after_timeout = False
        self._last_timeout_error: Optional[str] = None
        self._stopping = False
        self._last_continue_clicks = 0
        self._total_continue_clicks = 0
        self._last_use_reasoning = False
        self._last_reasoning_chars = 0
        self._last_reasoning_active = False
        self._last_reasoning_wait_sec = 0.0
        self._last_web_search_seen = False
        self._last_web_search_active = False
        self._last_web_search_wait_sec = 0.0
        self._last_reasoning_toggle_found = False
        self._last_debug_artifact_dir = ""
        self._last_debug_artifact_error = ""
        self._last_request_journal: dict[str, Any] = {}
        self._request_journal_history: Deque[dict[str, Any]] = deque(maxlen=REQUEST_JOURNAL_LIMIT)
        self.watchdog_hang_limit = WATCHDOG_HANG_LIMIT
        self._consecutive_hangs = 0
        self._watchdog_restarts = 0
        self._watchdog_restart_pending = False
        self._last_watchdog_reason = ""

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def set_answer_stable_sec(self, value: Any) -> float:
        stable = clamp_answer_stable_sec(value)
        with self._status_lock:
            self.answer_stable_sec = stable
        return stable

    def set_reasoning_mode(self, value: Any) -> str:
        mode = normalize_reasoning_mode(value)
        with self._status_lock:
            self.reasoning_mode = mode
        return mode

    def get_reasoning_mode(self) -> str:
        with self._status_lock:
            return self.reasoning_mode

    def diagnostics(self) -> dict[str, Any]:
        with self._status_lock:
            active_for = time.monotonic() - self._active_since if self._active_since is not None else 0.0
            return {
                "answer_stable_sec": self.answer_stable_sec,
                "reasoning_mode": self.reasoning_mode,
                "last_continue_clicks": self._last_continue_clicks,
                "total_continue_clicks": self._total_continue_clicks,
                "last_use_reasoning": self._last_use_reasoning,
                "last_reasoning_chars": self._last_reasoning_chars,
                "last_reasoning_active": self._last_reasoning_active,
                "last_reasoning_wait_sec": round(self._last_reasoning_wait_sec, 1),
                "last_web_search_seen": self._last_web_search_seen,
                "last_web_search_active": self._last_web_search_active,
                "last_web_search_wait_sec": round(self._last_web_search_wait_sec, 1),
                "last_reasoning_toggle_found": self._last_reasoning_toggle_found,
                "last_debug_artifact_dir": self._last_debug_artifact_dir,
                "last_debug_artifact_error": self._last_debug_artifact_error,
                "last_request_journal": self._last_request_journal,
                "request_journal_count": len(self._request_journal_history),
                "request_journal_tail": list(self._request_journal_history)[-5:],
                "consecutive_hangs": self._consecutive_hangs,
                "watchdog_hang_limit": self.watchdog_hang_limit,
                "watchdog_restarts": self._watchdog_restarts,
                "watchdog_restart_pending": self._watchdog_restart_pending,
                "last_watchdog_reason": self._last_watchdog_reason,
                "active_command": self._active_kind,
                "active_for_sec": round(active_for, 1),
                "queued_commands": self._commands.qsize(),
                "recovering_after_timeout": self._recovering_after_timeout,
                "last_timeout_error": self._last_timeout_error,
            }

    def _record_client_diagnostics(self, client: DeepSeekWebClient) -> None:
        with self._status_lock:
            self._last_continue_clicks = int(getattr(client, "last_continue_clicks", 0) or 0)
            self._total_continue_clicks = int(getattr(client, "total_continue_clicks", 0) or 0)
            self._last_use_reasoning = bool(getattr(client, "last_use_reasoning", False))
            self._last_reasoning_chars = int(getattr(client, "last_reasoning_chars", 0) or 0)
            self._last_reasoning_active = bool(getattr(client, "last_reasoning_active", False))
            self._last_reasoning_wait_sec = float(getattr(client, "last_reasoning_wait_sec", 0.0) or 0.0)
            self._last_web_search_seen = bool(getattr(client, "last_web_search_seen", False))
            self._last_web_search_active = bool(getattr(client, "last_web_search_active", False))
            self._last_web_search_wait_sec = float(getattr(client, "last_web_search_wait_sec", 0.0) or 0.0)
            self._last_reasoning_toggle_found = bool(getattr(client, "last_reasoning_toggle_found", False))
            self._last_debug_artifact_dir = str(getattr(client, "last_debug_artifact_dir", "") or "")
            self._last_debug_artifact_error = str(getattr(client, "last_debug_artifact_error", "") or "")
            journal = getattr(client, "last_request_journal", {}) or {}
            if isinstance(journal, dict) and journal:
                self._last_request_journal = journal
                if not self._request_journal_history or self._request_journal_history[-1] != journal:
                    self._request_journal_history.append(journal)

    def _set_client(self, client: Optional[DeepSeekWebClient]) -> None:
        with self._status_lock:
            self._client = client

    def _mark_command_started(self, command: WorkerCommand) -> None:
        with self._status_lock:
            self._active_command = command
            self._active_kind = command.kind
            self._active_since = time.monotonic()

    def _mark_command_finished(self, command: WorkerCommand) -> None:
        with self._status_lock:
            if self._active_command is command:
                self._active_command = None
                self._active_kind = None
                self._active_since = None

    def _mark_command_timed_out(self, command: WorkerCommand, timeout: float) -> None:
        command.cancelled.set()
        with self._status_lock:
            self._recovering_after_timeout = True
            self._ready = False
            self._last_timeout_error = f"Browser command timed out: {command.kind} after {timeout:g}s"
            self._note_hang_locked(self._last_timeout_error)

    def _note_hang_locked(self, reason: str) -> None:
        self._consecutive_hangs += 1
        self._last_watchdog_reason = reason
        if self._consecutive_hangs >= self.watchdog_hang_limit:
            self._watchdog_restart_pending = True

    def _note_hang(self, reason: str) -> bool:
        with self._status_lock:
            self._note_hang_locked(reason)
            return self._watchdog_restart_pending

    def _note_successful_command(self) -> None:
        with self._status_lock:
            self._consecutive_hangs = 0
            if not self._recovering_after_timeout:
                self._watchdog_restart_pending = False

    def _watchdog_restart_completed(self, reason: str) -> None:
        with self._status_lock:
            if self._watchdog_restart_pending:
                self._watchdog_restarts += 1
                self._watchdog_restart_pending = False
                self._consecutive_hangs = 0
                self._last_watchdog_reason = reason

    def _mark_recovered(self) -> None:
        with self._status_lock:
            self._recovering_after_timeout = False
            self._last_timeout_error = None
            self._ready = True
            self._startup_error = None

    def _recovery_error(self) -> Optional[RuntimeError]:
        with self._status_lock:
            if not self._recovering_after_timeout:
                return None
            detail = self._last_timeout_error or "previous browser command timed out"
        return RuntimeError(f"Browser worker is recovering after timeout: {detail}")

    def _busy_error(self, requested_kind: str) -> RuntimeError:
        with self._status_lock:
            active_kind = self._active_kind or "unknown"
            active_for = time.monotonic() - self._active_since if self._active_since is not None else 0.0
        return RuntimeError(
            f"Browser is busy with {active_kind} for {active_for:.1f}s; cannot run {requested_kind}"
        )

    def _interrupt_active_client(self, reason: str, join_timeout: float = 5.0) -> None:
        with self._status_lock:
            client = self._client
            command = self._active_command
        if command is not None:
            command.cancelled.set()
        if client is None:
            return
        cancel = getattr(client, "cancel_active_request", None)
        if callable(cancel):
            try:
                cancel()
                logging.info("DeepSeek browser cancellation requested during %s.", reason)
            except Exception as exc:
                logging.warning("DeepSeek browser cancellation during %s failed: %s", reason, exc)

    def _restart_client(self, client: DeepSeekWebClient, reason: str) -> DeepSeekWebClient:
        logging.warning("Restarting DeepSeek browser after %s.", reason)
        try:
            client.close()
        except Exception:
            pass
        replacement = DeepSeekWebClient(
            headless=self.headless,
            user_data_dir=self.user_data_dir,
            answer_stable_sec=self.answer_stable_sec,
        )
        replacement.start()
        self._set_client(replacement)
        self._watchdog_restart_completed(reason)
        self._mark_recovered()
        return replacement

    @staticmethod
    def _should_restart_client(exc: Exception) -> bool:
        text = str(exc).lower()
        recoverable = (
            "target page, context or browser has been closed",
            "target closed",
            "page closed",
            "browser has been closed",
            "execution context was destroyed",
        )
        return any(fragment in text for fragment in recoverable)

    @staticmethod
    def _is_stop_close_error(exc: Exception) -> bool:
        text = str(exc).lower()
        expected = (
            "target page, context or browser has been closed",
            "target closed",
            "browser has been closed",
            "connection closed",
            "connection closed while reading from the driver",
        )
        return any(fragment in text for fragment in expected)

    def _with_client_recovery(self, client: DeepSeekWebClient, action_name: str, action, command: Optional[WorkerCommand] = None):
        try:
            return client, action(client)
        except Exception as exc:
            if not self._should_restart_client(exc):
                raise
            if command is not None and command.cancelled.is_set():
                raise
            logging.warning("Restarting DeepSeek browser after %s failed: %s", action_name, exc)
            try:
                client.close()
            except Exception:
                pass
            replacement = DeepSeekWebClient(
                headless=self.headless,
                user_data_dir=self.user_data_dir,
                answer_stable_sec=self.answer_stable_sec,
            )
            replacement.start()
            self._set_client(replacement)
            self._watchdog_restart_completed(f"{action_name}: {exc}")
            self._mark_recovered()
            return replacement, action(replacement)

    @staticmethod
    def _send_text_prompt_with_options(
        client: DeepSeekWebClient,
        prompt: str,
        timeout_sec: int,
        use_reasoning: bool,
        request_id: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        try:
            return client.send_text_prompt(
                prompt,
                timeout_sec=timeout_sec,
                use_reasoning=use_reasoning,
                request_id=request_id,
                cancel_event=cancel_event,
            )
        except TypeError as exc:
            if "cancel_event" not in str(exc):
                if "request_id" not in str(exc) and "use_reasoning" not in str(exc):
                    raise
            else:
                try:
                    return client.send_text_prompt(
                        prompt,
                        timeout_sec=timeout_sec,
                        use_reasoning=use_reasoning,
                        request_id=request_id,
                    )
                except TypeError as retry_exc:
                    if "request_id" not in str(retry_exc) and "use_reasoning" not in str(retry_exc):
                        raise
                    exc = retry_exc
            if "request_id" not in str(exc) and "use_reasoning" not in str(exc):
                raise
        try:
            return client.send_text_prompt(prompt, timeout_sec=timeout_sec, use_reasoning=use_reasoning)
        except TypeError as exc:
            if "use_reasoning" not in str(exc):
                raise
            return client.send_text_prompt(prompt, timeout_sec=timeout_sec)

    def _thread_main(self) -> None:
        client = DeepSeekWebClient(
            headless=self.headless,
            user_data_dir=self.user_data_dir,
            answer_stable_sec=self.answer_stable_sec,
        )
        try:
            client.start()
            self._set_client(client)
            with self._status_lock:
                self._ready = True
                self._startup_error = None
        except Exception as exc:
            with self._status_lock:
                self._ready = False
                self._startup_error = exc
            self._started_event.set()
            return
        finally:
            self._started_event.set()

        while True:
            command = self._commands.get()
            if command.cancelled.is_set():
                command.response.put(TimeoutError(f"Browser command was cancelled before start: {command.kind}"))
                if not self._stopping:
                    self._mark_recovered()
                continue
            self._mark_command_started(command)
            try:
                if command.kind == "stop":
                    client.close()
                    self._set_client(None)
                    command.response.put(None)
                    return
                if command.kind == "ask_text":
                    client.set_answer_stable_sec(command.payload.get("answer_stable_sec"))
                    client, answer = self._with_client_recovery(
                        client,
                        "ask_text",
                        lambda active: self._send_text_prompt_with_options(
                            active,
                            str(command.payload.get("prompt") or ""),
                            int(command.payload.get("timeout") or DEFAULT_TIMEOUT_SEC),
                            bool(command.payload.get("use_reasoning")),
                            request_id=str(command.payload.get("request_id") or ""),
                            cancel_event=command.cancelled,
                        ),
                        command=command,
                    )
                    if command.cancelled.is_set():
                        client = self._restart_client(client, f"late cancelled {command.kind}")
                        command.response.put(TimeoutError(f"Browser command timed out and was cancelled: {command.kind}"))
                        continue
                    self._record_client_diagnostics(client)
                    self._note_successful_command()
                    command.response.put(answer)
                    continue
                if command.kind == "new_chat":
                    client, _unused = self._with_client_recovery(
                        client,
                        "new_chat",
                        lambda active: active.start_new_chat(),
                        command=command,
                    )
                    if command.cancelled.is_set():
                        client = self._restart_client(client, f"late cancelled {command.kind}")
                        command.response.put(TimeoutError(f"Browser command timed out and was cancelled: {command.kind}"))
                        continue
                    self._note_successful_command()
                    command.response.put(None)
                    continue
                if command.kind == "open_browser":
                    client, _unused = self._with_client_recovery(
                        client,
                        "open_browser",
                        lambda active: active.open_chat_page(),
                        command=command,
                    )
                    if command.cancelled.is_set():
                        client = self._restart_client(client, f"late cancelled {command.kind}")
                        command.response.put(TimeoutError(f"Browser command timed out and was cancelled: {command.kind}"))
                        continue
                    self._note_successful_command()
                    command.response.put(None)
                    continue
                command.response.put(RuntimeError(f"Unknown browser command: {command.kind}"))
            except Exception as exc:
                if command.kind == "ask_text":
                    self._record_client_diagnostics(client)
                    if isinstance(exc, TimeoutError) and not command.cancelled.is_set():
                        if self._note_hang(f"Browser answer timed out: {exc}"):
                            try:
                                client = self._restart_client(client, "watchdog repeated answer timeouts")
                            except Exception as restart_exc:
                                with self._status_lock:
                                    self._ready = False
                                    self._startup_error = restart_exc
                                command.response.put(restart_exc)
                                continue
                if command.cancelled.is_set() and not self._stopping:
                    try:
                        client = self._restart_client(client, f"cancelled {command.kind}")
                    except Exception as restart_exc:
                        with self._status_lock:
                            self._ready = False
                            self._startup_error = restart_exc
                        command.response.put(restart_exc)
                        continue
                    command.response.put(TimeoutError(f"Browser command timed out and was cancelled: {command.kind}"))
                    continue
                if self._stopping:
                    try:
                        client.close()
                    except Exception:
                        pass
                    self._set_client(None)
                    command.response.put(exc)
                    return
                command.response.put(exc)
            finally:
                self._mark_command_finished(command)

    def status(self) -> tuple[bool, bool, Optional[str]]:
        with self._status_lock:
            error = str(self._startup_error) if self._startup_error else None
            return self._started_event.is_set(), self._ready, error

    def _request(self, kind: str, payload: dict[str, Any], timeout: int) -> Any:
        if not self._started_event.wait(timeout=120):
            raise TimeoutError("Browser did not initialize within 120 seconds")
        with self._status_lock:
            if self._startup_error is not None:
                raise RuntimeError(f"Browser startup failed: {self._startup_error}")
        if kind != "stop":
            recovery_error = self._recovery_error()
            if recovery_error is not None:
                raise recovery_error
            if not self._request_gate.acquire(blocking=False):
                raise self._busy_error(kind)
            gate_acquired = True
        else:
            gate_acquired = False
        response: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        command = WorkerCommand(kind=kind, payload=payload, response=response)
        try:
            self._commands.put(command)
            try:
                result = response.get(timeout=timeout)
            except queue.Empty as exc:
                self._mark_command_timed_out(command, timeout)
                self._interrupt_active_client(f"{kind}_timeout")
                raise TimeoutError(f"Browser command timed out: {kind}") from exc
            if isinstance(result, Exception):
                raise result
            return result
        finally:
            if gate_acquired:
                self._request_gate.release()

    def ask_text(
        self,
        prompt: str,
        timeout: int = DEFAULT_TIMEOUT_SEC,
        use_reasoning: Optional[bool] = None,
        request_id: Optional[str] = None,
    ) -> str:
        with self._status_lock:
            answer_stable_sec = self.answer_stable_sec
            mode = self.reasoning_mode
        if use_reasoning is None:
            use_reasoning = mode == "on"
        result = self._request(
            "ask_text",
            {
                "prompt": prompt,
                "timeout": timeout,
                "answer_stable_sec": answer_stable_sec,
                "use_reasoning": bool(use_reasoning),
                "reasoning_mode": mode,
                "request_id": request_id or "",
            },
            timeout + WORKER_RESPONSE_GRACE_SEC,
        )
        return str(result or "").strip()

    def ask_text_stream(
        self,
        prompt: str,
        timeout: int = DEFAULT_TIMEOUT_SEC,
        use_reasoning: Optional[bool] = None,
        request_id: Optional[str] = None,
    ):
        answer = self.ask_text(prompt, timeout=timeout, use_reasoning=use_reasoning, request_id=request_id)
        for start in range(0, len(answer), 180):
            yield answer[start : start + 180]

    def new_chat(self, timeout: int = 40) -> None:
        self._request("new_chat", {}, timeout)

    def open_browser(self, timeout: int = 60) -> None:
        self._request("open_browser", {}, timeout)

    def stop(self) -> None:
        if not self._thread.is_alive():
            return
        with self._status_lock:
            self._stopping = True
            active = self._active_command is not None
        if active:
            self._interrupt_active_client("stop", join_timeout=3)
            self._commands.put(WorkerCommand(kind="stop", payload={}, response=queue.Queue(maxsize=1)))
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logging.error("Browser stop did not finish while an active command was running.")
            return
        try:
            self._request("stop", {}, 8)
        except Exception as exc:
            if self._is_stop_close_error(exc):
                logging.info("Browser stop noticed an already closed browser connection: %s", exc)
                self._set_client(None)
            else:
                logging.error("Browser stop failed: %s", exc)
                self._interrupt_active_client("stop_after_error", join_timeout=3)
        self._thread.join(timeout=5)

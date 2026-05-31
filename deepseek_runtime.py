#!/usr/bin/env python3
"""Text-only DeepSeek browser runtime used by the DBill OpenAI adapter GUI."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from playwright.sync_api import BrowserContext, Page, sync_playwright


DEEPSEEK_URL = "https://chat.deepseek.com/"
DEFAULT_TIMEOUT_SEC = 360
DEFAULT_ANSWER_STABLE_SEC = 2.5
MIN_ANSWER_STABLE_SEC = 0.5
WORKER_RESPONSE_GRACE_SEC = 60
DEFAULT_USER_DATA_DIR = "./deepseek_profile"


class DeepSeekChatLengthLimitReached(RuntimeError):
    """Raised when DeepSeek asks the user to continue in a fresh web chat."""


def clamp_answer_stable_sec(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_ANSWER_STABLE_SEC
    return max(MIN_ANSWER_STABLE_SEC, min(parsed, 30.0))


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
    NEW_CHAT_SELECTORS = [
        "a:has-text('New Chat')",
        "button:has-text('New Chat')",
        "a:has-text('Новый чат')",
        "button:has-text('Новый чат')",
        "button[aria-label*='New Chat']",
        "button[aria-label*='Новый чат']",
    ]
    CONTINUE_BUTTON_SELECTORS = [
        "button:has-text('Continue')",
        "button:has-text('Продолжить')",
        "[role='button']:has-text('Continue')",
        "[role='button']:has-text('Продолжить')",
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
        self._request_lock = threading.Lock()
        self._playwright = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    def set_answer_stable_sec(self, value: Any) -> float:
        self.answer_stable_sec = clamp_answer_stable_sec(value)
        return self.answer_stable_sec

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

    def send_text_prompt(self, prompt: str, timeout_sec: int = DEFAULT_TIMEOUT_SEC) -> str:
        message = (prompt or "").strip()
        if not message:
            raise ValueError("Empty text prompt")

        with self._request_lock:
            for attempt in range(2):
                if self._page_has_chat_length_limit_notice():
                    logging.info("DeepSeek chat length limit was already visible; opening a fresh chat.")
                    self._start_new_chat_locked()

                input_handle = self._wait_for_input_ready(timeout=60)
                previous_messages = self._get_assistant_messages_texts()
                self._focus_and_fill_input(input_handle, message)
                self._send_current_message(input_handle, previous_messages, message)
                try:
                    answer = self._wait_for_new_answer(previous_messages, message, timeout_sec)
                except DeepSeekChatLengthLimitReached:
                    if attempt:
                        raise
                    logging.info("DeepSeek chat reached the length limit; retrying in a fresh chat.")
                    self._start_new_chat_locked()
                    continue
                if not self._is_chat_length_limit_text(answer):
                    return answer
                if attempt:
                    raise DeepSeekChatLengthLimitReached("DeepSeek chat length limit reached after retry")
                logging.info("DeepSeek returned its chat length limit text; retrying in a fresh chat.")
                self._start_new_chat_locked()
            raise DeepSeekChatLengthLimitReached("DeepSeek chat length limit reached")

    def start_new_chat(self) -> None:
        if self._page is None:
            raise RuntimeError("DeepSeek page is not initialized")
        with self._request_lock:
            self._start_new_chat_locked()

    def _start_new_chat_locked(self) -> None:
        if self._page is None:
            raise RuntimeError("DeepSeek page is not initialized")
        for selector in self.NEW_CHAT_SELECTORS:
            try:
                button = self._page.query_selector(selector)
                if button is None or not button.is_visible():
                    continue
                button.click()
                time.sleep(0.8)
                self._wait_for_input_ready(timeout=20)
                return
            except Exception:
                continue
        clicked = self._page.evaluate(
            """
            () => {
              const variants = ['new chat', 'новый чат'];
              for (const item of document.querySelectorAll('a,button,div,[role="button"]')) {
                const text = (item.innerText || item.textContent || '').trim().toLowerCase();
                if (text && variants.some((sample) => text.includes(sample))) {
                  item.click();
                  return true;
                }
              }
              return false;
            }
            """
        )
        if clicked:
            time.sleep(0.8)
            self._wait_for_input_ready(timeout=20)
            return
        self._page.goto(DEEPSEEK_URL, wait_until="domcontentloaded", timeout=60000)
        time.sleep(1.0)
        self._wait_for_input_ready(timeout=30)

    def _wait_for_input_ready(self, timeout: int = 60):
        if self._page is None:
            raise RuntimeError("DeepSeek page is not initialized")
        deadline = time.monotonic() + timeout
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
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

    def _send_current_message(self, input_handle, previous_messages: Optional[list[str]] = None, question: str = "") -> None:
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
                self._ensure_message_submitted(input_handle, previous_messages, question)
                return
        input_handle.press("Enter")
        time.sleep(0.5)
        self._ensure_message_submitted(input_handle, previous_messages, question)

    def _wait_for_send_enabled(self, timeout: int = 5) -> bool:
        if self._page is None:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
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
        timeout: int = 5,
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._submission_observed(input_handle, previous_messages or [], question):
                return
            time.sleep(0.25)
        for key in ("Enter", "Control+Enter"):
            try:
                input_handle.press(key)
            except Exception:
                pass
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if self._submission_observed(input_handle, previous_messages or [], question):
                    return
                time.sleep(0.25)
        raise RuntimeError("DeepSeek did not accept the message")

    def _submission_observed(self, input_handle, previous_messages: list[str], question: str) -> bool:
        if not self._get_input_text(input_handle):
            return True
        messages = self._get_assistant_messages_texts()
        if not messages:
            return False
        previous_count = len(previous_messages)
        previous_answer = self._latest_assistant_message_text(previous_messages)
        raw_text = self._latest_assistant_message_text(messages)
        previous = "" if len(messages) > previous_count else previous_answer
        text = self._sanitize_answer_text(raw_text, question, previous)
        if text:
            logging.info("DeepSeek submission confirmed by a new assistant answer while composer was still populated.")
            return True
        return False

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
        previous_answer = self._latest_assistant_message_text(previous_messages)
        previous_count = len(previous_messages)
        candidate = ""
        last_seen = ""
        last_change_ts = time.monotonic()
        continuation_prefix = ""
        continue_clicks = 0
        self.last_continue_clicks = 0

        while time.monotonic() < deadline:
            if self._page_has_chat_length_limit_notice():
                raise DeepSeekChatLengthLimitReached("DeepSeek chat length limit notice is visible")
            messages = self._get_assistant_messages_texts()
            raw_text = self._latest_assistant_message_text(messages)
            previous = "" if len(messages) > previous_count else previous_answer
            text = self._sanitize_answer_text(raw_text, question, previous)
            if self._is_chat_length_limit_text(text):
                raise DeepSeekChatLengthLimitReached("DeepSeek chat length limit response was returned")
            if continuation_prefix and text:
                text = self._merge_continued_answer(continuation_prefix, text)
            if text:
                last_seen = text
                if text != candidate:
                    candidate = text
                    last_change_ts = time.monotonic()
                if time.monotonic() - last_change_ts >= self.answer_stable_sec:
                    if self._click_continue_if_available():
                        continue_clicks += 1
                        self.last_continue_clicks = continue_clicks
                        self.total_continue_clicks += 1
                        continuation_prefix = candidate
                        last_change_ts = time.monotonic()
                        logging.info(
                            "DeepSeek continuation accepted: click=%s answer_chars=%s",
                            continue_clicks,
                            len(candidate),
                        )
                        time.sleep(0.5)
                        continue
                    return text
            time.sleep(0.5)
        raise TimeoutError(f"DeepSeek answer timed out. Last fragment: {last_seen[:250]!r}")

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
        return answer

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


@dataclass
class WorkerCommand:
    kind: str
    payload: dict[str, Any]
    response: "queue.Queue[Any]"


class BrowserWorker:
    def __init__(
        self,
        headless: bool = False,
        user_data_dir: str = DEFAULT_USER_DATA_DIR,
        answer_stable_sec: float = DEFAULT_ANSWER_STABLE_SEC,
    ):
        self.headless = headless
        self.user_data_dir = user_data_dir
        self.answer_stable_sec = clamp_answer_stable_sec(answer_stable_sec)
        self._commands: "queue.Queue[WorkerCommand]" = queue.Queue()
        self._thread = threading.Thread(target=self._thread_main, name="DBillBrowser", daemon=True)
        self._started_event = threading.Event()
        self._status_lock = threading.Lock()
        self._ready = False
        self._startup_error: Optional[Exception] = None
        self._last_continue_clicks = 0
        self._total_continue_clicks = 0

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def set_answer_stable_sec(self, value: Any) -> float:
        stable = clamp_answer_stable_sec(value)
        with self._status_lock:
            self.answer_stable_sec = stable
        return stable

    def diagnostics(self) -> dict[str, Any]:
        with self._status_lock:
            return {
                "answer_stable_sec": self.answer_stable_sec,
                "last_continue_clicks": self._last_continue_clicks,
                "total_continue_clicks": self._total_continue_clicks,
            }

    def _record_continuations(self, client: DeepSeekWebClient) -> None:
        with self._status_lock:
            self._last_continue_clicks = int(getattr(client, "last_continue_clicks", 0) or 0)
            self._total_continue_clicks = int(getattr(client, "total_continue_clicks", 0) or 0)

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

    def _with_client_recovery(self, client: DeepSeekWebClient, action_name: str, action):
        try:
            return client, action(client)
        except Exception as exc:
            if not self._should_restart_client(exc):
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
            with self._status_lock:
                self._ready = True
                self._startup_error = None
            return replacement, action(replacement)

    def _thread_main(self) -> None:
        client = DeepSeekWebClient(
            headless=self.headless,
            user_data_dir=self.user_data_dir,
            answer_stable_sec=self.answer_stable_sec,
        )
        try:
            client.start()
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
            try:
                if command.kind == "stop":
                    client.close()
                    command.response.put(None)
                    return
                if command.kind == "ask_text":
                    client.set_answer_stable_sec(command.payload.get("answer_stable_sec"))
                    client, answer = self._with_client_recovery(
                        client,
                        "ask_text",
                        lambda active: active.send_text_prompt(
                            str(command.payload.get("prompt") or ""),
                            timeout_sec=int(command.payload.get("timeout") or DEFAULT_TIMEOUT_SEC),
                        ),
                    )
                    self._record_continuations(client)
                    command.response.put(answer)
                    continue
                if command.kind == "new_chat":
                    client, _unused = self._with_client_recovery(
                        client,
                        "new_chat",
                        lambda active: active.start_new_chat(),
                    )
                    command.response.put(None)
                    continue
                if command.kind == "open_browser":
                    client, _unused = self._with_client_recovery(
                        client,
                        "open_browser",
                        lambda active: active.open_chat_page(),
                    )
                    command.response.put(None)
                    continue
                command.response.put(RuntimeError(f"Unknown browser command: {command.kind}"))
            except Exception as exc:
                command.response.put(exc)

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
        response: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        self._commands.put(WorkerCommand(kind=kind, payload=payload, response=response))
        try:
            result = response.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"Browser command timed out: {kind}") from exc
        if isinstance(result, Exception):
            raise result
        return result

    def ask_text(self, prompt: str, timeout: int = DEFAULT_TIMEOUT_SEC) -> str:
        with self._status_lock:
            answer_stable_sec = self.answer_stable_sec
        result = self._request(
            "ask_text",
            {"prompt": prompt, "timeout": timeout, "answer_stable_sec": answer_stable_sec},
            timeout + WORKER_RESPONSE_GRACE_SEC,
        )
        return str(result or "").strip()

    def ask_text_stream(self, prompt: str, timeout: int = DEFAULT_TIMEOUT_SEC):
        answer = self.ask_text(prompt, timeout=timeout)
        for start in range(0, len(answer), 180):
            yield answer[start : start + 180]

    def new_chat(self, timeout: int = 40) -> None:
        self._request("new_chat", {}, timeout)

    def open_browser(self, timeout: int = 60) -> None:
        self._request("open_browser", {}, timeout)

    def stop(self) -> None:
        if not self._thread.is_alive():
            return
        try:
            self._request("stop", {}, 20)
        except Exception as exc:
            logging.error("Browser stop failed: %s", exc)
        self._thread.join(timeout=5)

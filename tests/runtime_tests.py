#!/usr/bin/env python3
"""Focused checks for chat-only DeepSeek continuation assembly."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from _path import PROJECT_ROOT  # noqa: F401
import deepseek_runtime
from deepseek_runtime import DeepSeekWebClient


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class ContinueClient(DeepSeekWebClient):
    def __init__(self) -> None:
        super().__init__()
        self.continued = False
        self.continue_clicks = 0

    def _get_assistant_messages_texts(self) -> list[str]:
        return ["part two"] if self.continued else ["part one"]

    def _click_continue_if_available(self) -> bool:
        if self.continued:
            return False
        self.continued = True
        self.continue_clicks += 1
        return True


class SlowChunkClient(DeepSeekWebClient):
    def __init__(self, clock: FakeClock) -> None:
        super().__init__()
        self.clock = clock

    def _get_assistant_messages_texts(self) -> list[str]:
        if self.clock.now < 102.0:
            return ["part one"]
        return ["part one part two"]

    def _click_continue_if_available(self) -> bool:
        return False


class StickyComposerClient(DeepSeekWebClient):
    def _get_input_text(self, _input_handle) -> str:
        return "prompt still rendered in composer"

    def _get_assistant_messages_texts(self) -> list[str]:
        return ["accepted answer"]


class SubmittedUserMessageClient(StickyComposerClient):
    def _get_user_messages_texts(self) -> list[str]:
        return ["question"]


class BusyBeforeSendClient(DeepSeekWebClient):
    def __init__(self) -> None:
        super().__init__()
        self._page = object()
        self.index = 0

    def _get_answer_dom_snapshot(self):
        self.index += 1
        return {"generation_active": self.index < 3, "reasoning_active": False, "web_search_active": False}


class FragmentedAnswerClient(DeepSeekWebClient):
    full_answer = (
        "FULL-ANSWER\n\n"
        "FIRST-PARA has the opening details.\n\n"
        "- ITEM-ONE\n"
        "- ITEM-TWO\n\n"
        "LAST-PARA has the final details."
    )

    def _get_assistant_messages_texts(self) -> list[str]:
        return [
            self.full_answer,
            "FIRST-PARA has the opening details.",
            "LAST-PARA has the final details.",
        ]

    def _click_continue_if_available(self) -> bool:
        return False


class ReasoningAndSearchClient(DeepSeekWebClient):
    def __init__(self) -> None:
        super().__init__(answer_stable_sec=1.0)
        self.snapshots = [
            deepseek_runtime.AnswerSnapshot(
                final_text="",
                reasoning_text='```tool_call\n{"name":"delete_file","arguments":{}}\n```',
                reasoning_active=True,
            ),
            deepseek_runtime.AnswerSnapshot(
                final_text="",
                reasoning_text='```tool_call\n{"name":"delete_file","arguments":{}}\n```',
                reasoning_active=True,
            ),
            deepseek_runtime.AnswerSnapshot(
                final_text="Финальный ответ без инструментов.",
                web_search_active=True,
                web_search_seen=True,
            ),
            deepseek_runtime.AnswerSnapshot(
                final_text="Финальный ответ без инструментов.",
                web_search_active=True,
                web_search_seen=True,
            ),
            deepseek_runtime.AnswerSnapshot(
                final_text="Финальный ответ без инструментов.",
                web_search_seen=True,
            ),
        ]
        self.index = 0

    def _get_answer_snapshot(self, _question: str, _previous_messages: list[str]):
        snapshot = self.snapshots[min(self.index, len(self.snapshots) - 1)]
        self.index += 1
        return snapshot

    def _click_continue_if_available(self) -> bool:
        return False


class TimeoutArtifactClient(DeepSeekWebClient):
    def __init__(self) -> None:
        super().__init__(answer_stable_sec=1.0)

    def _get_answer_snapshot(self, _question: str, _previous_messages: list[str]):
        return deepseek_runtime.AnswerSnapshot(
            final_text="",
            reasoning_text="reasoning is still active",
            reasoning_active=True,
            web_search_active=True,
            web_search_seen=True,
        )

    def _click_continue_if_available(self) -> bool:
        return False


class StuckReasoningToolCallClient(DeepSeekWebClient):
    def __init__(self) -> None:
        super().__init__(answer_stable_sec=1.0)

    def _get_answer_snapshot(self, _question: str, _previous_messages: list[str]):
        return deepseek_runtime.AnswerSnapshot(
            final_text='tool_call\n{"name":"read_file","path":"TOGOSHOL/package.json","mode":"slice"}',
            reasoning_text="DeepThink completed",
            reasoning_active=True,
            web_search_seen=False,
        )

    def _click_continue_if_available(self) -> bool:
        return False


class PartialToolCallDuringReasoningClient(DeepSeekWebClient):
    def __init__(self) -> None:
        super().__init__(answer_stable_sec=1.0)
        self.index = 0

    def _get_answer_snapshot(self, _question: str, _previous_messages: list[str]):
        self.index += 1
        if self.index < 8:
            return deepseek_runtime.AnswerSnapshot(
                final_text='tool_call\n{"name":"read_file","arguments":{"filepath":"pack',
                reasoning_text="DeepThink still drafting",
                reasoning_active=True,
            )
        return deepseek_runtime.AnswerSnapshot(
            final_text='tool_call\n{"name":"read_file","arguments":{"filepath":"package.json"}}',
            reasoning_text="DeepThink completed but flag is sticky",
            reasoning_active=True,
        )

    def _click_continue_if_available(self) -> bool:
        return False


class ActiveGenerationToolCallClient(DeepSeekWebClient):
    def __init__(self) -> None:
        super().__init__(answer_stable_sec=1.0)
        self.index = 0

    def _get_answer_snapshot(self, _question: str, _previous_messages: list[str]):
        self.index += 1
        return deepseek_runtime.AnswerSnapshot(
            final_text='tool_call\n{"name":"read_file","arguments":{"filepath":"package.json"}}',
            reasoning_text="DeepThink completed",
            reasoning_active=True,
            generation_active=self.index < 8,
        )

    def _click_continue_if_available(self) -> bool:
        return False


class StickyGenerationFinalAnswerClient(DeepSeekWebClient):
    def __init__(self) -> None:
        super().__init__(answer_stable_sec=1.0)

    def _get_answer_snapshot(self, _question: str, _previous_messages: list[str]):
        return deepseek_runtime.AnswerSnapshot(
            final_text="Сегодня 13 июня 2026 года. DEEPBILL_WEB_SEARCH_DONE.",
            web_search_seen=True,
            generation_active=True,
        )

    def _click_continue_if_available(self) -> bool:
        return False


class LengthLimitRetryClient(DeepSeekWebClient):
    def __init__(self) -> None:
        super().__init__()
        self.new_chat_count = 0
        self.sent_count = 0

    def _wait_for_input_ready(self, timeout: int = 60):
        return object()

    def _get_assistant_messages_texts(self) -> list[str]:
        return []

    def _get_user_messages_texts(self) -> list[str]:
        return []

    def _wait_for_chat_idle_before_send(self, timeout: int = 60) -> None:
        pass

    def _focus_and_fill_input(self, _input_handle, _text: str) -> None:
        pass

    def _send_current_message(
        self,
        _input_handle,
        previous_messages=None,
        question: str = "",
        previous_user_messages=None,
    ) -> None:
        self.sent_count += 1

    def _wait_for_new_answer(self, _previous_messages: list[str], _question: str, _timeout: int) -> str:
        if not self.new_chat_count:
            raise deepseek_runtime.DeepSeekChatLengthLimitReached("limit")
        return "fresh answer"

    def _start_new_chat_locked(self) -> None:
        self.new_chat_count += 1


class StaleNewChatButton:
    def __init__(self) -> None:
        self.click_count = 0

    def is_visible(self) -> bool:
        return True

    def click(self) -> None:
        self.click_count += 1


class StaleNewChatPage:
    def __init__(self) -> None:
        self.limit_visible = True
        self.button = StaleNewChatButton()
        self.goto_count = 0

    def query_selector(self, _selector: str):
        return self.button

    def goto(self, _url: str, wait_until: str, timeout: int) -> None:
        self.goto_count += 1
        self.limit_visible = False


class StaleNewChatClient(DeepSeekWebClient):
    def __init__(self) -> None:
        super().__init__()
        self._page = StaleNewChatPage()

    def _page_has_chat_length_limit_notice(self) -> bool:
        return self._page.limit_visible

    def _wait_for_input_ready(self, timeout: int = 60):
        return object()


def main() -> None:
    clock = FakeClock()
    original_monotonic = deepseek_runtime.time.monotonic
    original_sleep = deepseek_runtime.time.sleep
    deepseek_runtime.time.monotonic = clock.monotonic
    deepseek_runtime.time.sleep = clock.sleep
    try:
        client = ContinueClient()
        answer = client._wait_for_new_answer([], "question", timeout=30)
        slow_clock = FakeClock()
        deepseek_runtime.time.monotonic = slow_clock.monotonic
        deepseek_runtime.time.sleep = slow_clock.sleep
        slow_answer = SlowChunkClient(slow_clock)._wait_for_new_answer([], "question", timeout=30)
        fragmented_clock = FakeClock()
        deepseek_runtime.time.monotonic = fragmented_clock.monotonic
        deepseek_runtime.time.sleep = fragmented_clock.sleep
        fragmented_answer = FragmentedAnswerClient()._wait_for_new_answer([], "question", timeout=30)
        reasoning_clock = FakeClock()
        deepseek_runtime.time.monotonic = reasoning_clock.monotonic
        deepseek_runtime.time.sleep = reasoning_clock.sleep
        reasoning_client = ReasoningAndSearchClient()
        reasoning_client._start_request_journal("runtime-reasoning", "question", 30, True)
        reasoning_answer = reasoning_client._wait_for_new_answer([], "question", timeout=30)
        reasoning_client._finish_request_journal("ok", answer_chars=len(reasoning_answer))
        stuck_reasoning_clock = FakeClock()
        deepseek_runtime.time.monotonic = stuck_reasoning_clock.monotonic
        deepseek_runtime.time.sleep = stuck_reasoning_clock.sleep
        stuck_tool_client = StuckReasoningToolCallClient()
        stuck_tool_client._start_request_journal("runtime-stuck-tool-call", "question", 30, True)
        stuck_tool_answer = stuck_tool_client._wait_for_new_answer([], "question", timeout=30)
        stuck_tool_client._finish_request_journal("ok", answer_chars=len(stuck_tool_answer))
        partial_tool_clock = FakeClock()
        deepseek_runtime.time.monotonic = partial_tool_clock.monotonic
        deepseek_runtime.time.sleep = partial_tool_clock.sleep
        partial_tool_client = PartialToolCallDuringReasoningClient()
        partial_tool_client._start_request_journal("runtime-partial-tool-call", "question", 30, True)
        partial_tool_answer = partial_tool_client._wait_for_new_answer([], "question", timeout=30)
        partial_tool_client._finish_request_journal("ok", answer_chars=len(partial_tool_answer))
        active_generation_clock = FakeClock()
        deepseek_runtime.time.monotonic = active_generation_clock.monotonic
        deepseek_runtime.time.sleep = active_generation_clock.sleep
        active_generation_client = ActiveGenerationToolCallClient()
        active_generation_client._start_request_journal("runtime-active-generation-tool-call", "question", 30, True)
        active_generation_answer = active_generation_client._wait_for_new_answer([], "question", timeout=30)
        active_generation_client._finish_request_journal("ok", answer_chars=len(active_generation_answer))
        sticky_generation_clock = FakeClock()
        deepseek_runtime.time.monotonic = sticky_generation_clock.monotonic
        deepseek_runtime.time.sleep = sticky_generation_clock.sleep
        sticky_generation_client = StickyGenerationFinalAnswerClient()
        sticky_generation_client._start_request_journal("runtime-sticky-generation", "question", 30, True)
        sticky_generation_answer = sticky_generation_client._wait_for_new_answer([], "question", timeout=30)
        sticky_generation_client._finish_request_journal("ok", answer_chars=len(sticky_generation_answer))
        timeout_clock = FakeClock()
        deepseek_runtime.time.monotonic = timeout_clock.monotonic
        deepseek_runtime.time.sleep = timeout_clock.sleep
        with tempfile.TemporaryDirectory() as debug_dir:
            original_debug_dir = deepseek_runtime.DEBUG_ARTIFACT_DIR
            deepseek_runtime.DEBUG_ARTIFACT_DIR = debug_dir
            try:
                timeout_client = TimeoutArtifactClient()
                timeout_client._start_request_journal("runtime-timeout", "question", 1, True)
                try:
                    timeout_client._wait_for_new_answer([], "question", timeout=1)
                    raise AssertionError("timeout artifact client unexpectedly returned")
                except TimeoutError:
                    pass
                artifact_files = list(Path(debug_dir).glob("*.json"))
                artifact_dir_seen = timeout_client.last_debug_artifact_dir
            finally:
                deepseek_runtime.DEBUG_ARTIFACT_DIR = original_debug_dir
        stale_new_chat = StaleNewChatClient()
        stale_new_chat._start_new_chat_locked()
        busy_before_send = BusyBeforeSendClient()
        busy_before_send._start_request_journal("runtime-busy-before-send", "question", 5, False)
        busy_before_send._wait_for_chat_idle_before_send(timeout=5)
        busy_before_send._finish_request_journal("ok")
    finally:
        deepseek_runtime.time.monotonic = original_monotonic
        deepseek_runtime.time.sleep = original_sleep

    assert answer == "part one\n\npart two"
    assert client.continue_clicks == 1
    assert DeepSeekWebClient._merge_continued_answer(
        "alpha long shared tail for overlap",
        "long shared tail for overlap beta",
    ) == "alpha long shared tail for overlap beta"

    assert slow_answer == "part one part two"
    assert fragmented_answer == FragmentedAnswerClient.full_answer
    assert reasoning_answer == "Финальный ответ без инструментов."
    assert reasoning_client.last_reasoning_chars > 0
    assert reasoning_client.last_web_search_seen
    assert "delete_file" not in reasoning_answer
    assert reasoning_client.last_request_journal["request_id"] == "runtime-reasoning"
    assert any(event["stage"] == "answer_state" for event in reasoning_client.last_request_journal["events"])
    assert '"path":"TOGOSHOL/package.json"' in stuck_tool_answer
    assert any(
        event["stage"] == "complete_tool_call_returned_after_reasoning_grace"
        for event in stuck_tool_client.last_request_journal["events"]
    )
    assert partial_tool_answer.endswith('"filepath":"package.json"}}')
    assert any(
        event["stage"] == "complete_tool_call_waiting_for_reasoning_flag"
        for event in partial_tool_client.last_request_journal["events"]
    )
    assert active_generation_answer.endswith('"filepath":"package.json"}}')
    active_generation_events = active_generation_client.last_request_journal["events"]
    assert any(event.get("generation_active") for event in active_generation_events if event["stage"] == "answer_state")
    assert any(
        event["stage"] == "complete_tool_call_returned_after_reasoning_grace"
        for event in active_generation_events
    )
    assert sticky_generation_answer.endswith("DEEPBILL_WEB_SEARCH_DONE.")
    sticky_generation_events = sticky_generation_client.last_request_journal["events"]
    assert any(
        event["stage"] == "stable_answer_waiting_for_generation_flag"
        for event in sticky_generation_events
    )
    assert any(
        event["stage"] == "stable_answer_returned_after_generation_grace"
        for event in sticky_generation_events
    )
    assert artifact_files
    assert artifact_dir_seen.endswith(Path(debug_dir).name)
    assert DeepSeekWebClient._strip_reasoning_text(
        "Reasoning:\nI may call a tool.\n\nFinal answer:\nSafe final answer."
    ) == "Safe final answer."
    assert DeepSeekWebClient._latest_assistant_message_text(
        [FragmentedAnswerClient.full_answer, "LAST-PARA has the final details."]
    ) == FragmentedAnswerClient.full_answer
    assert DeepSeekWebClient._select_assistant_answer_text(
        ["OK", 'We need to respond only with "OK" as per user instruction. No tool use.'],
        "Ответь только словом OK.",
        "",
    ) == "OK"
    assert DeepSeekWebClient._sanitize_answer_text(
        'We need to answer with only "LONG-FOLLOWUP-OK". The user said "Answer with only LONG-FOLLOWUP-OK." '
        "So just output that exact string.",
        "Answer with only LONG-FOLLOWUP-OK.",
        "",
    ) == "LONG-FOLLOWUP-OK"
    assert DeepSeekWebClient._sanitize_answer_text(
        'We need to respond to the user. The assistant already made the tool call and got the result. '
        'Now we need to answer: confirm shortly. So we can respond with something like: '
        '"Команда выполнена. Результат: 123" or just "123". Since no additional tool is needed.',
        'Вызови run_terminal_command с командой python3 -c "print(123)", затем коротко подтверди результат.',
        "",
    ) == "123"
    assert DeepSeekWebClient._sanitize_answer_text(
        "We need to update app.py and then run tests.",
        "Что делать дальше?",
        "",
    ) == "We need to update app.py and then run tests."
    assert DeepSeekWebClient._sanitize_answer_text(
        "We need to compute 17 * 19. 17*19 = 323. Return final answer and marker.",
        "Реши коротко: 17 * 19. Верни только финальный ответ и маркер DEEPBILL_DEEPTHINK_DONE.",
        "",
    ) == ""
    assert DeepSeekWebClient._sanitize_answer_text(
        'We need to output exactly "323 DEEPBILL_RT_DONE" as the final answer.',
        "Ответь только этой строкой: 323 DEEPBILL_RT_DONE",
        "",
    ) == "323 DEEPBILL_RT_DONE"
    assert DeepSeekWebClient._sanitize_answer_text(
        'We need to output a short confirmation of the result. The command printed "123". '
        'So just say something like "Результат: 123" or "Команда выполнена, вывод: 123". Keep it concise.',
        'Вызови run_terminal_command с командой python3 -c "print(123)", затем коротко подтверди результат.',
        "",
    ) == "123"
    assert DeepSeekWebClient._sanitize_answer_text(
        'Мы получили результат read_file: содержимое "orange". Теперь нужно подтвердить. '
        'Ответ: "Содержимое файла изменено на orange."\n\n'
        "Но по инструкции, если не нужны инструменты, отвечаем обычным текстом. Сейчас инструменты не нужны.",
        "После этого ещё раз прочитай файл через read_file и коротко подтверди.",
        "",
    ) == "Содержимое файла изменено на orange."
    assert DeepSeekWebClient._is_chat_length_limit_text("Length limit reached. Please start a new chat.")
    retry_client = LengthLimitRetryClient()
    assert retry_client.send_text_prompt("question", timeout_sec=1) == "fresh answer"
    assert retry_client.new_chat_count == 1
    assert retry_client.sent_count == 2
    assert stale_new_chat._page.button.click_count == 1
    assert stale_new_chat._page.goto_count == 1
    assert not stale_new_chat._page.limit_visible
    assert any(
        event["stage"] == "chat_busy_before_send"
        for event in busy_before_send.last_request_journal["events"]
    )
    assert any(
        event["stage"] == "chat_idle_before_send"
        for event in busy_before_send.last_request_journal["events"]
    )

    sticky_composer = StickyComposerClient()
    assert not sticky_composer._submission_observed(object(), [], "question", [])
    assert SubmittedUserMessageClient()._submission_observed(object(), [], "question", [])

    class ClosedClient:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class ReplacementClient:
        created: list["ReplacementClient"] = []

        def __init__(self, headless: bool = False, user_data_dir: str = "", answer_stable_sec: float = 0) -> None:
            self.started = False
            self.__class__.created.append(self)

        def start(self) -> None:
            self.started = True

    closed = ClosedClient()
    worker = deepseek_runtime.BrowserWorker(headless=True, user_data_dir="profile")
    original_client_class = deepseek_runtime.DeepSeekWebClient
    deepseek_runtime.DeepSeekWebClient = ReplacementClient
    try:
        def action(client):
            if client is closed:
                raise RuntimeError("Target page, context or browser has been closed")
            return "recovered"

        replacement, result = worker._with_client_recovery(closed, "test", action)
    finally:
        deepseek_runtime.DeepSeekWebClient = original_client_class

    assert result == "recovered"
    assert closed.closed
    assert replacement is ReplacementClient.created[-1]
    assert replacement.started

    class SlowGateClient:
        entered = threading.Event()
        release = threading.Event()

        def __init__(self, headless: bool = False, user_data_dir: str = "", answer_stable_sec: float = 0) -> None:
            self.started = False

        def start(self) -> None:
            self.started = True

        def close(self) -> None:
            self.release.set()

        def set_answer_stable_sec(self, _value) -> None:
            pass

        def send_text_prompt(self, _prompt: str, timeout_sec: int = 1) -> str:
            self.entered.set()
            self.release.wait(timeout=2)
            return "slow done"

    original_client_class = deepseek_runtime.DeepSeekWebClient
    deepseek_runtime.DeepSeekWebClient = SlowGateClient
    busy_worker = deepseek_runtime.BrowserWorker(headless=True, user_data_dir="profile")
    busy_worker.start()
    busy_result: dict[str, object] = {}

    def run_slow_request() -> None:
        busy_result["answer"] = busy_worker._request(
            "ask_text",
            {"prompt": "slow", "timeout": 1, "answer_stable_sec": 1},
            timeout=2,
        )

    busy_thread = threading.Thread(target=run_slow_request)
    busy_thread.start()
    assert SlowGateClient.entered.wait(timeout=1)
    try:
        busy_worker._request("new_chat", {}, timeout=1)
        raise AssertionError("busy browser accepted a second command")
    except RuntimeError as exc:
        assert "Browser is busy with ask_text" in str(exc)
    SlowGateClient.release.set()
    busy_thread.join(timeout=2)
    busy_worker.stop()
    deepseek_runtime.DeepSeekWebClient = original_client_class
    assert busy_result["answer"] == "slow done"

    class TimeoutRecoveryClient:
        created: list["TimeoutRecoveryClient"] = []
        first_prompt_started = threading.Event()
        first_client_closed = threading.Event()

        def __init__(self, headless: bool = False, user_data_dir: str = "", answer_stable_sec: float = 0) -> None:
            self.index = len(self.__class__.created)
            self.closed = threading.Event()
            self.__class__.created.append(self)

        def start(self) -> None:
            pass

        def close(self) -> None:
            self.closed.set()
            if self.index == 0:
                self.__class__.first_client_closed.set()

        def set_answer_stable_sec(self, _value) -> None:
            pass

        def send_text_prompt(self, _prompt: str, timeout_sec: int = 1) -> str:
            if self.index == 0:
                self.__class__.first_prompt_started.set()
                self.closed.wait(timeout=2)
                raise RuntimeError("Target page, context or browser has been closed")
            return "recovered answer"

    deepseek_runtime.DeepSeekWebClient = TimeoutRecoveryClient
    recovery_worker = deepseek_runtime.BrowserWorker(headless=True, user_data_dir="profile")
    recovery_worker.start()
    try:
        recovery_worker._request(
            "ask_text",
            {"prompt": "will timeout", "timeout": 1, "answer_stable_sec": 1},
            timeout=0.1,
        )
        raise AssertionError("timed-out browser command unexpectedly returned")
    except TimeoutError as exc:
        assert "Browser command timed out: ask_text" in str(exc)
    assert TimeoutRecoveryClient.first_prompt_started.wait(timeout=1)
    assert TimeoutRecoveryClient.first_client_closed.wait(timeout=4)
    for _ in range(50):
        if not recovery_worker.diagnostics()["recovering_after_timeout"]:
            break
        original_sleep(0.02)
    assert not recovery_worker.diagnostics()["recovering_after_timeout"]
    assert recovery_worker._request(
        "ask_text",
        {"prompt": "after recovery", "timeout": 1, "answer_stable_sec": 1},
        timeout=1,
    ) == "recovered answer"
    recovery_worker.stop()
    deepseek_runtime.DeepSeekWebClient = original_client_class

    class WatchdogTimeoutClient:
        created: list["WatchdogTimeoutClient"] = []
        attempts = 0

        def __init__(self, headless: bool = False, user_data_dir: str = "", answer_stable_sec: float = 0) -> None:
            self.closed = False
            self.__class__.created.append(self)

        def start(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

        def set_answer_stable_sec(self, _value) -> None:
            pass

        def send_text_prompt(
            self,
            _prompt: str,
            timeout_sec: int = 1,
            use_reasoning: bool = False,
            request_id: str = "",
        ) -> str:
            self.__class__.attempts += 1
            if self.__class__.attempts <= 2:
                raise TimeoutError("DeepSeek answer timed out. Last fragment: ''")
            return f"watchdog recovered {request_id} reasoning={use_reasoning}"

    deepseek_runtime.DeepSeekWebClient = WatchdogTimeoutClient
    watchdog_worker = deepseek_runtime.BrowserWorker(headless=True, user_data_dir="profile")
    watchdog_worker.watchdog_hang_limit = 2
    watchdog_worker.start()
    try:
        for index in range(2):
            try:
                watchdog_worker.ask_text("watchdog timeout", timeout=1, request_id=f"watchdog-{index}")
                raise AssertionError("watchdog timeout request unexpectedly returned")
            except TimeoutError:
                pass
        watchdog_diag = watchdog_worker.diagnostics()
        assert watchdog_diag["watchdog_restarts"] >= 1
        assert watchdog_diag["consecutive_hangs"] == 0
        assert "watchdog" in watchdog_diag["last_watchdog_reason"]
        assert watchdog_worker.ask_text(
            "after watchdog",
            timeout=1,
            use_reasoning=True,
            request_id="watchdog-ok",
        ).startswith("watchdog recovered watchdog-ok reasoning=True")
        assert watchdog_worker.diagnostics()["consecutive_hangs"] == 0
    finally:
        watchdog_worker.stop()
        deepseek_runtime.DeepSeekWebClient = original_client_class
    print("runtime_tests: ok")


if __name__ == "__main__":
    main()

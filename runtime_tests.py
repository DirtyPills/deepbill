#!/usr/bin/env python3
"""Focused checks for chat-only DeepSeek continuation assembly."""

from __future__ import annotations

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


class LengthLimitRetryClient(DeepSeekWebClient):
    def __init__(self) -> None:
        super().__init__()
        self.new_chat_count = 0
        self.sent_count = 0

    def _wait_for_input_ready(self, timeout: int = 60):
        return object()

    def _get_assistant_messages_texts(self) -> list[str]:
        return []

    def _focus_and_fill_input(self, _input_handle, _text: str) -> None:
        pass

    def _send_current_message(self, _input_handle, _previous_messages=None, _question: str = "") -> None:
        self.sent_count += 1

    def _wait_for_new_answer(self, _previous_messages: list[str], _question: str, _timeout: int) -> str:
        if not self.new_chat_count:
            raise deepseek_runtime.DeepSeekChatLengthLimitReached("limit")
        return "fresh answer"

    def _start_new_chat_locked(self) -> None:
        self.new_chat_count += 1


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
    assert DeepSeekWebClient._latest_assistant_message_text(
        [FragmentedAnswerClient.full_answer, "LAST-PARA has the final details."]
    ) == FragmentedAnswerClient.full_answer
    assert DeepSeekWebClient._is_chat_length_limit_text("Length limit reached. Please start a new chat.")
    retry_client = LengthLimitRetryClient()
    assert retry_client.send_text_prompt("question", timeout_sec=1) == "fresh answer"
    assert retry_client.new_chat_count == 1
    assert retry_client.sent_count == 2

    StickyComposerClient()._ensure_message_submitted(object(), [], "question", timeout=1)

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
    print("runtime_tests: ok")


if __name__ == "__main__":
    main()

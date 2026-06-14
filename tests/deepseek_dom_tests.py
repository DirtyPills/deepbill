#!/usr/bin/env python3
"""DOM regressions for the DeepSeek web runtime selectors."""

from __future__ import annotations

import tempfile

from playwright.sync_api import sync_playwright

from _path import PROJECT_ROOT  # noqa: F401
from deepseek_runtime import DeepSeekWebClient


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"ok - {message}")


HTML = """
<!doctype html>
<html>
  <body>
    <main>
      <div class="_58b31c9">
        <div id="deepthink" tabindex="0" aria-pressed="false" class="f79352dc ds-toggle-button ds-toggle-button--m">
          <span>DeepThink</span>
        </div>
        <div id="search" tabindex="0" aria-pressed="true" class="f79352dc ds-toggle-button ds-toggle-button--m ds-toggle-button--selected">
          <span>Search</span>
        </div>
      </div>
      <article data-testid="assistant-message">
        <section class="ds-deepthink-panel">
          <span>DeepThink completed</span>
        </section>
        <div class="markdown">
tool_call
{"name":"read_file","path":"TOGOSHOL/package.json","mode":"slice","offset":1,"limit":200}
        </div>
      </article>
    </main>
    <script>
      const deepthink = document.getElementById("deepthink");
      deepthink.addEventListener("click", () => {
        const next = deepthink.getAttribute("aria-pressed") !== "true";
        deepthink.setAttribute("aria-pressed", String(next));
        deepthink.classList.toggle("ds-toggle-button--selected", next);
      });
    </script>
  </body>
</html>
"""


ACTIVE_HTML = """
<!doctype html>
<html>
  <body>
    <main>
      <article data-testid="assistant-message">
        <div role="status" aria-busy="true" class="loading">Thinking...</div>
      </article>
    </main>
  </body>
</html>
"""


USER_MESSAGE_HTML = """
<!doctype html>
<html>
  <body>
    <main>
      <article data-testid="user-message">
        <div>please create index.html</div>
      </article>
      <article data-testid="assistant-message">
        <div class="markdown">I will create index.html.</div>
      </article>
      <article>
        <div>generic article text should not confirm user submission</div>
      </article>
    </main>
  </body>
</html>
"""


UNLABELLED_REASONING_HTML = """
<!doctype html>
<html>
  <body>
    <main>
      <article data-testid="assistant-message">
        <section class="ds-deepthink-panel">
          <header>DeepThink</header>
          <div class="markdown">We need to respond only with the word OK. So just output OK.</div>
        </section>
        <div class="markdown">OK</div>
      </article>
    </main>
  </body>
</html>
"""


def main() -> None:
    with tempfile.TemporaryDirectory() as profile_dir:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch_persistent_context(
                user_data_dir=profile_dir,
                headless=True,
                viewport={"width": 900, "height": 700},
            )
            try:
                page = browser.new_page()
                page.set_content(HTML)
                client = DeepSeekWebClient(headless=True)
                client._page = page

                client._set_reasoning_enabled(True)
                assert_true(client.last_reasoning_toggle_found, "DeepThink aria-pressed toggle is found")
                assert_true(
                    page.locator("#deepthink").get_attribute("aria-pressed") == "true",
                    "DeepThink aria-pressed toggle is enabled by runtime",
                )

                snapshot = client._get_answer_dom_snapshot()
                final_text = "\n".join(snapshot.get("final_texts") or [])
                assert_true("TOGOSHOL/package.json" in final_text, "final answer text keeps direct tool call")
                assert_true(not snapshot.get("reasoning_active"), "selected DeepThink toggle is not active reasoning")
                assert_true(not snapshot.get("web_search_seen"), "selected Search toggle is not treated as web search")
                assert_true(not snapshot.get("web_search_active"), "selected Search toggle is not active web search")
                assert_true(not snapshot.get("generation_active"), "finished answer is not treated as active generation")

                page.set_content(UNLABELLED_REASONING_HTML)
                clean_snapshot = client._get_answer_dom_snapshot()
                clean_text = "\n".join(clean_snapshot.get("final_texts") or [])
                assert_true("We need to respond" not in clean_text, "unlabelled DeepThink body is stripped")
                assert_true("OK" in clean_text, "final answer below DeepThink panel is preserved")

                page.set_content(ACTIVE_HTML)
                active_snapshot = client._get_answer_dom_snapshot()
                assert_true(active_snapshot.get("reasoning_active"), "real busy status is still detected as active reasoning")
                assert_true(active_snapshot.get("generation_active"), "busy status is detected as active generation")

                page.set_content(USER_MESSAGE_HTML)
                user_texts = client._get_user_messages_texts()
                assert_true(user_texts == ["please create index.html"], "user message selector ignores assistant and generic article blocks")
            finally:
                browser.close()
    print("deepseek_dom_tests: ok")


if __name__ == "__main__":
    main()

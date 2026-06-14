# DeepThink and Web Search Implementation Plan

## Goal

Add support for DeepSeek chat with reasoning enabled while preserving the current adapter contract:

- normal mode works exactly as today;
- reasoning mode waits for reasoning and final answer, but returns only the final answer;
- auto mode can route harder agent requests to reasoning;
- DeepSeek's built-in web search can run for a long time without breaking the adapter;
- OpenAI-compatible clients, especially Roo Code and Continue, still receive normal Chat Completions, SSE chunks, native `tool_calls`, and estimated `usage`.

This document is a plan only. It does not implement the feature.

## Current Project State

### GUI

`app.py` owns user settings in `dbill_settings.json`, starts `BrowserWorker`, and optionally starts `OpenAIAdapter`.

Current persisted settings:

- `adapter_enabled`
- `adapter_port`
- `timeout_sec`
- `answer_stable_sec`

Current GUI controls already expose:

- adapter port and start/stop;
- answer timeout;
- final-answer stability wait;
- manual chat send;
- open browser/new chat.

### Browser Runtime

`deepseek_runtime.py` contains two layers:

- `DeepSeekWebClient`: Playwright automation for the actual DeepSeek page.
- `BrowserWorker`: single worker thread and command queue used by GUI and adapter.

Important current runtime behavior:

- Prompt entry is filled through several fallback methods: clipboard, insert text, type, fill, DOM.
- The send path waits for input ready, fills input, sends, then calls `_wait_for_new_answer`.
- `_wait_for_new_answer` polls visible assistant message candidates, waits until text is stable for `answer_stable_sec`, clicks visible `Continue`, and joins continuation segments.
- `_get_assistant_messages_texts` clones visible answer nodes and removes controls such as buttons, copy/download UI, inputs, textareas, scripts, and styles.
- The runtime already handles DeepSeek length-limit notices by opening a fresh chat and retrying.
- The worker now has busy/recovery diagnostics and avoids letting a stuck browser command block all future requests silently.

### Adapter

`openai_adapter.py` exposes:

- `/v1/chat/completions`
- `/chat/completions`
- `/v1/completions`
- `/v1/models`
- `/health`

Important current adapter behavior:

- `_build_prompt` converts OpenAI messages, tool calls, and tool results into a single text prompt for the DeepSeek web UI.
- `_tools_to_prompt` injects a tool-call protocol when OpenAI `tools` are present.
- `ToolCallParser` converts DeepSeek text such as `tool_call ...` into native OpenAI `tool_calls`.
- Sync and stream chat paths currently ask the browser for a complete answer first, then parse/stream the cleaned result.
- Estimated `usage` is already returned for sync and stream chat completions.
- A single-browser gate has explicit backpressure: one active browser request,
  a bounded wait queue, `429 server_busy` when the queue is full or the browser
  stays occupied longer than `DEEPBILL_ADAPTER_BUSY_TIMEOUT`, and
  `503 server_unavailable` when the circuit breaker opens after repeated
  runtime failures.

### Tests

Current deterministic coverage:

- `tests/runtime_tests.py`: continuation joining, length-limit retry, worker recovery/busy behavior.
- `tests/tool_call_parser_tests.py`: dirty tool-call parsing and UI-noise cleanup.
- `tests/adapter_route_tests.py`: HTTP contract, SSE, tool calls, retry, create/read/edit chains, usage.
- `tests/live_continue_route_test.py`: live adapter checks against the logged-in DeepSeek browser, including large answers and Continue.

## Proposed User-Facing Behavior

Add GUI setting:

```text
Reasoning mode: Off | Auto | On
```

Recommended UI: `ttk.Combobox` or three radio buttons. A true checkbox is awkward because there are three states. If the UI text must feel like a flag, label it `Использовать рассуждения` and place a compact mode selector next to it.

Mode semantics:

- `Off`: current behavior, no DeepThink/reasoning toggle.
- `On`: every request uses DeepThink/reasoning.
- `Auto`: adapter/runtime chooses reasoning for hard requests.

Persist as:

```json
{
  "reasoning_mode": "off|auto|on"
}
```

Default should be `off` to avoid changing existing behavior.

## Request Routing Design

Introduce a small structured request options object passed from adapter to worker:

```python
@dataclass
class BrowserRequestOptions:
    reasoning_mode: str  # off | auto | on
    use_reasoning: bool
    allow_web_search_wait: bool
```

Practical minimal version can just pass `use_reasoning: bool` first, then add richer options when search probes are implemented.

### Off

Adapter sends requests exactly as now.

### On

Adapter calls:

```python
worker.ask_text(prompt, timeout=timeout, use_reasoning=True)
```

Worker forwards to:

```python
DeepSeekWebClient.send_text_prompt(prompt, timeout_sec, use_reasoning=True)
```

The browser client toggles DeepThink on before sending.

### Auto

Auto should be deterministic and logged. The agent does not directly control DeepSeek Web, so "agent decides" needs one or both of these mechanisms:

1. Model aliases exposed by `/v1/models`:
   - `deepseek-chat` or `deepbill` means GUI/default routing;
   - `deepbill-deepthink` forces reasoning;
   - `deepbill-auto` requests auto routing.

2. Adapter heuristics when GUI mode is `auto`:
   - use reasoning when prompt/messages are large;
   - use reasoning when tools are present and task looks multi-step;
   - use reasoning for codebase analysis, debugging, refactoring, tests, architecture, long planning, unclear failures;
   - use normal chat for short direct answers, simple file reads, confirmations, and small tool-followups.

Recommended initial heuristic:

```text
use_reasoning if:
- total estimated prompt tokens > normal_context_soft_limit
- tools are present and latest user asks to implement/debug/refactor/test/analyze
- message count >= 6 and tool result chars are high
- explicit words appear: сложн, архитектур, рефактор, debug, traceback, тест, план, проанализируй
```

Log every decision:

```text
stage=reasoning_route request_id=... gui_mode=auto use_reasoning=true reason=tools+debug
```

## DeepSeek Web UI Automation

Add selectors to `DeepSeekWebClient`:

```python
REASONING_TOGGLE_SELECTORS = [...]
WEB_SEARCH_TOGGLE_SELECTORS = [...]
REASONING_BLOCK_SELECTORS = [...]
WEB_SEARCH_STATUS_SELECTORS = [...]
```

DeepSeek Web labels may be localized, so selectors should support at least:

- `DeepThink`
- `R1`
- `Reason`
- `Reasoning`
- `Глубокое мышление`
- `Рассуждения`
- `Поиск`
- `Search`
- `Web Search`
- `Internet`

Implementation shape:

```python
def _set_reasoning_enabled(self, enabled: bool) -> None:
    # Find the visible toggle/button.
    # Detect current pressed/selected state via aria-pressed, aria-selected, data-state,
    # class names, or visible selected styling.
    # Click only if current state differs.
```

Important: this must run while holding `_request_lock`, immediately before `_focus_and_fill_input`.

Do not rely on text-only button names forever. Add a generic DOM fallback that looks at buttons near the composer and checks labels/aria/title/text.

## Waiting for Reasoning and Final Answer

Current `_wait_for_new_answer` returns when latest assistant text is stable. With DeepThink, that may be too early or may capture reasoning text if the UI renders reasoning as visible text.

Add a new answer collector that returns structured parts:

```python
@dataclass
class AnswerSnapshot:
    final_text: str
    reasoning_text: str
    web_search_active: bool
    reasoning_active: bool
    raw_texts: list[str]
```

Then change waiting logic from "latest visible text stable" to:

1. detect all visible answer candidates;
2. split reasoning panels from final answer panels;
3. wait while reasoning indicator is active;
4. wait while web search indicator is active;
5. wait until final answer text is non-empty and stable for `answer_stable_sec`;
6. click Continue if present;
7. return final answer only.

The main invariant:

```text
DeepThink reasoning may be read for state detection, but must never be returned to OpenAI clients or included in tool-call parsing.
```

## Filtering Reasoning Text

Current `_get_assistant_messages_texts` removes controls and returns cleaned visible text. For reasoning mode, add a second layer:

```python
def _get_answer_snapshot(self, question: str, previous_messages: list[str]) -> AnswerSnapshot:
    ...
```

Suggested DOM strategy:

- identify likely reasoning nodes by text/aria/class:
  - `reasoning`
  - `thinking`
  - `deepthink`
  - `мысл`
  - `рассужд`
  - collapsed/expanded reasoning containers;
- remove reasoning nodes from cloned final answer candidates before reading final text;
- separately read reasoning nodes only for diagnostics and "is it still active?" checks;
- keep code blocks and final markdown intact;
- keep the existing control removal for Copy/Download.

Fallback text strategy if DOM structure is hard:

- strip leading sections with headings like `Thinking`, `Thought`, `Reasoning`, `Рассуждения`;
- strip collapsible panel text if final answer also appears as a separate later candidate;
- prefer the lowest visible final answer candidate after removing known reasoning containers.

Do not use broad regex removal as the primary mechanism; it can delete legitimate answer text.

## Web Search Handling

DeepSeek built-in web search happens inside the web chat. The adapter should not fetch the internet itself.

Runtime requirements:

- do not time out just because search produces no answer for a while;
- recognize search-in-progress indicators;
- wait until search is done and final answer stabilizes;
- return only the final answer, not search status UI, sources panel controls, or internal progress text.

Add search-aware state detection:

```python
def _page_has_active_web_search(self) -> bool:
    # visible labels/spinners near assistant answer:
    # "Searching", "Search web", "Поиск", "Ищу", "Sources", loading spinner etc.
```

Waiting rule:

```text
If search is active, keep waiting inside the original request timeout.
Only return after search is inactive and final answer text is stable.
```

The existing worker recovery/busy timeout fix remains important: long search should occupy the browser cleanly, while overlapping requests get `server_busy` instead of corrupting the browser state.

## Larger Context and Output Window

Reasoning chat may accept larger input and produce larger output. Treat this as a routing/config concern:

- add config constants:
  - `DEEPBILL_REASONING_CONTEXT_SOFT_LIMIT`
  - `DEEPBILL_REASONING_TIMEOUT`
  - `DEEPBILL_REASONING_MAX_TIMEOUT`
  - `DEEPBILL_REASONING_ANSWER_STABLE_SEC` optionally;
- in `auto`, route oversized prompts to reasoning instead of failing early;
- increase timeout ceiling only for reasoning/search requests;
- keep `usage` estimation working exactly as now.

Do not remove current context buffering. Instead, make routing use estimated token counts and existing `MAX_CONTEXT_BUFFER_CHARS`.

## Adapter Changes

Add to `OpenAIAdapter.__init__`:

```python
self.reasoning_mode = ...
```

But the setting lives in GUI/browser worker, so cleaner design:

- `BrowserWorker` owns default `reasoning_mode`;
- GUI updates worker setting;
- adapter asks worker diagnostics/config or receives mode at construction.

Minimal implementation path:

1. GUI passes `reasoning_mode` to `BrowserWorker`.
2. `BrowserWorker.ask_text(..., use_reasoning: Optional[bool] = None)`.
3. `OpenAIAdapter._ask_text` computes `use_reasoning`.
4. Adapter passes it to worker.

Update logging:

```text
chat.completions ... reasoning_mode=auto use_reasoning=true
stage=deepseek_request ... reasoning=true search_wait=true
stage=deepseek_response ... answer_chars=... reasoning_chars=... search_seen=true
```

Update `/health` diagnostics:

```json
{
  "reasoning_mode": "auto",
  "last_use_reasoning": true,
  "last_reasoning_chars": 1234,
  "last_web_search_seen": true,
  "last_web_search_wait_sec": 17.2
}
```

## GUI Changes

In `default_settings`:

```python
"reasoning_mode": "off"
```

In `load_settings`:

- normalize to `off|auto|on`;
- default unknown values to `off`.

In `DBillChatApp.__init__`:

- create `self.reasoning_mode_var`;
- pass mode to `BrowserWorker`.

In `_build_ui`:

- add one compact row near timeout controls:

```text
Использовать рассуждения: [Off/Auto/On]
```

In apply handler:

```python
def _apply_reasoning_mode(self) -> None:
    mode = normalize_reasoning_mode(...)
    self.browser.set_reasoning_mode(mode)
    self.settings["reasoning_mode"] = mode
    self._persist_settings()
```

Manual chat should use the same worker default as adapter.

## Tool Calls With Reasoning

Reasoning text must be stripped before `_parse_assistant_answer`.

Flow remains:

```text
DeepSeek final answer only -> ToolCallParser -> native OpenAI tool_calls
```

Risks:

- DeepThink may include tool-call-looking JSON inside reasoning.
- If reasoning leaks into adapter parsing, tools could execute from hidden reasoning instead of final answer.

Mitigation:

- never pass reasoning text to `ToolCallParser`;
- add tests where reasoning contains fake `tool_call` and final answer is plain text;
- add tests where final answer contains real `tool_call` after reasoning.

## Test Plan

### Deterministic Runtime Tests

Add fake clients/pages to `tests/runtime_tests.py`:

- reasoning block visible, final answer later appears;
- reasoning contains fake tool call, final answer is plain answer;
- reasoning active for several polls, final answer stable only after reasoning inactive;
- web search active for several polls, final answer returned after search inactive;
- Continue button after reasoning final answer still joins correctly;
- length-limit retry still works with reasoning setting.

### Deterministic Adapter Tests

Extend `tests/adapter_route_tests.py`:

- `reasoning_mode=off` keeps old request path;
- `reasoning_mode=on` passes `use_reasoning=True` to worker;
- `reasoning_mode=auto` routes complex/tool-heavy request to reasoning;
- simple request in auto stays normal;
- stream response returns final answer only and includes `usage`;
- tool-call parsing ignores fake tool calls from reasoning;
- overlapping long web-search request returns `429 server_busy` for a second request.

### Parser Tests

Probably no parser changes if filtering happens before `ToolCallParser`.

Add parser tests only if a text fallback stripper is introduced.

### Live Tests

Create a new script:

```bash
.venv/bin/python tests/live_deepthink_search_test.py --base-url http://127.0.0.1:8080/v1 --timeout 600
```

Live checks:

1. Health reports reasoning diagnostics.
2. Reasoning-on request completes.
3. Returned text does not contain reasoning panel markers.
4. A fake final tool-call request with reasoning enabled returns native `tool_calls`.
5. A built-in web-search prompt completes and returns final answer.
6. During web search, a concurrent request returns `server_busy`, not a broken adapter state.
7. Short follow-up after web search returns the new answer, with no prior answer/search tail leakage.

Example live prompt for web search:

```text
Используй встроенный поиск в интернете DeepSeek и кратко ответь:
какая дата указана сегодня на странице example.com или в результатах поиска?
В конце напиши маркер DEEPBILL_WEB_SEARCH_DONE.
```

The exact target should be stable and harmless. If DeepSeek refuses search or UI labels change, log screenshot/diagnostics and mark the test inconclusive rather than silently passing.

## Implementation Phases

### Phase 1: Settings and Routing

- Add `reasoning_mode` setting to GUI.
- Add worker setter/getter and diagnostics.
- Add adapter routing decision and logging.
- Add deterministic tests proving the selected mode reaches the worker.

### Phase 2: DeepThink Toggle

- Add DeepThink selector list and `_set_reasoning_enabled`.
- Ensure state detection prevents accidental toggling off.
- Add a manual GUI log line when mode changes.
- Add live manual smoke test.

### Phase 3: Structured Answer Snapshot

- Replace raw `_get_assistant_messages_texts` use inside `_wait_for_new_answer` with snapshot logic.
- Preserve current answer joining, sanitizing, Continue handling, length-limit detection.
- Add reasoning filtering tests.

### Phase 4: Web Search Wait

- Add search-active detection.
- Update wait loop to keep waiting while search is active.
- Add diagnostics for search seen/wait duration.
- Add live web-search test.

### Phase 5: Larger Reasoning Limits

- Add reasoning-specific timeout/context env vars.
- Update auto routing to use estimated tokens.
- Update docs and examples.

## Acceptance Criteria

- `Off` mode produces the same deterministic test behavior as today.
- `On` mode toggles DeepThink before sending every request.
- `Auto` mode logs why it chose reasoning or normal mode.
- Reasoning text is never returned to OpenAI clients.
- Reasoning text is never parsed for tool calls.
- Built-in web search waits until final answer is stable.
- Long search/reasoning requests do not corrupt the worker queue.
- `/health` exposes enough diagnostics to debug stuck reasoning/search.
- All deterministic tests pass:

```bash
.venv/bin/python -m py_compile app.py deepseek_runtime.py openai_adapter.py tool_call_parser.py
.venv/bin/python tests/runtime_tests.py
.venv/bin/python tests/tool_call_parser_tests.py
.venv/bin/python tests/adapter_route_tests.py
```

- Live tests pass when the GUI is logged into DeepSeek:

```bash
.venv/bin/python tests/live_continue_route_test.py --base-url http://127.0.0.1:8080/v1 --timeout 360
.venv/bin/python tests/live_deepthink_search_test.py --base-url http://127.0.0.1:8080/v1 --timeout 600
```

## Main Risks

- DeepSeek may change DOM labels/classes for DeepThink or web search.
- Reasoning UI may be rendered inside the same assistant container as final text.
- Web search may expose source cards/buttons that current answer collector could read as prose.
- Auto routing can be surprising if it silently picks reasoning for too many requests.
- Larger outputs increase chances of browser stalls; worker busy/recovery diagnostics must stay visible.

## Recommended Defaults

- `reasoning_mode`: `off`
- Auto routing enabled only when GUI mode is `auto`
- Reasoning timeout: `600` seconds
- Search/reasoning live test timeout: `600` seconds
- Keep final-answer stability default at `2.5` seconds initially; tune only after live evidence.

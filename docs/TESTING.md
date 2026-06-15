# Testing

## Deterministic checks

Run after installation:

```bash
.venv/bin/python -m py_compile app.py deepseek_runtime.py openai_adapter.py tool_call_parser.py scripts/dbill_service.py tests/*.py
.venv/bin/python tests/runtime_tests.py
.venv/bin/python tests/tool_call_parser_tests.py
.venv/bin/python tests/deepseek_dom_tests.py
.venv/bin/python tests/adapter_route_tests.py
.venv/bin/python tests/roocode_simulation_tests.py
.venv/bin/python tests/roocode_heavy_simulation_tests.py
```

`tests/adapter_route_tests.py` uses a deterministic worker and covers plain
requests, SSE shape, dirty tool-call cleanup, retry behavior, backpressure,
circuit breaker recovery, create/read tool chains, edit chains, and
read/edit/read content propagation.

`tests/roocode_simulation_tests.py` runs a Roo Code-like loop with real local
tool execution, including create/read, multiple tool calls in one assistant
message, edit/read/terminal, and reasoning model routing.

`tests/roocode_heavy_simulation_tests.py` covers longer Roo-style workflows:
large project creation, invalid native tool repair, required argument autofill,
`attempt_completion`, serial requests, large final responses, and streamed native
tool calls.

Queue/order regressions are covered in `tests/adapter_route_tests.py`: while one
request is repairing an invalid tool call, a second request must wait and must not
reach the browser worker until repair is complete.

## Live browser checks

1. Start the GUI with a visible browser.
2. Log in to DeepSeek in the persistent profile.
3. Start the adapter in the GUI.
4. Run:

```bash
.venv/bin/python tests/live_continue_route_test.py --base-url http://127.0.0.1:8080/v1 --timeout 360
```

The live route test uses a sandbox in `/tmp/deepbill_live_continue_tools` and
checks:

- plain chat
- SSE completion end marker
- terminal tool call with a fixed safe command
- large monolithic-code answer until the runtime reports a captured `Continue`
- create/read agent flow
- create/read/edit/read agent flow

## Large answer continuation

The live script now sends large monolithic-code requests with larger generated
validation engines until `/health` reports a higher `total_continue_clicks`
value. It checks that the joined code answer stays substantial and sends a short
follow-up request to catch answer-tail leakage. The prompt asks for a final
marker and logs whether DeepSeek honored it, but the hard assertion is the
captured Continue click because tail-marker compliance varies between replies.
Use `--require-continuation` for that strict assertion; the default live run
keeps going through the tool sandbox when DeepSeek declines every large request
before exposing a Continue button.

Before and after a live run, inspect:

```bash
curl http://127.0.0.1:8080/v1/health
tail -n 80 logs/adapter_traffic.jsonl
```

There should be one ordered request chain at a time through the browser:
`http.chat.received -> queue.acquired -> browser.request -> browser.response ->
validating/repair if needed -> http.chat.completed -> queue.released`.

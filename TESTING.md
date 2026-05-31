# Testing

## Deterministic checks

Run after installation:

```bash
.venv/bin/python -m py_compile app.py deepseek_runtime.py openai_adapter.py tool_call_parser.py
.venv/bin/python runtime_tests.py
.venv/bin/python tool_call_parser_tests.py
.venv/bin/python adapter_route_tests.py
```

`adapter_route_tests.py` uses a deterministic worker and covers plain requests,
SSE shape, dirty tool-call cleanup, retry behavior, create/read tool chains,
edit chains, and read/edit/read content propagation.

## Live browser checks

1. Start the GUI with a visible browser.
2. Log in to DeepSeek in the persistent profile.
3. Start the adapter in the GUI.
4. Run:

```bash
.venv/bin/python live_continue_route_test.py --base-url http://127.0.0.1:8080/v1 --timeout 360
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

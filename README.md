# DBillTOGIT

`DBillTOGIT` is the chat and OpenAI-compatible adapter slice of the DEEPBILL
project. It keeps the DeepSeek browser automation, manual chat GUI, tool-aware
OpenAI adapter, adapter status controls, and client examples. Screenshot
capture, screen hotkeys, voice capture, screen docks, and screenshot assets are
not part of this copy.

## What Runs

- `app.py` starts the Tk GUI and the persistent Playwright DeepSeek browser.
- `deepseek_runtime.py` sends text prompts, removes DeepSeek UI controls from
  collected answer text, auto-clicks visible `Continue` buttons, and joins
  continuation segments.
- `openai_adapter.py` exposes Chat Completions and legacy completions for Roo
  Code, Continue, and other OpenAI-compatible clients.
- `tool_call_parser.py` converts tool-call text coming from the web UI into
  native OpenAI `tool_calls`.

## Install

Linux:

```bash
chmod +x install.sh run_linux.sh
./install.sh
```

Windows:

```bat
install_windows.bat
```

The installer creates `.venv`, installs Python dependencies, and downloads the
Playwright Chromium browser.

## Run

Linux:

```bash
./run_linux.sh
```

Windows:

```bat
run_windows.bat
```

The first start opens the DeepSeek web page through a persistent browser
profile. Log in there if DeepSeek asks for authentication. The GUI has an
`Open Browser` button to bring the DeepSeek tab forward again.

The local `deepseek_profile/` directory is created automatically on first
launch. It stores the browser login only on the user's machine and is excluded
from Git.

## Adapter

Start the adapter from the GUI. The default base URL is:

```text
http://127.0.0.1:8080/v1
```

Use model `deepseek-chat` and any non-empty API key in an OpenAI-compatible
client. The adapter supports `/v1/models`, `/v1/chat/completions`,
`/chat/completions`, `/v1/completions`, SSE shaped responses, and native
OpenAI tool-call replies.

Response defaults are prepared for large answers:

- Browser answer timeout default: `360` seconds.
- DeepSeek answer finish wait default: `2.5` seconds. Change `Finish wait,
  sec` in the GUI if a slow connection pauses between streamed browser chunks.
- Adapter timeout default: `360` seconds through `DEEPBILL_ADAPTER_TIMEOUT`.
- Adapter accepted timeout ceiling default: `1800` seconds through
  `DEEPBILL_ADAPTER_MAX_TIMEOUT`.

## Verify

Deterministic checks:

```bash
.venv/bin/python runtime_tests.py
.venv/bin/python tool_call_parser_tests.py
.venv/bin/python adapter_route_tests.py
```

Live Continue-like checks need the GUI adapter running and logged in:

```bash
.venv/bin/python live_continue_route_test.py --base-url http://127.0.0.1:8080/v1 --timeout 360
```

`live_continue_route_test.py` executes a small safe local tool sandbox, checks
create, read, edit, read agent flows through the OpenAI route, and repeats a
large monolithic-code request while watching the runtime's `Continue` counter.
Add `--require-continuation` when the large-answer stress probe must fail if
DeepSeek declines every big code request.

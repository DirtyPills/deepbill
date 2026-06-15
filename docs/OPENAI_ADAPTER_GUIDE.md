# DBill OpenAI Adapter Guide

## Purpose

The adapter presents DeepSeek Web as an OpenAI-compatible local HTTP service.
It is intended for clients such as Roo Code and Continue that already know how
to call Chat Completions and `tools`.

```text
Client -> DBill OpenAI adapter -> Playwright browser -> DeepSeek Web
```

## GUI Flow

1. Run `app.py` through the platform launcher.
2. Use `Open Browser` and log in to DeepSeek if the profile is not authorized.
3. Set the adapter port if needed.
4. Press `Start Adapter`.
5. Copy the OpenAI base URL from the GUI.

## Continue Example

`config/continue-deepbill.config.yaml` is included as a starting point. The essential
values are:

```yaml
provider: openai
model: deepseek-chat
apiBase: http://127.0.0.1:8080/v1
apiKey: local
```

## Roo Code Example

Configure an OpenAI-compatible provider:

- Base URL: `http://127.0.0.1:8080/v1`
- Model: `deepseek-chat`
- API key: any placeholder value

Roo Code requires native OpenAI-style tool calls. The adapter therefore never
relies on Roo receiving XML/prose tool markup. It parses DeepSeek's web text,
validates the call against the incoming OpenAI `tools` schema, repairs incomplete
calls before responding, and returns `finish_reason: "tool_calls"` with native
`message.tool_calls`.

## Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /chat/completions`
- `POST /v1/completions`

`stream: true` returns Server-Sent Events shaped like OpenAI chunks. Chat
stream responses are assembled from the final cleaned browser answer first so
DOM rerenders do not duplicate partially changed text.

## Large Answers

DeepSeek Web can stop a large answer behind a visible `Continue` button. The
browser runtime clicks that button automatically, keeps waiting inside the
original timeout budget, and joins continuation segments without repeating
overlapping text.

The text collector removes real page controls such as copy/download buttons
from cloned answer DOM before reading answer text. Code text inside a code block
is kept; toolbar buttons are not treated as assistant prose.

Tune timeouts with environment variables before launch:

```bash
export DEEPBILL_ADAPTER_TIMEOUT=360
export DEEPBILL_ADAPTER_MAX_TIMEOUT=1800
```

Clients may also pass `timeout` in request JSON.

## Tool Calls

When OpenAI `tools` are present, the adapter asks DeepSeek for fenced
`tool_call` JSON blocks, parses supported dirty web-UI variants, and returns
native `tool_calls` to the client. The client executes the tool, then sends a
new message list containing the assistant tool call and `role: tool` result.

The adapter keeps the single DeepSeek browser locked for the whole HTTP turn:
browser request, response collection, parser validation, any repair prompt, and
the final HTTP/SSE response. That prevents the next Roo API request from entering
the browser while the previous response is still being repaired.

The final prompt rejects hidden/meta reasoning and pins user-visible prose to
Russian unless the user explicitly asks for another language. Tool JSON, paths,
code, shell commands, and schema keys remain exact and are not translated.

## Queue And Diagnostics

Useful environment variables:

```bash
export DEEPBILL_ADAPTER_BUSY_TIMEOUT=1800
export DEEPBILL_ADAPTER_QUEUE_LIMIT=16
export DEEPBILL_TRAFFIC_LOG_DIR=logs
export DEEPBILL_TRAFFIC_LOG_FILE=adapter_traffic.jsonl
export DEEPBILL_TRAFFIC_LOG_DETAILED=1
```

`GET /health` exposes:

- adapter queue counters and active request state
- browser diagnostics, including reasoning/search/continue counters
- the traffic log path, detailed-mode flag, and compact recent events

## Traffic Journal

The GUI shows a compact `Adapter Traffic` log. The detailed file log is toggled
with `Detailed file log` and writes JSONL records such as:

- `http.chat.received`
- `queue.wait`, `queue.acquired`, `queue.released`
- `browser.request`, `browser.response`, `browser.retry`
- `repair.tool_call.*`, `repair.meta.*`
- `http.chat.completed` / `http.chat.stream.completed`

Detailed mode stores full prompts and answers. Keep `logs/` out of git.

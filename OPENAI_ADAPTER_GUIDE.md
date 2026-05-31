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

`continue-deepbill.config.yaml` is included as a starting point. The essential
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

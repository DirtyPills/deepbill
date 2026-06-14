#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
PID_FILE="$ROOT_DIR/runtime/deepbill_adapter.pid"
META_FILE="$ROOT_DIR/runtime/deepbill_adapter.env"
LOG_FILE="$ROOT_DIR/logs/deepbill_adapter.log"

port_from_settings() {
  "$PYTHON" - "$ROOT_DIR/dbill_settings.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
except Exception:
    raw = {}
try:
    port = int(raw.get("adapter_port", 8080))
except Exception:
    port = 8080
print(max(1024, min(port, 65535)))
PY
}

extract_port_arg() {
  local fallback
  fallback="$(port_from_settings)"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --port)
        if [[ $# -ge 2 ]]; then
          printf '%s\n' "$2"
          return 0
        fi
        ;;
      --port=*)
        printf '%s\n' "${1#--port=}"
        return 0
        ;;
    esac
    shift
  done
  printf '%s\n' "$fallback"
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

start_db() {
  if [[ ! -x "$PYTHON" ]]; then
    printf 'DBill venv is missing. Run %s/install.sh first.\n' "$ROOT_DIR" >&2
    return 1
  fi
  if is_running; then
    printf 'DBill quiet adapter is already running (pid %s).\n' "$(cat "$PID_FILE")"
    return 0
  fi
  mkdir -p "$ROOT_DIR/runtime" "$ROOT_DIR/logs"
  rm -f "$PID_FILE"
  local port
  port="$(extract_port_arg "$@")"
  (
    cd "$ROOT_DIR"
    nohup "$PYTHON" "$ROOT_DIR/scripts/dbill_service.py" "$@" >>"$LOG_FILE" 2>&1 &
    printf '%s\n' "$!" >"$PID_FILE"
    printf 'PORT=%s\n' "$port" >"$META_FILE"
  )
  sleep 1
  if is_running; then
    printf 'DBill quiet adapter started (pid %s, log %s).\n' "$(cat "$PID_FILE")" "$LOG_FILE"
    return 0
  fi
  printf 'DBill quiet adapter did not stay running. Check %s.\n' "$LOG_FILE" >&2
  return 1
}

stop_db() {
  if ! is_running; then
    rm -f "$PID_FILE" "$META_FILE"
    printf 'DBill quiet adapter is not running.\n'
    return 0
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      rm -f "$META_FILE"
      printf 'DBill quiet adapter stopped.\n'
      return 0
    fi
    sleep 1
  done
  kill -KILL "$pid" 2>/dev/null || true
  rm -f "$PID_FILE" "$META_FILE"
  printf 'DBill quiet adapter was force-stopped after timeout.\n'
}

status_db() {
  if ! is_running; then
    rm -f "$PID_FILE" "$META_FILE"
    printf 'DBill quiet adapter: stopped.\n'
    return 1
  fi
  local pid port url
  pid="$(cat "$PID_FILE")"
  port="$(sed -n 's/^PORT=//p' "$META_FILE" 2>/dev/null | tail -n 1)"
  if [[ -z "$port" ]]; then
    port="$(port_from_settings)"
  fi
  url="http://127.0.0.1:${port}/v1/health"
  printf 'DBill quiet adapter: running (pid %s).\n' "$pid"
  "$PYTHON" - "$url" <<'PY' || true
import json
import sys
import urllib.error
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    diag = payload.get("diagnostics") or {}
    adapter = diag.get("adapter") or {}
    print(f"health={payload.get('status')} ready={payload.get('ready')} active={adapter.get('active_request_id') or '-'} waiting={adapter.get('waiting_requests')}")
    print(f"circuit={adapter.get('circuit_state')} failures={adapter.get('circuit_failures')} retry_after={adapter.get('circuit_retry_after_sec')}")
except urllib.error.URLError as exc:
    print(f"health=unreachable error={exc}")
PY
}

case "${1:-status}" in
  start)
    shift
    start_db "$@"
    ;;
  stop)
    stop_db
    ;;
  status)
    status_db
    ;;
  restart)
    shift
    stop_db
    start_db "$@"
    ;;
  *)
    printf 'Usage: %s {start|stop|status|restart} [service args]\n' "$0" >&2
    exit 2
    ;;
esac

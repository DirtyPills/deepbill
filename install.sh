#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
chmod +x run_linux.sh scripts/dbillctl.sh scripts/dbill_service.py
printf 'DBillTOGIT dependencies are installed.\n'

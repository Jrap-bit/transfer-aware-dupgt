#!/bin/zsh
# One command for presentation day: loads the LLM (optional), starts the demo server, opens the browser.
#   zsh demo/start_demo.sh            # with live LLM
#   zsh demo/start_demo.sh --no-llm   # without it (the rest of the demo works offline)
cd "$(dirname "$0")/.."
if [[ "$1" != "--no-llm" ]] && command -v lms >/dev/null; then
  if ! lms ps 2>/dev/null | grep -q dupgt-qwen35-9b; then
    echo "Loading Qwen3.5-9B in LM Studio (about 15 s)…"
    lms server start >/dev/null 2>&1
    lms load qwen3.5-9b --identifier dupgt-qwen35-9b --gpu max -c 8192 -y >/dev/null 2>&1 \
      && echo "LLM ready." || echo "LLM did not load; the demo still works without it."
  fi
fi
( sleep 12 && open "http://127.0.0.1:8765" ) &
exec caffeinate -dis .venv/bin/python demo/server.py --port 8765

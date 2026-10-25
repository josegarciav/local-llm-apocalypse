#!/bin/zsh
# Maya — start everything with one command. Works with no internet.
set -e
cd "$(dirname "$0")"
PORT="${MAYA_PORT:-8080}"

# 1. Ollama (the model engine) — the official build with MLX image generation,
#    copied out of Ollama.app into ~/.ollama/engine and run by a login LaunchAgent.
ENGINE="$HOME/.ollama/engine/ollama"
if ! curl -s localhost:11434/api/version >/dev/null 2>&1; then
  echo "• starting Ollama…"
  if [ -f "$HOME/Library/LaunchAgents/com.maya.ollama.plist" ]; then
    launchctl bootstrap gui/$(id -u) "$HOME/Library/LaunchAgents/com.maya.ollama.plist" 2>/dev/null || launchctl kickstart -k gui/$(id -u)/com.maya.ollama
  elif [ -x "$ENGINE" ]; then
    (OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 nohup "$ENGINE" serve > "$HOME/.ollama/logs/maya-server.log" 2>&1 &)
  else
    (OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 nohup ollama serve >/tmp/ollama.log 2>&1 &)
  fi
  for i in {1..40}; do curl -s localhost:11434/api/version >/dev/null 2>&1 && break; sleep 0.5; done
fi
# 2. Maya UI server
if lsof -i :"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "• Maya already running on http://127.0.0.1:$PORT"
else
  echo "• starting Maya on http://127.0.0.1:$PORT"
  nohup python3 server.py > data/server.log 2>&1 &
  sleep 0.8
fi
open "http://127.0.0.1:$PORT" 2>/dev/null || true
echo "✓ ready — press Ctrl+C in this window does NOT stop it; run ./stop.sh to stop."

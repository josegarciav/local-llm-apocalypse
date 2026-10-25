#!/bin/zsh
# Stop the Maya UI server (Ollama keeps running as a background service).
PORT="${MAYA_PORT:-8080}"
PIDS=$(lsof -ti :"$PORT" -sTCP:LISTEN 2>/dev/null)
if [ -n "$PIDS" ]; then kill $PIDS && echo "• Maya stopped"; else echo "• Maya was not running"; fi

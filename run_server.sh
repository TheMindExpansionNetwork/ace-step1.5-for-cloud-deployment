#!/bin/bash
set -e

# ACE-Step 1.5 Persistent API Server Launcher
# This script sets up the environment and launches the persistent server.

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Configuration (Edit these or pass as env vars)
PORT=${PORT:-8000}
HOST=${HOST:-"0.0.0.0"}
WORKERS=1  # MUST be 1 to prevent multiple model loads

echo "🚀 Starting ACE-Step Persistent Server on ${HOST}:${PORT}..."
echo "Running in single-worker mode to keep models resident in VRAM."

# Activate virtualenv if present (common pattern)
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run uvicorn
# --workers 1 is critical for single GPU persistence
uvicorn server:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level info


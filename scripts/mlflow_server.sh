#!/usr/bin/env bash
# Launch MLflow tracking UI for local experiment viewing
# Usage: bash scripts/mlflow_server.sh [--port PORT]
set -euo pipefail

PORT="${1:-5000}"
echo "Starting MLflow UI at http://localhost:${PORT}"
echo "Tracking URI: mlruns/"
mlflow ui \
    --backend-store-uri mlruns \
    --host 127.0.0.1 \
    --port "${PORT}"
# Bind to localhost only for security. Change to 0.0.0.0 if remote access is needed.

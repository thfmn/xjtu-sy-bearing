#!/usr/bin/env bash
# Launch MLflow tracking UI for local experiment viewing
# Usage: bash scripts/mlflow_server.sh [--port PORT]
set -euo pipefail

PORT="${1:-5000}"
echo "Starting MLflow UI at http://localhost:${PORT}"
echo "Tracking URI: mlruns/"
mlflow ui \
    --backend-store-uri mlruns \
    --host 0.0.0.0 \
    --port "${PORT}"

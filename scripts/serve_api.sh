#!/usr/bin/env bash
# Start the farmer-palantir HTTP API on the VM (localhost only; use an SSH tunnel):
#   source /mnt/nvme/robot.env && FARMPAL_TOKEN=... bash scripts/serve_api.sh
#   ssh -L 8700:127.0.0.1:8700 <vm>   then open http://localhost:8700/map
# FARMPAL_* wins; LOBBOT_* (the repo this started in) is the fallback.
set -euo pipefail
cd "$(dirname "$0")/.."
export FARMPAL_TOKEN=${FARMPAL_TOKEN:-${LOBBOT_TOKEN:-}}
: "${FARMPAL_TOKEN:?set FARMPAL_TOKEN to a long random secret}"
export FARMPAL_JOBS=${FARMPAL_JOBS:-${LOBBOT_JOBS:-/mnt/nvme/jobs}}
export FARMPAL_SITES=${FARMPAL_SITES:-${LOBBOT_SITES:-/mnt/nvme/sites}}
exec python -m uvicorn api.server:app --host 127.0.0.1 --port "${PORT:-8700}"

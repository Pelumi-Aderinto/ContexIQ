#!/usr/bin/env bash
# Run the ContextIQ API (uvicorn with auto-reload) and the Streamlit UI together.
#
# Usage:  scripts/dev.sh        (or: make dev)
#
# Ctrl-C stops both processes; if either one exits, the other is stopped too. When ./.env
# exists it is exported first so CONTEXTIQ_* settings and provider keys reach both processes
# (the file must be shell-compatible KEY=value lines, as in .env.example).
#
# Environment overrides: VENV (default .venv), CONTEXTIQ_HOST / CONTEXTIQ_PORT (API bind,
# default 127.0.0.1:8000), CONTEXTIQ_UI_PORT (default 8501).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="${VENV:-.venv}"
if [[ ! -x "$VENV/bin/uvicorn" || ! -x "$VENV/bin/streamlit" ]]; then
  echo "dev.sh: $VENV is missing or incomplete; run 'make setup' (uv sync) first." >&2
  exit 1
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

API_HOST="${CONTEXTIQ_HOST:-127.0.0.1}"
API_PORT="${CONTEXTIQ_PORT:-8000}"
UI_PORT="${CONTEXTIQ_UI_PORT:-8501}"
URL_HOST="$API_HOST"
[[ "$API_HOST" == "0.0.0.0" ]] && URL_HOST="localhost"
export CONTEXTIQ_API_URL="${CONTEXTIQ_API_URL:-http://${URL_HOST}:${API_PORT}}"
# Make the ``app`` package importable for ``streamlit run`` regardless of how the venv was built.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

pids=()
cleanup() {
  trap - EXIT INT TERM
  echo
  echo "dev.sh: stopping API and UI..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "dev.sh: API -> http://${URL_HOST}:${API_PORT}/docs"
"$VENV/bin/uvicorn" app.api.main:app --reload --reload-dir app \
  --host "$API_HOST" --port "$API_PORT" &
pids+=("$!")

echo "dev.sh: UI  -> http://localhost:${UI_PORT}"
"$VENV/bin/streamlit" run app/ui/streamlit_app.py \
  --server.port "$UI_PORT" --server.address 127.0.0.1 --server.headless true &
pids+=("$!")

# Returns as soon as either process exits; the EXIT trap then stops the other one.
wait -n

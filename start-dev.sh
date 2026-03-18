#!/usr/bin/env bash
set -euo pipefail

# Run from the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "──────────────────────────────────────────────"
echo "  AskMyBookmark — Local Dev Server"
echo "──────────────────────────────────────────────"
echo ""

# Ensure the frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
  echo "Installing frontend dependencies..."
  (cd frontend && npm install)
  echo ""
fi

# Start the FastAPI backend in the background using the project venv
echo "Starting backend  → http://localhost:8000"
"$SCRIPT_DIR/.venv/bin/uvicorn" app.ask_my_bookmark:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Give the backend a moment to start
sleep 1

# Start the Next.js frontend in the background
echo "Starting frontend → http://localhost:3000"
(cd frontend && npm run dev) &
FRONTEND_PID=$!

echo ""
echo "Both servers are running. Press Ctrl+C to stop."
echo ""

# Graceful shutdown on Ctrl+C or script exit
cleanup() {
  echo ""
  echo "Shutting down..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT INT TERM

# Block until a signal is received
wait "$BACKEND_PID" "$FRONTEND_PID"

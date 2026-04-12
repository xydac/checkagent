#!/usr/bin/env bash
# CheckAgent scan entrypoint — install deps and run scan
set -euo pipefail

COMMAND="${1:-}"

install_deps() {
  local req="${1:-requirements.txt}"

  # requirements.txt
  if [ -f "$req" ]; then
    echo "Installing dependencies from $req"
    pip install -r "$req"
    return
  fi

  # pyproject.toml
  if [ -f "pyproject.toml" ]; then
    echo "Installing dependencies from pyproject.toml"
    pip install ".[dev]" 2>/dev/null || pip install . 2>/dev/null || true
    return
  fi

  echo "No requirements.txt or pyproject.toml found — skipping dep install"
}

run_scan() {
  local target="${1:?target is required}"
  local sarif_file="${2:-results.sarif}"
  local llm_judge="${3:-false}"

  SCAN_ARGS=(--sarif "$sarif_file")

  if [ "$llm_judge" = "true" ]; then
    SCAN_ARGS+=(--llm-judge)
  fi

  echo "Running: checkagent scan $target ${SCAN_ARGS[*]}"
  checkagent scan "$target" "${SCAN_ARGS[@]}"
}

case "$COMMAND" in
  install)
    install_deps "${2:-requirements.txt}"
    ;;
  scan)
    run_scan "${2:-}" "${3:-results.sarif}" "${4:-false}"
    ;;
  *)
    echo "Usage: entrypoint.sh <install|scan> [args...]" >&2
    exit 1
    ;;
esac

#!/usr/bin/env bash
# =============================================================================
# entrypoint.sh – Hyperion HALO container entrypoint
# =============================================================================
set -euo pipefail

COMMAND="${1:-serve}"

log() {
    echo "[entrypoint] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"
}

# ---------------------------------------------------------------------------
# GPU sanity check
# ---------------------------------------------------------------------------
check_gpu() {
    if ! nvidia-smi > /dev/null 2>&1; then
        log "WARNING: nvidia-smi not available – running in CPU-only mode"
        return
    fi
    log "GPU info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
        | while read -r line; do log "  $line"; done
}

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
preflight() {
    log "Python: $(python --version)"
    log "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
    check_gpu
    log "Config: ${HYPERION_CONFIG:-/workspace/config/hyperion_config.yaml}"
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
cmd_serve() {
    preflight
    log "Starting Hyperion HALO inference server..."
    exec python /workspace/main.py serve \
        --config "${HYPERION_CONFIG:-/workspace/config/hyperion_config.yaml}" \
        --port "${HYPERION_PORT:-8000}" \
        "$@"
}

cmd_benchmark() {
    preflight
    log "Running benchmarks..."
    exec python /workspace/main.py benchmark "$@"
}

cmd_test() {
    log "Running tests..."
    exec python -m pytest /workspace/tests/ -v "$@"
}

cmd_shell() {
    log "Starting interactive shell..."
    exec bash
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$COMMAND" in
    serve)      cmd_serve      "${@:2}" ;;
    benchmark)  cmd_benchmark  "${@:2}" ;;
    test)       cmd_test       "${@:2}" ;;
    shell)      cmd_shell ;;
    *)
        log "Unknown command: $COMMAND"
        log "Usage: entrypoint.sh [serve|benchmark|test|shell]"
        exit 1
        ;;
esac

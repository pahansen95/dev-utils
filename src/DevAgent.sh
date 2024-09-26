#!/usr/bin/env bash

set -eEou pipefail

log() { printf '%b\n' "$@"; }

log "INFO: Argv = $@"

declare CONTEXT="${BASH_SOURCE[0]}"; CONTEXT="$(realpath --strip "${CONTEXT%/*}/..")"
### Context is the Working Tree Root ###

command -v direnv &>/dev/null || {
  log 'ERROR: can not locate `direnv` on path'
  exit 1
}

pushd "${CONTEXT}" &>/dev/null || {
  log "DevAgent Directory does not exist"
  exit 1
}
source "${CONTEXT}/.user.env" || {
  log "ERROR: Failed to source the DevAgent Configs"
  exit 1
}
source "${CONTEXT}/.venv/bin/activate"
log "INFO: $(command -v python)"
log "INFO: $(python3 --version)"
PYTHONPATH="${CONTEXT}/src:${PYTHONPATH:-}" \
  python3 -m $@
log "fin"

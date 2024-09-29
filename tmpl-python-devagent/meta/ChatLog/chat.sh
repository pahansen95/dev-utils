#!/usr/bin/env bash

log() { printf 'LOG::%b\n' "$*" >&2; }

command -v 'DevAgent.sh' &> /dev/null || {
  log "DevAgent.sh not found in PATH"
  return 1
}

parse_chat_name() {
  local name="${1:?Missing Chat Name}"
  ### Split name by '/' & '.*'
  name="${name##*/}"
  name="${name%%.*}"
  ### Names can only have [a-zA-Z0-9_-] chars
  name="${name//[^a-zA-Z0-9_-]/}"
  printf '%s' "${name}"
}

declare chat_name; chat_name="$(parse_chat_name "${1:-"$(date '+%Y-%m-%d')"}")"
log "Chat Name: ${chat_name}"
DevAgent.sh \
  "${WORK_DIR}/meta/Context/${1:?Missing Context Name}.py" \
  "${WORK_DIR}/meta/ChatLog/${chat_name}.md"

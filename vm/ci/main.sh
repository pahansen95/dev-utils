#!/usr/bin/env bash

### Package Setup ###

# Check we were sourced
(return 0 &>/dev/null) || {
  printf 'FATAL: Source Me: %s' "${BASH_SOURCE[0]}" >&2
  exit 1
}

# Check for dependencies
command -v cloud-localds &>/dev/null || { log 'MISSING DEPENDENCY: cloud-localds'; return 1; }

declare -A CI_PKG_CFG=(
  [name]='ci'
  [context]="$(readlink -f "${BASH_SOURCE[0]%/*}")"
)

### Package Interface ###
: <<'DOC'
Renders the Cloud-Init Datastore Content into the specified Workdir.
3 Files will be rendered:

- [WORKDIR/]userdata
- [WORKDIR/]metadata
- [WORKDIR/]network
DOC
ci_render_datastore() {
  local -A fn_kwargs=(
    [tmpl]='' # What Template to use
    [workdir]='' # Where to store the rendered files
    [cfg]='' # The Config to use
  ); parse_kv fn_kwargs "$@"
  for key in "${!fn_kwargs[@]}"; do
    local val="${fn_kwargs[${key}]}"
    case "${key}" in
      tmpl)
        [[ $val =~ ^debian$ ]] || { log "Unsupported Template: ${val}"; return 1; }
        local userdata_tmpl="${CI_PKG_CFG[context]}/tmpl/${val}/userdata.tmpl.sh"
        local network_tmpl="${CI_PKG_CFG[context]}/tmpl/${val}/network.tmpl.sh"
        local metadata_tmp="${CI_PKG_CFG[context]}/tmpl/metadata.tmpl.sh"
        [[ -f "${userdata_tmpl}" && -f "${network_tmpl}" && -f "${metadata_tmpl}" ]] || {
          log "Missing one of the required CloudInit templates...\n  ${userdata_tmpl}\n  ${network_tmpl}\n  ${metadata_tmpl}"
          return 1;
        }
        ;;
      workdir)
        [[ -d "${val}" ]] || { log "Missing Workdir: ${val}"; return 1; }
        ;;
      cfg)
        log "Using Cloud Init Config...\n$(jq . <<< "${val}")"
        local -a userdata_args=(
          "$(jq -cr '.username' <<< "${val}")"
          "$(jq -cr '.hostname' <<< "${val}")"
          "$(jq -c '.ssh_keys_json' <<< "${val}")"
        ) || { log "Failed to parse CI Userdata Config Values"; return 1; }
        local -a network_args=(
          "$(jq -cr '.hwaddr' <<< "${val}")"
        ) || { log "Failed to parse CI Network Config Values"; return 1; }
        local -a metadata_args=(
          "$(jq -cr '.id' <<< "${val}")"
          "$(jq -cr '.hostname' <<< "${val}")"
        ) || { log "Failed to parse CI Metadata Config Values"; return 1; }
        ;;
      *) log "Unknown Function kwarg: ${key}"; return 1 ;;
    esac
  done
  ### Render the Cloud-Init Datastore Content
  bash "${userdata_tmpl}" "${userdata_args[@]}" > "${fn_kwarg[workdir]}/userdata"
  bash "${network_tmpl}" "${network_args[@]}" > "${fn_kwarg[workdir]}/network"
  bash "${metadata_tmpl}" "${metadata_args[@]}" > "${fn_kwarg[workdir]}/metadata"
  [[ -f "${fn_kwarg[workdir]}/userdata" && -f "${fn_kwarg[workdir]}/network" && -f "${fn_kwarg[workdir]}/metadata" ]] || {
    log "Failed to render Cloud-Init Datastore Content"
    return 1
  }
}

: <<'DOC'
Assemble the Datastore into a Cloud-Init ISO Image at the specified path.
Pass the individual Cloud-Init Files to this function. (See ci_render_datastore for filepaths)
DOC
ci_assemble_datastore() {
  local -A fn_kwargs=(
    [userdata]=''
    [network]=''
    [metadata]=''
    [datastore]=''
  ); parse_kv fn_kwargs "$@"
  for key in "${!fn_kwargs[@]}"; do
    local val="${fn_kwargs[${key}]}"
    case "${key}" in
      userdata|network|metadata)
        [[ -f "${val}" ]] || { log "Missing ${key} File: ${val}"; return 1; }
        ;;
      datastore)
        [[ -f "${val}" ]] && { log "Datastore Already Exists: ${val}"; return 1; }
        ;;
      *) log "Unknown Function kwarg: ${key}"; return 1 ;;
    esac
  done

  cloud-localds \
    --network-config="${fn_kwarg[network]}" \
    "${fn_kwarg[datastore]}" \
    "${fn_kwarg[userdata]}" "${fn_kwarg[metadata]}"

  [[ -f "${fn_kwarg[datastore]}" ]] || {
    log "Failed to create Cloud-Init Datastore: ${fn_kwarg[datastore]}"
    return 1
  }
}
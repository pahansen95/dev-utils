#!/usr/bin/env bash

# Check we were sourced
(return 0 &>/dev/null) || {
  printf 'FATAL: Source Me: %s' "${BASH_SOURCE[0]}" >&2
  exit 1
}

### Static Globals ###
declare -r _qemu_vm_ci_uuid='4f287577-77cf-4386-ba2a-9b843dbab5c4'

_join_arr() {
  local _sep="${1:?Missing separator}"; shift
  local -n _fn_nr_arr="${1:?Missing array nameref argument}"; shift
  for (( idx=0; idx<${#_fn_nr_arr[@]}; idx++ )); do
    if [[ idx -lt $(( ${#_fn_nr_arr[@]} - 1 )) ]]; then
      printf -- "%s%s" "${_fn_nr_arr[${idx}]}" "${_sep}"
    else
      printf -- "%s" "${_fn_nr_arr[${idx}]}"
    fi
  done
}

### Block Device Functions ###

render_blkdev_file_driver_opts() {
  local -A _fn_kwargs=(
    ### The Following are common to all Block Devices
    [read_only]=false
    [cache_direct]=true
    [cache_noflush]=false
    [discard]=false
    [detect_zeros]=true
    ### The Following are File only Options
    [aio]=threads # One of: threads | native | io_uring
  ); parse_kv _fn_kwargs "$@"
  local -a _opt_lines=()
  for key in "${!_fn_kwargs[@]}"; do
    local val="${_fn_kwargs[${key}]}"
    case "${key}" in
      read_only|cache_direct|cache_noflush|discard|detect_zeros)
        [[ $val =~ ^(true|false)$ ]] || { log "Invalid Boolean Value: ${val}"; return 1; }
        [[ "${key}" == "read_only" ]] && key="read-only"
        [[ "${key}" == "cache_direct" ]] && key="cache.direct"
        [[ "${key}" == "cache_noflush" ]] && key="cache.noflush"
        [[ "${key}" == "detect_zeros" ]] && key="detect-zeroes"
        eval "${val}" && val="on"
        eval "${val}" || val="off"
        ;;
      aio) [[ $val =~ ^(threads|native|io_uring)$ ]] || { log "Invalid AIO Value: ${val}; expected one of: threads | native | io_uring"; return 1; } ;;
      *) log "Unknown File Driver Option: ${key}"; return 1 ;;
    esac
    _opt_lines+=( "${key}=${val}" )
  done
  _join_arr , _opt_lines
}

render_blkdev_raw_driver_opts() {
  local -A _fn_kwargs=(
    ### The Following are common to all Block Devices
    [read_only]=false
    [cache_direct]=true
    [cache_noflush]=false
    [discard]=false
    [detect_zeros]=true
    ### The Following are Raw only Options
  ); parse_kv _fn_kwargs "$@"
  local -a _opt_lines=()
  for key in "${!_fn_kwargs[@]}"; do
    local val="${_fn_kwargs[${key}]}"
    case "${key}" in
      read_only|cache_direct|cache_noflush|discard|detect_zeros)
        [[ $val =~ ^(true|false)$ ]] || { log "Invalid Boolean Value: ${val}"; return 1; }
        [[ "${key}" == "read_only" ]] && key="read-only"
        [[ "${key}" == "cache_direct" ]] && key="cache.direct"
        [[ "${key}" == "cache_noflush" ]] && key="cache.noflush"
        [[ "${key}" == "detect_zeros" ]] && key="detect-zeroes"
        eval "${val}" && val="on"
        eval "${val}" || val="off"
        ;;
      *) log "Unknown Raw Driver Option: ${key}"; return 1 ;;
    esac
    _opt_lines+=( "${key}=${val}" )
  done
  _join_arr , _opt_lines
}

render_blkdev_qcow2_driver_opts() {
  local -A _fn_kwargs=(
    ### The Following are common to all Block Devices
    [read-only]=false
    [cache_direct]=true
    [cache_noflush]=false
    [discard]=false
    [detect_zeros]=true
    ### The Following are QCoW2 only Options
    [backing]="" # A Filepath or 'null' to disable backing. Omit to use the default backing file if any.    
  ); parse_kv _fn_kwargs "$@"
  local -a _opt_lines=()
  for key in "${!_fn_kwargs[@]}"; do
    local val="${_fn_kwargs[${key}]}"
    case "${key}" in
      read_only|cache_direct|cache_noflush|discard|detect_zeros)
        [[ $val =~ ^(true|false)$ ]] || { log "Invalid Boolean Value: ${val}"; return 1; }
        [[ "${key}" == "read_only" ]] && key="read-only"
        [[ "${key}" == "cache_direct" ]] && key="cache.direct"
        [[ "${key}" == "cache_noflush" ]] && key="cache.noflush"
        [[ "${key}" == "detect_zeros" ]] && key="detect-zeroes"
        eval "${val}" && val="on"
        eval "${val}" || val="off"
        ;;
      backing)
        [[ -z "${val:-}"]] && continue # Backing is Optional
        [[ "${val}" != 'null' && ! -f "${val}" ]] || { log "Invalid Backing File: ${val}"; return 1; }
      *) log "Unknown QCow2 Driver Option: ${key}"; return 1 ;;
    esac
    _opt_lines+=( "${key}=${val}" )
  done
  _join_arr , _opt_lines
}

render_blkdev_opts() {
  local _blkdev_type="${1:?Missing blkdev type}"; shift
  local _opts_json="${1:?Missing blkdev opts}"; shift
  mapfile -t "$(jq -rc 'to_entries[] | "\(.key)=\(.value)"' <<< "${_opts_json}")"
  case "${_blkdev_type}" in
    file) render_blkdev_file_driver_opts "${MAPFILE[@]}" ;;
    raw) render_blkdev_raw_driver_opts "${MAPFILE[@]}" ;;
    qcow2) render_blkdev_qcow2_driver_opts "${MAPFILE[@]}" ;;
    *) log "Block Device Type Doesn't support Options: ${_blkdev_type}"; return 1 ;;
  esac
}

### Network Device Functions ###

render_netdev_tap_opts() {
  local -A _fn_kwargs=(
    [ifname]=
    [script]=false
    [downscript]=false
  ); parse_kv _fn_kwargs "$@"
  local -a _opt_lines=()
  for key in "${!_fn_kwargs[@]}"; do
    local val="${_fn_kwargs[${key}]}"
    case "${key}" in
      ifname) [[ -n "${val}" ]] || { log "Missing Interface Name"; return 1; } ;;
      script|downscript)
        [[ $val =~ ^(true|false)$ ]] || { log "Invalid Boolean Value: ${val}"; return 1; }
        eval "${val}" && { log "ERROR: ${key} is not supported in 'on' mode"; return 1;}
        eval "${val}" || val="off"
        ;;
      *) log "Unknown Tap Option: ${key}"; return 1 ;;
    esac
    _opt_lines+=( "${key}=${val}" )
  done
  _join_arr , _opt_lines
}

render_netdev_user_opts() {
  local -A _fn_kwargs=(
    [ipv4]=false
    [ipv4_net]=
    [ipv4_host]=
    [ipv4_dns]=
    [ipv6]=false
    [ipv6_net]=
    [ipv6_host]=
    [ipv6_dns]=
    [restrict]=false
    [port_forward]="" # JSON Array of Port Forwarding Configs
  ); parse_kv _fn_kwargs "$@"
  local -a _opt_lines=()
  for key in "${!_fn_kwargs[@]}"; do
    local val="${_fn_kwargs[${key}]}"
    case "${key}" in
      ipv4|ipv6|restrict)
        [[ $val =~ ^(true|false)$ ]] || { log "Invalid Boolean Value: ${val}"; return 1; }
        eval "${val}" && val="on"
        eval "${val}" || val="off"
        ;;
      port_forward)
        key="hostfwd"
        [[ -n "${val:-}" ]] || { log "Missing Port Forwarding Config"; return 1; }
        mapfile -t < <(jq -c '.[]' <<< "${val}")
        val="" # Reset the Value
        for (( idx=0; idx<=${#MAPFILE[@]}; idx++ )); do
          log "Parsing Port Forwarding Config #$(( idx + 1 ))"
          local cfg="${MAPFILE[${idx}]}"
          local proto; proto="$(jq -r '.proto' <<< "${cfg}")"
          [[ $proto =~ ^(tcp|udp)$ ]] || { log "Invalid Port Forwarding Protocol: ${proto}"; return 1; }
          val+="${proto}:"
          local host_addr; host_addr="$(jq -r '.host.addr // ""' <<< "${cfg}")"
          [[ -n "${host_addr:-}" ]] && val+="${host_addr}:"
          local host_port; host_port="$(jq -r '.host.port // ""' <<< "${cfg}")"
          [[ -n "${host_port:-}" ]] || { log "Missing Host Port"; return 1; }
          val+="${host_port}-"
          local guest_addr; guest_addr="$(jq -r '.guest.addr // ""' <<< "${cfg}")"
          [[ -n "${guest_addr:-}" ]] && val+="${guest_addr}:"
          local guest_port; guest_port="$(jq -r '.guest.port // ""' <<< "${cfg}")"
          [[ -n "${guest_port:-}" ]] || { log "Missing Guest Port"; return 1; }
          val+="${guest_port}"
          _opt_lines+=( "${key}=${val}" ) # Append here
        done
        continue # Short Circuit since we handle repeating Port Forward Lines differently
        ;;
      ipv[46]_*)
        [[ $key =~ ^ipv([46])_(.+) ]] && {
          [[ "${BASH_REMATCH[1]}" -eq 6 ]] && key="ipv6-${BASH_REMATCH[2]}"
          [[ "${BASH_REMATCH[1]}" -eq 4 ]] && key="${BASH_REMATCH[2]}"
        }
        ;;
      *) log "Unknown User Option: ${key}"; return 1 ;;
    esac
    _opt_lines+=( "${key}=${val}" )
  done
  _join_arr , _opt_lines
  ### Some Soft Sanity Checks to help w/ debugging ###
  [[ ${_fn_kwargs[ipv4]} =~ ^on$ ]] && {
    # Warn if the IPv4 Configs look malformed
    [[ -z "${_fn_kwargs[net]:-}" ]] || {
      [[ ${_fn_kwargs[net]} =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/[0-9]+$ ]] || { log "WARNING! Invalid IPv4 Network: ${_fn_kwargs[net]}"; }
    }
    [[ -z "${_fn_kwargs[host]:-}" ]] || {
      [[ ${_fn_kwargs[host]} =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || { log "WARNING! Invalid IPv4 Host: ${_fn_kwargs[host]}"; }
    }
    [[ -z "${_fn_kwargs[dns]:-}" ]] || {
      [[ ${_fn_kwargs[dns]} =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || { log "WARNING! Invalid IPv4 DNS: ${_fn_kwargs[dns]}"; }
    }
  }
  [[ ${_fn_kwargs[ipv4]} =~ ^off$ ]] && {
    # Warn if there are IPv4 Configs
    [[ -z "${_fn_kwargs[ipv4_net]:-}" && -z "${_fn_kwargs[ipv4_host]:-}" && -z "${_fn_kwargs[dns]:-}" ]] || {
      log "WARNING! IPv4 Configs are present but IPv4 is disabled"
    }
  }
  [[ ${_fn_kwargs[ipv6]} =~ ^on$ ]] && {
    # Warn if the IPv6 Configs look malformed
    [[ -z "${_fn_kwargs[ipv6_net]:-}" ]] || {
      [[ ${_fn_kwargs[ipv6_net]} =~ ^[0-9a-fA-F:]+/[0-9]+$ ]] || { log "WARNING! Invalid IPv6 Network: ${_fn_kwargs[ipv6_net]}"; }
    }
    [[ -z "${_fn_kwargs[ipv6_host]:-}" ]] || {
      [[ ${_fn_kwargs[ipv6_host]} =~ ^[0-9a-fA-F:]+$ ]] || { log "WARNING! Invalid IPv6 Host: ${_fn_kwargs[ipv6_host]}"; }
    }
    [[ -z "${_fn_kwargs[ipv6_dns]:-}" ]] || {
      [[ ${_fn_kwargs[ipv6_dns]} =~ ^[0-9a-fA-F:]+$ ]] || { log "WARNING! Invalid IPv6 DNS: ${_fn_kwargs[ipv6_dns]}"; }
    }
  }
  [[ ${_fn_kwargs[ipv6]} =~ ^off$ ]] && {
    # Warn if there are IPv6 Configs
    [[ -z "${_fn_kwargs[ipv6_net]:-}" && -z "${_fn_kwargs[ipv6_host]:-}" && -z "${_fn_kwargs[ipv6_dns]:-}" ]] || {
      log "WARNING! IPv6 Configs are present but IPv6 is disabled"
    }
  }
}

render_netdev_opts() {
  local _netdev_type="${1:?Missing netdev type}"; shift
  local _opts_json="${1:?Missing netdev opts}"; shift
  mapfile -t "$(jq -rc 'to_entries[] | "\(.key)=\(.value)"' <<< "${_opts_json}")"
  case "${_netdev_type}" in
    tap) render_netdev_tap_opts "${MAPFILE[@]}" ;;
    user) render_netdev_user_opts "${MAPFILE[@]}" ;;
    *) log "Net Device Type Doesn't support Options: ${_netdev_type}"; return 1 ;;
  esac
}

### VM Functions ###

run_vm() {
  local -A fn_kwargs=(
    [workdir]=
    [exec]=false
    [qemu_system]=
    [hostname]=
    [id]=
    [machine_cfg]='{"type":"microvm","accel":"kvm"}'
    [cpu]=2
    [mem]=4096
    [debug]=false
    [serial_cfg]='{"type":"telnet","conn":"127.0.0.1:56789}'
    [root_vol_cfg]='{"file":"/path/to/root.img","uuid":"00000000-0000-0000-0000-000000000000","fmt":"qcow2","opts":{"file":{},"qcow2":{}}}' # Note `fmt` can be any Supported Format listed in qemu-img --help
    [extra_vol_cfg]='[{"file":"/path/to/extra1.img","uuid":"11111111-1111-1111-1111-111111111111","fmt":"raw","opts":{"file":{},"raw":{}}}]' # Note: Opts are 
    [ci_datastore]='{"file":"/path/to/ci-datastore.iso","fmt":"file"}'
    [network_cfg]="$(
      jq -cn '[
        {
          "hwaddr": "00:11:22:33:44:55",
          "mode": "tap",
          "opts": {
            "ifname": "tap0"
          }
        },
        {
          "hwaddr": "11:22:33:44:55:66",
          "mode": "user",
          "opts": {
            "ipv4": true,
            "net": "162.254.0.10/24",
            "host": "162.254.0.2",
            "dns": "162.254.0.3",
            "ipv6": true,
            "ipv6_net": "",
            "ipv6_host": "",
            "ipv6_dns": "",
            "port_forward": [
              {
                "proto": "tcp",
                "host": {
                  "addr": "127.0.0.1",
                  "port": "50022"
                },
                "guest": {
                  "addr": "162.254.0.2",
                  "port": "22"
                }
              },                
              {
                "proto": "udp",
                "host": {
                  "addr": "127.0.0.1",
                  "port": "51820"
                },
                "guest": {
                  "addr": "162.254.0.2",
                  "port": "51820"
                }
              },                
            ]
          }
        }
      ]'
    )"
    [kernel_cfg]='{"kernel":"/path/to/vmlinuz","initrd":"/path/to/initrd","opts":["root=LABEL=ROOT","console=ttyS0"]}'
  ); parse_kv fn_kwargs "$@"
  local -A volumes=() netinfs=()
  local -i extra_volumes_count=0 volumes_count=0 netinfs_count=0
  for key in "${!fn_kwargs[@]}"; do
    local val="${fn_kwargs[${key}]}"
    case "${key}" in
      qemu_system)
        [[ -n "${val:-}" ]] || { log "Missing QEMU System"; return 1; }
        command -v "${val}" &>/dev/null || { log "MISSING DEPENDENCY: ${val}"; return 1; }
        ;;
      machine_cfg)
        [[ -n "${val:-}" ]] || { log "Missing Machine Config"; return 1; }
        mapfile -t < <(jq -r 'to_entries[] | "\(.key)=\(.value)"' <<< "${val}")
        local machine_opts_csv; machine_opts_csv="$(_join_arr , MAPFILE)" || { log "Failed to parse Machine Config"; return 1; }
        ;;
      serial_cfg)
        [[ -n "${val:-}" ]] || { log "Missing Serial Config"; return 1; }
        local serial_type; serial_type="$(jq -r '.type' <<< "${val}")"
        [[ $serial_type =~ ^telnet|stdio$ ]] || { log "Invalid Serial Type: ${serial_type}"; return 1; }
        ;;
      root_vol_cfg)
        [[ -n "${val:-}" ]] || { log "Missing Root Block Device"; return 1; }
        local -A volumes=(
          [file_0]="$(jq -r '.file' <<< "${val}")"
          [uuid_0]="$(jq -r '.uuid' <<< "${val}")"
          [fmt_0]="$(jq -r '.fmt' <<< "${val}")"
        )
        local _fmt="${volumes["fmt_0"]}"
        volumes["opts_file_0"]="$( render_blkdev_opts file "$(jq -cr '.opts.file // {}' <<< "${val}")" )"
        [[ "${_fmt}" == file ]] || volumes["opts_${_fmt}_0"]="$( render_blkdev_opts "${_fmt}" "$(jq -cr --arg fmt "${_fmt}" '.opts[$fmt] // {}' <<< "${val}")" )"
        [[ -f "${volumes["file_0"]}" ]] || { log "Missing Root Device: ${volumes["file_0"]}"; return 1; }
        [[ ${volumes["uuid_0"],,} =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] || { log "Invalid Root UUID: ${volumes["uuid_0"]}"; return 1; }
        [[ ${volumes["uuid_0"],,} != "${_qemu_vm_ci_uuid}" ]] || { log "Requested UUID is reserved for the Cloud-Init Datastore: ${_qemu_vm_ci_uuid}"; return 1; }
        volumes_count="$(( volumes_count + 1 ))"
        ;;
      extra_vol_cfg)
        [[ -n "${val:-}" ]] || { log "Missing Extra Block Devices"; return 1; }
        mapfile -t < <(jq -c '.[]' <<< "${val}")
        extra_volumes_count="${#MAPFILE[@]}"
        for (( idx=1; idx<=${extra_volumes_count}; idx++ )); do
          local blk_cfg="${MAPFILE[${idx}]}"
          volumes+=(
            ["file_${idx}"]="$(jq -r '.file' <<< "${blk_cfg}")"
            ["uuid_${idx}"]="$(jq -r '.uuid' <<< "${blk_cfg}")"
            ["fmt_${idx}"]="$(jq -r '.fmt' <<< "${blk_cfg}")"
          )
          local _fmt="${volumes["fmt_${idx}"]}"
          volumes["opts_file_${idx}"]="$( render_blkdev_opts file "$(jq -cr '.opts.file // {}' <<< "${blk_cfg}")" )"
          [[ "${_fmt}" == file ]] || volumes["opts_${_fmt}_${idx}"]="$( render_blkdev_opts "${_fmt}" "$(jq -cr --arg fmt "${_fmt}" '.opts[$fmt] // {}' <<< "${blk_cfg}")" )"
          [[ -f "${volumes["file_${idx}"]}" ]] || { log "Missing Extra Block Device: ${volumes["file_${idx}"]}"; return 1; }
          [[ ${volumes["uuid_${idx}"],,} =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] || { log "Invalid Extra UUID: ${volumes["uuid_${idx}"]}"; return 1; }
          [[ ${volumes["uuid_${idx}"],,} != "${_qemu_vm_ci_uuid}" ]] || { log "Requested UUID is reserved for the Cloud-Init Datastore: ${_qemu_vm_ci_uuid}"; return 1; }
        done
        volumes_count="$(( extra_volumes_count + 1 ))"
        ;;
      ci_datastore)
        [[ -n "${val:-}" ]] || { log "Missing Cloud Init Datastore"; return 1; }
        # Defer Appending the Cloud Init Volume to the volumes config
        local ci_file; ci_file="$(jq -r '.file' <<< "${val}")"
        local ci_fmt; ci_fmt="$(jq -r '.fmt' <<< "${val}")"
        ;;
      network_cfg)
        [[ -n "${val:-}" ]] || { log "Missing Network Config"; return 1; }
        mapfile -t < <(jq -c '.[]' <<< "${val}")
        netinfs_count="${#MAPFILE[@]}"
        for (( idx=1; idx<=${netinfs_count}; idx++ )); do
          local net_cfg="${MAPFILE[${idx}]}"
          netinfs+=(
            ["hwaddr_${idx}"]="$(jq -r '.hwaddr' <<< "${net_cfg}")"
            ["mode_${idx}"]="$(jq -r '.mode' <<< "${net_cfg}")"
          )
          local _mode="${netinfs["mode_${idx}"]}"
          netinfs["opts_${_mode}_${idx}"]="$( render_netdev_opts "${_mode}" "$(jq -cr '.opts // {}' <<< "${net_cfg}")" )"
          [[ ${netinfs["hwaddr_${idx}"]} =~ ^([0-9a-f]{2}:){5}[0-9a-f]{2}$ ]] || { log "Invalid MAC Address: ${netinfs["hwaddr_${idx}"]}"; return 1; }
          [[ ${netinfs["mode_${idx}"]} =~ ^(tap|user)$ ]] || { log "Invalid Network Mode: ${netinfs["mode_${idx}"]}"; return 1; }
        done
        ;;
      kernel_cfg)
        [[ -n "${val:-}" ]] || { log "Missing Kernel Config"; return 1; }
        local kernel_file; kernel_file="$(jq -r '.kernel' <<< "${val}")" || { log "Failed to parse Kernel Config"; return 1; }
        [[ -f "${kernel_file}" ]] || { log "Missing Kernel File: ${kernel_file}"; return 1; }
        local initrd_file; initrd_file="$(jq -r '.initrd' <<< "${val}")" || { log "Failed to parse Initrd Config"; return 1; }
        [[ -f "${initrd_file}" ]] || { log "Missing Initrd File: ${initrd_file}"; return 1; }
        local -a kernel_opts; kernel_opts=( "$(jq -c '.opts[] // ""' <<< "${val}")" ) || { log "Failed to parse Kernel Options"; return 1; }
        local kernel_cmdline; kernel_cmdline="$(_join_arr " " kernel_opts)"
        local boot_from_kernel=true
        ;;
      debug) [[ $val =~ ^(true|false)$ ]] || { log "Invalid Debug Value: ${val}"; return 1; } ;;
      exec) [[ $val =~ ^(true|false)$ ]] || { log "Invalid Exec Value: ${val}"; return 1; } ;;
      workdir)
        [[ -n "${val:-}" ]] || { log "Missing VM Config: ${key}"; return 1; }
        [[ -d "${val}" ]] || { log "Workdir doesn't exist: ${val}"; return 1; }
        ;;
      hostname|id|cpu|mem) [[ -n "${val:-}" ]] || { log "Missing VM Config: ${key}"; return 1; } ;;
      *) log "Unknown Function kwarg: ${key}"; return 1 ;;
    esac
  done
  ### Deferred ###
  # Append the Cloud-Init Volume to the volumes config
  [[ -n "${ci_file:-}" ]] && {
    local -i ci_idx="$(( extra_volumes_count + 1 ))"
    volumes+=(
      ["file_${ci_idx}"]="${ci_file}"
      ["uuid_${ci_idx}"]="${_qemu_vm_ci_uuid}"
      ["fmt_${ci_idx}"]="${ci_fmt}"
    )
    local _fmt="${volumes["fmt_${ci_idx}"]}"
    volumes["opts_file_${ci_idx}"]="$( render_blkdev_opts file '{"read_only":true}' )"
    [[ "${_fmt}" == 'file' ]] || volumes["opts_${_fmt}_${ci_idx}"]="$( render_blkdev_opts "${_fmt}" '{"read_only":true}' )"
    [[ -f "${volumes["file_${ci_idx}"]}" ]] || { log "Missing Cloud-Init Datastore: ${volumes["file_${ci_idx}"]}"; return 1; }
    [[ ${volumes["uuid_${ci_idx}"]} =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] || { log "Invalid Cloud-Init UUID: ${volumes["uuid_${ci_idx}"]}"; return 1; }
    volumes_count="$(( volumes_count + 1 ))"
  }

  ### Setup Qemu Args ###

  local -a qemu_argv=(
    -name "guest-${fn_kwargs[hostname]}-${fn_kwargs[id]}"
    -machine "${machine_opts_csv}"
    -cpu host
    -smp "cpus=${fn_kwargs[cpu]}"
    -m "size=${fn_kwargs[mem]}"
    -display "none"
  )

  ### Setup the Serial Console Options
  case "${serial_type}" in
    stdio)
      eval "${fn_kwarg[exec]}" || { log "ERROR: Cannot use a stdio Serial Console in VM Forking Mode"; return 1; }
      qemu_argv+=( -serial "stdio" )
      ;;
    telnet)
      eval "${fn_kwargs[exec]}" && { log "ERROR: Cannot use a telnet Serial Console in VM Exec Mode"; return 1; }
      local serial_conn; serial_conn="$(jq -r '.conn' <<< "${fn_kwargs[serial_cfg]}")"
      if eval "${fn_kwargs[debug]}"; then qemu_argv+=( -serial "telnet:${serial_conn},server=on,wait=yes" ); else qemu_argv+=( -serial "telnet:${serial_conn},server=on,wait=no" ); fi
    *) log "Unsupported Serial Type: ${serial_type}"; return 1 ;;
  esac

  ### Setup the BlockDevices
  for (( idx=0; idx<${volumes_count}; idx++ )); do
    local _vol_file="${volumes["file_${idx}"]}"
    local _vol_uuid="${volumes["uuid_${idx}"]}"
    local _vol_fmt="${volumes["fmt_${idx}"]}"
    local _vol_opts_file="${volumes["opts_file_${idx}"]}"
    [[ -n "${_vol_opts_file:-}" ]] && qemu_argv+=( -blockdev "driver=file,node-name=file${idx},filename=${_vol_file},${_vol_opts_file}" )
    [[ -n "${_vol_opts_file:-}" ]] || qemu_argv+=( -blockdev "driver=file,node-name=file${idx},filename=${_vol_file}" )
    case "${_vol_fmt}" in
      file) : ;;
      qcow2|raw)
        local _vol_opts="${volumes["opts_${_vol_fmt}_${idx}"]}"
        [[ -n "${_vol_opts:-}" ]] && qemu_argv+=( -blockdev "driver=${_vol_fmt},node-name=blk${idx},file=file${idx},${_vol_opts}" )
        [[ -n "${_vol_opts:-}" ]] || qemu_argv+=( -blockdev "driver=${_vol_fmt},node-name=blk${idx},file=file${idx}" )
        ;;
      *) log "TODO: Implement Block Device Format: ${_vol_fmt}"; return 1 ;;
    esac
    qemu_argv+=( -device "virtio-blk-device,drive=blk${idx},serial=${volumes["uuid_${idx}]}" )
  done

  ### Setup the Network Device
  for (( idx=0; idx<${netinfs_count}; idx++ )); do
    local _net_hwaddr="${netinfs["hwaddr_${idx}"]}"
    local _net_mode="${netinfs["mode_${idx}"]}"
    local _net_opts="${netinfs["opts_${_net_mode}_${idx}"]}"
    # TODO: Revist these cases; should we support no options?
    case "${_net_mode}" in
      tap)
        [[ -n "${_net_opts:-}" ]] && qemu_argv+=( -netdev "tap,id=net${idx},${_net_opts}" )
        [[ -n "${_net_opts:-}" ]] || qemu_argv+=( -netdev "tap,id=net${idx}" )
        ;;
      user)
        [[ -n "${_net_opts:-}" ]] && qemu_argv+=( -netdev "user,id=net${idx},${_net_opts}" )
        [[ -n "${_net_opts:-}" ]] || qemu_argv+=( -netdev "user,id=net${idx}" )
        ;;
      *) log "TODO: Implement Network Mode: ${_net_mode}"; return 1 ;;
    esac
    qemu_argv+=( -device "virtio-net-device,netdev=net${idx},mac=${_net_hwaddr}" )
  done

  ### Misc. Options

  # Forking
  eval "${fn_kwargs[exec]}" || {
    qemu_argv+=(
      -daemonize
      -pidfile "${fn_kwargs[workdir]}/guest-${fn_kwargs[hostname]}-${fn_kwargs[id]}.pid"
    )
  }
  
  # Direct Kernel Boot
  eval "${boot_from_kernel:-false}" && {
    qemu_argv+=(
      -kernel "${kernel_file}"
      -initrd "${initrd_file}"
    )
    [[ -n "${kernel_cmdline:-}" ]] && qemu_argv+=( -append "${kernel_cmdline}" )
  }

  ### Run the VM ###
  # Exec Mode
  eval "${fn_kwargs[exec]}" && { exec sudo "${fn_kwargs[qemu_system]}" "${qemu_argv[@]}"; } # This will never return
  # Forking Mode
  sudo "${fn_kwargs[qemu_system]}" "${qemu_argv[@]}"
}

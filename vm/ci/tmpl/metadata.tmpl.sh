#!/usr/bin/env bash

### Duplicate Stdout to fd3 so we can prevent leakage of unwanted output
exec 3>&1
exec 1>&2

declare id="${1}"; shift
declare hostname="${1}"; shift

cat >&3 <<EOF
instance_id: "guest-${id}"
local_hostname: "${hostname}"
EOF

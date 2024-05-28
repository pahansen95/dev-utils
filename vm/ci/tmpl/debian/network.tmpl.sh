#!/usr/bin/env bash

### Duplicate Stdout to fd3 so we can prevent leakage of unwanted output
exec 3>&1
exec 1>&2

declare hwaddr="${1}"; shift

cat >&3 <<EOF
network:
  version: 2
  ethernets:
    uplink:
      dhcp4: true
      dhcp6: true
      optional: false
      match:
        macaddress: "${hwaddr}"
      set-name: uplink
EOF

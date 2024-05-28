#!/usr/bin/env bash

### Duplicate Stdout to fd3 so we can prevent leakage of unwanted output
exec 3>&1
exec 1>&2

declare username="${1}"; shift
declare hostname="${1}"; shift
declare ssh_keys_json="${1}"; shift

cat >&3 <<EOF
#cloud-config
power_state:
  mode: reboot
  message: "Cloud-Init has completed Initial Configurations; Rebooting..."
package_update: true
packages:
  - ssh
  - unattended-upgrades
groups:
  - remote-access
users:
  - name: debug
    lock_passwd: false
    groups: remote-access, sudo
    plain_text_passwd: debug
    sudo: "ALL=(ALL) NOPASSWD:ALL"
    shell: /bin/bash
  - name: "${username}"
    lock_passwd: false
    passwd: '*' # Disable password login
    groups: remote-access, sudo
    sudo: "ALL=(ALL) NOPASSWD:ALL"
    shell: {{ user.shell | default('/bin/sh') }}
    ssh_authorized_keys:
$(jq -r '.[]' <<< "${ssh_keys_json}" | awk '{ print "      - " $0 }')
manage_etc_hosts: true
hostname: "${hostname}"
fqdn: "${hostname}"
disable_root: true
timezone: "UTC"
write_files:
  - path: /etc/ssh/sshd_config
    owner: root:root
    permissions: "0644"
    defer: true
    content: |
      # OpenSSH Server Config File
      LogLevel ERROR
      AuthenticationMethods publickey
      AuthorizedKeysFile %h/.ssh/authorized_keys
      ClientAliveCountMax 5
      ClientAliveInterval 60
      Compression yes
      DenyGroups !remote-access
      DisableForwarding yes
      GSSAPIAuthentication no
      HostbasedAuthentication no
      KbdInteractiveAuthentication no
      KerberosAuthentication no
      PasswordAuthentication no
      PermitEmptyPasswords no
      PermitRootLogin no
      PubkeyAuthentication yes
      Subsystem sftp internal-sftp
  - path: /etc/apt/apt.conf.d/10periodic
    owner: root:root
    permissions: "0644"
    content: |
      APT::Periodic::Enable "1";
      APT::Periodic::Update-Package-Lists "1";
      APT::Periodic::Download-Upgradeable-Packages "0";
      APT::Periodic::AutocleanInterval "1";
      APT::Periodic::CleanInterval "1";
      APT::Periodic::Unattended-Upgrade "6h";
  - path: /etc/apt/apt.conf.d/50unattended-upgrades
    owner: root:root
    permissions: "0644"
    content: |
      // For full documentation of the "Unattended-Upgrade" Configuration Group please see
      // https://github.com/mvo5/unattended-upgrades/blob/5f979a25fda0f399a6c426e9972ed4c2a0e15cf0/data/50unattended-upgrades.Ubuntu
      // Custom Values
      // ...

      // Default Values
      Unattended-Upgrade::AutoFixInterruptedDpkg "true";
      Unattended-Upgrade::Allow-downgrade "false";
      Unattended-Upgrade::Allowed-Origins {
        "\${distro_id}:\${distro_codename}";
        "\${distro_id}:\${distro_codename}-security";
        "\${distro_id}:\${distro_codename}-updates";
        "\${distro_id}:\${distro_codename}-proposed";
        "\${distro_id}:\${distro_codename}-backports";
      };
      Unattended-Upgrade::Automatic-Reboot "false";
      Unattended-Upgrade::Automatic-Reboot-Time "now";
      Unattended-Upgrade::DevRelease "false";
      Unattended-Upgrade::MinimalSteps "true";
      Unattended-Upgrade::InstallOnShutdown "false";
      Unattended-Upgrade::Package-Blacklist {
      };
      Unattended-Upgrade::Remove-New-Unused-Dependencies "true";
      Unattended-Upgrade::Remove-Unused-Kernel-Packages "true";
EOF

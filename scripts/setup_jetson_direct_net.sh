#!/usr/bin/env bash
#
# setup_jetson_direct_net.sh
#
# Turn a Jetson's wired ethernet into a self-contained, plug-and-play network so a
# laptop can connect over a direct cable with ZERO host configuration — no external
# DHCP, no router, no WiFi. NetworkManager "shared" mode makes the Jetson run its own
# DHCP server and sit at a fixed 10.42.0.1; any machine on default/Automatic settings
# that plugs in gets a 10.42.0.x lease and reaches the Jetson at 10.42.0.1.
#
# This only touches the WIRED connection — the Jetson's WiFi (and its internet) is
# left completely alone, so you can still SSH in over WiFi while setting this up.
#
# Usage (on the Jetson):
#     sudo bash setup_jetson_direct_net.sh           # enable shared mode
#     sudo bash setup_jetson_direct_net.sh --reset    # revert to normal DHCP client
#
# Idempotent: safe to run repeatedly. Designed to be copied to any colleague's Jetson.

set -euo pipefail

CONN_NAME="jetson-direct"   # our dedicated NetworkManager profile

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root:  sudo bash $0 ${*:-}" >&2
  exit 1
fi

# --- find the real wired ethernet device (exclude the USB-gadget usb* bridges) ------
ETH_DEV="$(nmcli -t -f DEVICE,TYPE device status \
  | awk -F: '$2=="ethernet" && $1 !~ /^usb/ {print $1; exit}')"

if [[ -z "${ETH_DEV}" ]]; then
  echo "ERROR: no wired ethernet device found (type=ethernet, excluding usb*)." >&2
  echo "Devices seen:" >&2
  nmcli device status >&2
  exit 1
fi
echo "[*] Wired ethernet device: ${ETH_DEV}"

# --- reset mode: drop our profile, let NetworkManager fall back to normal DHCP -------
if [[ "${1:-}" == "--reset" ]]; then
  if nmcli -t -f NAME connection show | grep -qx "${CONN_NAME}"; then
    nmcli connection delete "${CONN_NAME}"
    echo "[✓] Removed '${CONN_NAME}'. The wired port is a normal DHCP client again."
  else
    echo "[i] No '${CONN_NAME}' profile present; nothing to reset."
  fi
  exit 0
fi

# --- enable shared mode --------------------------------------------------------------
# A dedicated profile with high autoconnect priority so it wins on boot over any
# auto-created "Wired connection" without us having to hunt down its name.
if nmcli -t -f NAME connection show | grep -qx "${CONN_NAME}"; then
  echo "[*] Updating existing '${CONN_NAME}' profile"
  nmcli connection modify "${CONN_NAME}" \
    connection.interface-name "${ETH_DEV}" \
    connection.autoconnect yes connection.autoconnect-priority 999 \
    ipv4.method shared ipv6.method ignore
else
  echo "[*] Creating '${CONN_NAME}' profile"
  nmcli connection add type ethernet con-name "${CONN_NAME}" ifname "${ETH_DEV}" \
    connection.autoconnect yes connection.autoconnect-priority 999 \
    ipv4.method shared ipv6.method ignore
fi

echo "[*] Activating '${CONN_NAME}' on ${ETH_DEV}"
if ! nmcli connection up "${CONN_NAME}"; then
  echo "[!] Could not bring it up now (is the ethernet cable plugged in?)." >&2
  echo "    The profile is saved and will activate when a cable is connected." >&2
fi

echo
echo "[*] Wired IPv4 now:"
ip -4 addr show "${ETH_DEV}" 2>/dev/null | sed 's/^/    /' || true

cat <<EOF

[✓] Done. This Jetson is now a self-contained network on ${ETH_DEV}:
      - serves DHCP, fixed address 10.42.0.1
      - host machines need NO configuration (just default/Automatic networking)

    From any laptop, plug a cable into this port, then:
      ssh -L 11434:localhost:11434 ${SUDO_USER:-<user>}@10.42.0.1

    To undo:  sudo bash $0 --reset
EOF

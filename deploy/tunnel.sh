#!/usr/bin/env bash
# Run the public tunnel as a service, so it survives the terminal that started it.
#
#   bash deploy/tunnel.sh start    # start it and print the public URL
#   bash deploy/tunnel.sh url      # print the current URL again
#   bash deploy/tunnel.sh status   # is it up, and is the app behind it healthy
#   bash deploy/tunnel.sh stop     # take the site offline
#   bash deploy/tunnel.sh restart  # new tunnel, NEW URL
#
# Why this exists: `cloudflared tunnel --url ...` typed into the EC2 browser terminal is a
# foreground process owned by that SSH session. Close the tab, lose the session, let it time
# out — the tunnel dies with it, and every visitor gets HTTP 530 ("origin unreachable") while
# the app itself is still running perfectly. As a systemd unit it outlives the terminal, comes
# back after a reboot, and restarts itself if it drops.
#
# The URL is random and CHANGES on every restart. That is what a free quick tunnel is; a stable
# hostname needs a Cloudflare account and a named tunnel.

set -euo pipefail

UNIT=/etc/systemd/system/kyr-tunnel.service
APP_PORT="${KYR_PORT:-8000}"

die() { echo "  ✗ $*" >&2; exit 1; }

install_unit() {
  command -v cloudflared >/dev/null || die "cloudflared is not installed. Run:
  curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /tmp/cf
  sudo install -m 755 /tmp/cf /usr/local/bin/cloudflared"

  sudo tee "$UNIT" >/dev/null <<EOF
[Unit]
Description=KnowYourRightsAI public tunnel
After=network-online.target kyr.service
Wants=network-online.target

[Service]
User=$USER
# --no-autoupdate: an update mid-demo would restart the tunnel and change the URL.
ExecStart=/usr/local/bin/cloudflared tunnel --no-autoupdate --url http://localhost:${APP_PORT}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
}

# cloudflared prints the URL once, in a banner, and never again. Read it back out of the journal
# rather than asking the operator to have kept the scrollback.
find_url() {
  sudo journalctl -u kyr-tunnel --no-pager -n 200 2>/dev/null \
    | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1
}

case "${1:-status}" in
  start)
    install_unit
    sudo systemctl enable --now kyr-tunnel >/dev/null 2>&1 || sudo systemctl restart kyr-tunnel
    echo "  waiting for the tunnel to come up…"
    for _ in $(seq 1 30); do
      URL=$(find_url || true)
      [ -n "${URL:-}" ] && break
      sleep 2
    done
    [ -n "${URL:-}" ] || die "no URL yet. Check:  sudo journalctl -u kyr-tunnel -n 40"
    echo
    echo "  ════════════════════════════════════════════════════════════"
    echo "    $URL"
    echo "  ════════════════════════════════════════════════════════════"
    echo "  It survives closing this terminal. It changes if the tunnel restarts."
    ;;
  url)
    URL=$(find_url || true)
    [ -n "${URL:-}" ] && echo "  $URL" || die "no tunnel URL found — is it running? bash deploy/tunnel.sh status"
    ;;
  stop)
    sudo systemctl disable --now kyr-tunnel >/dev/null 2>&1 || true
    echo "  ✓ tunnel stopped — the link is dead until you start it again"
    ;;
  restart)
    sudo systemctl restart kyr-tunnel
    sleep 6
    echo "  ✓ restarted — NEW url: $(find_url || echo 'not up yet, try: bash deploy/tunnel.sh url')"
    ;;
  status)
    if systemctl is-active --quiet kyr-tunnel; then
      echo "  tunnel : running   $(find_url || echo '(url not in recent journal)')"
    else
      echo "  tunnel : stopped"
    fi
    if systemctl is-active --quiet kyr; then
      READY=$(curl -fsS "http://127.0.0.1:${APP_PORT}/api/health" 2>/dev/null \
              | grep -o '"ready":[^,}]*' | head -1 || echo 'unreachable')
      echo "  app    : running   ${READY:-unknown}"
    else
      echo "  app    : stopped   —  sudo systemctl start kyr"
    fi
    ;;
  *) die "usage: $0 {start|url|status|stop|restart}" ;;
esac

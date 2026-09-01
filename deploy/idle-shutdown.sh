#!/bin/bash
# Stop the instance once nobody has asked it anything for a while.
#
# This is the other half of the wake-on-demand setup: the Lambda starts the box when someone
# opens the link, and this stops it again afterwards. Between demos you pay for the EBS volume
# and nothing else.
#
# Install on the instance:
#   sudo cp idle-shutdown.sh /usr/local/bin/ && sudo chmod +x /usr/local/bin/idle-shutdown.sh
#   ( crontab -l 2>/dev/null; echo "*/5 * * * * /usr/local/bin/idle-shutdown.sh" ) | crontab -
#
# Activity is measured by the app's own LLM call counter rather than CPU load or SSH sessions,
# because the box is *supposed* to sit idle-but-warm between questions during a demo.

set -uo pipefail

IDLE_LIMIT_SECONDS="${KYR_IDLE_LIMIT:-1800}"   # 30 minutes
GRACE_AFTER_BOOT="${KYR_BOOT_GRACE:-900}"      # never stop within 15 min of booting
HEALTH_URL="${KYR_HEALTH_URL:-http://127.0.0.1:8000/api/usage}"
STATE_FILE=/var/tmp/kyr-activity

# Don't shut down while the machine is still starting up — models take minutes to load and a
# visitor arriving at that moment would be killed mid-boot.
UPTIME_SECONDS=$(cut -d. -f1 /proc/uptime)
if [ "$UPTIME_SECONDS" -lt "$GRACE_AFTER_BOOT" ]; then
  exit 0
fi

CALLS=$(curl -fsS --max-time 5 "$HEALTH_URL" 2>/dev/null \
        | python3 -c 'import sys,json; print(json.load(sys.stdin).get("total_calls",""))' 2>/dev/null)

# If the app is unreachable, treat the machine as idle rather than leaving it running for
# ever on a crashed service — but require it to stay unreachable across several checks.
if [ -z "$CALLS" ]; then
  CALLS="unreachable"
fi

NOW=$(date +%s)
PREV_CALLS=$(cut -d' ' -f1 "$STATE_FILE" 2>/dev/null || echo "")
LAST_CHANGE=$(cut -d' ' -f2 "$STATE_FILE" 2>/dev/null || echo "$NOW")

if [ "$CALLS" != "$PREV_CALLS" ]; then
  echo "$CALLS $NOW" > "$STATE_FILE"      # something happened; reset the clock
  exit 0
fi

IDLE_FOR=$((NOW - LAST_CHANGE))
if [ "$IDLE_FOR" -ge "$IDLE_LIMIT_SECONDS" ]; then
  logger -t kyr-idle "idle for ${IDLE_FOR}s (calls=${CALLS}) — stopping the instance"
  # `shutdown -h` on an EBS-backed instance stops it rather than terminating, so the disk and
  # everything on it survives until the next wake.
  sudo shutdown -h now
fi

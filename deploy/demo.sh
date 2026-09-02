#!/usr/bin/env bash
# Control the demo from your own machine. Run it from anywhere with the AWS CLI configured.
#
#   ./deploy/demo.sh status     what is running, and what it is costing
#   ./deploy/demo.sh start      wake it up (~2 min) and print the URL
#   ./deploy/demo.sh stop       put it to sleep — billing for compute stops
#   ./deploy/demo.sh pause      sleep AND take the public link offline
#   ./deploy/demo.sh resume     put the link back and wake it
#   ./deploy/demo.sh logs       tail the server log over SSH
#
# Set these once, or export them in your shell:
INSTANCE_ID="${KYR_INSTANCE_ID:-}"
LAMBDA_NAME="${KYR_LAMBDA:-kyr-waker}"
SSH_KEY="${KYR_SSH_KEY:-$HOME/.ssh/knowyourrights.pem}"

set -uo pipefail

die() { echo "  ✗ $*" >&2; exit 1; }
usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

# Help must work before anything is configured, so the checks come after it.
case "${1:-status}" in -h|--help|help) usage ;; esac
command -v aws >/dev/null || die "The AWS CLI is not installed: https://aws.amazon.com/cli/"
[ -n "$INSTANCE_ID" ] || die "Set KYR_INSTANCE_ID (looks like i-0123456789abcdef0), or edit this file."

state() {
  aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null
}
public_ip() {
  aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null
}
wake_url() {
  aws lambda get-function-url-config --function-name "$LAMBDA_NAME" \
    --query FunctionUrl --output text 2>/dev/null
}

wait_ready() {
  # The instance reaching "running" is not the same as the app being able to answer — the
  # models take another 40-140 s. Wait for the app's own readiness signal.
  local ip deadline=$((SECONDS + 300))
  echo "  waiting for the models to load…"
  while [ $SECONDS -lt $deadline ]; do
    ip=$(public_ip)
    if [ -n "$ip" ] && [ "$ip" != "None" ]; then
      if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$SSH_KEY" \
           "ubuntu@$ip" 'curl -fsS localhost:8000/api/health' 2>/dev/null \
           | grep -q '"ready": *true'; then
        echo "  ✓ ready"
        return 0
      fi
    fi
    sleep 10
  done
  echo "  still loading — check with: $0 logs"
}

case "${1:-status}" in

  status)
    S=$(state)
    echo "  instance : $INSTANCE_ID  →  $S"
    [ "$S" = "running" ] && echo "  address  : $(public_ip)"
    U=$(wake_url); [ -n "$U" ] && echo "  wake URL : $U" || echo "  wake URL : offline (paused)"
    echo
    if [ "$S" = "running" ]; then
      echo "  costing ~\$0.034/hour while it stays up. Stop it when you are done:"
      echo "      $0 stop"
    else
      echo "  costing ~\$2.40/month for the disk. Nothing else."
    fi
    ;;

  start)
    S=$(state)
    if [ "$S" = "running" ]; then echo "  already running"; else
      echo "  starting…"
      aws ec2 start-instances --instance-ids "$INSTANCE_ID" >/dev/null
      aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
    fi
    wait_ready
    U=$(wake_url)
    echo
    echo "  Share this:  ${U:-$(public_ip)}"
    ;;

  stop)
    echo "  stopping — compute billing ends when it reaches 'stopped'…"
    aws ec2 stop-instances --instance-ids "$INSTANCE_ID" >/dev/null
    aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID"
    echo "  ✓ stopped. Disk only from here: ~\$2.40/month."
    echo "  The wake link still works — anyone opening it starts the box again."
    echo "  To prevent that too:  $0 pause"
    ;;

  pause)
    echo "  stopping the instance…"
    aws ec2 stop-instances --instance-ids "$INSTANCE_ID" >/dev/null
    aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID"
    echo "  taking the public link offline…"
    aws lambda delete-function-url-config --function-name "$LAMBDA_NAME" 2>/dev/null \
      && echo "  ✓ link removed" || echo "  (no link was configured)"
    echo
    echo "  ✓ Paused. Nothing can wake it. Cost: ~\$2.40/month for the disk."
    echo "  Bring it back with:  $0 resume"
    ;;

  resume)
    echo "  restoring the public link…"
    aws lambda create-function-url-config --function-name "$LAMBDA_NAME" \
      --auth-type NONE >/dev/null 2>&1 || true
    aws lambda add-permission --function-name "$LAMBDA_NAME" --statement-id public \
      --action lambda:InvokeFunctionUrl --principal '*' \
      --function-url-auth-type NONE >/dev/null 2>&1 || true
    echo "  starting the instance…"
    aws ec2 start-instances --instance-ids "$INSTANCE_ID" >/dev/null
    aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
    wait_ready
    echo
    echo "  Share this:  $(wake_url)"
    ;;

  logs)
    IP=$(public_ip)
    [ -n "$IP" ] && [ "$IP" != "None" ] || die "The instance is not running. Start it first."
    ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" "ubuntu@$IP" 'journalctl -u kyr -f'
    ;;

  *)
    usage
    ;;
esac

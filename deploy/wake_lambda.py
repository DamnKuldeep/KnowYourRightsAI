"""A Lambda that boots the demo on first visit, so the EC2 instance can stay stopped.

The problem with "only pay when used" on AWS is that nothing scales a 2.3 GB model to zero and
back on request. This is the practical substitute: a **Lambda Function URL** — always
reachable, free at demo volumes, no API Gateway — that starts the instance when someone opens
the link and shows a waiting page until the app answers.

Combined with the idle-shutdown timer on the instance itself, the box is only running while
somebody is actually using it. Between demos you pay for the EBS volume and nothing else.

Deploy: see DEPLOY_AWS.md. Needs `ec2:DescribeInstances` and `ec2:StartInstances` on one
instance id, and these environment variables:

    INSTANCE_ID   i-0123456789abcdef0
    APP_URL       https://demo.example.com     (where the app is reachable once up)
    REGION        ap-south-1                   (optional; defaults to the Lambda's own region)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import boto3

INSTANCE_ID = os.environ["INSTANCE_ID"]
APP_URL = os.environ.get("APP_URL", "").rstrip("/")
REGION = os.environ.get("REGION") or os.environ.get("AWS_REGION", "ap-south-1")

ec2 = boto3.client("ec2", region_name=REGION)

# The instance boots in ~40 s but the models take another 40-140 s, so the page polls the
# app's own readiness endpoint rather than trusting "instance is running".
POLL_SECONDS = 10


def instance_state() -> str:
    response = ec2.describe_instances(InstanceIds=[INSTANCE_ID])
    return response["Reservations"][0]["Instances"][0]["State"]["Name"]


def app_is_ready() -> bool:
    """The app reports ready:false while the models load — wait for the real signal."""
    if not APP_URL:
        return False
    try:
        with urllib.request.urlopen(f"{APP_URL}/api/health", timeout=4) as response:
            return bool(json.loads(response.read()).get("ready"))
    except (urllib.error.URLError, ValueError, TimeoutError, OSError):
        return False


def handler(event, context):
    state = instance_state()

    if state == "stopped":
        ec2.start_instances(InstanceIds=[INSTANCE_ID])
        return page("Starting the demo",
                    "The server was asleep to save cost. Booting it now — this takes about "
                    "two minutes, most of it loading the language models.", refresh=POLL_SECONDS)

    if state in ("pending", "stopping", "shutting-down"):
        return page("Starting the demo", f"The server is {state}. Hold on a moment.",
                    refresh=POLL_SECONDS)

    if state == "running":
        if app_is_ready():
            return {"statusCode": 302, "headers": {"Location": APP_URL or "/",
                                                   "Cache-Control": "no-store"}, "body": ""}
        return page("Almost ready",
                    "The server is up and loading the retrieval models. Nearly there.",
                    refresh=POLL_SECONDS)

    return page("Unavailable", f"The demo instance is {state}. Please try again shortly.",
                refresh=0)


def page(title: str, message: str, refresh: int) -> dict:
    meta = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — KnowYourRightsAI</title>{meta}
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0;
         min-height: 100vh; display: grid; place-items: center;
         background: #fbfaf8; color: #23201c; }}
  @media (prefers-color-scheme: dark) {{ body {{ background: #161513; color: #ece8e2; }} }}
  .card {{ max-width: 30rem; padding: 2.5rem 2rem; text-align: center; }}
  h1 {{ font-size: 1.25rem; margin: 0 0 .75rem; font-weight: 600; }}
  p {{ line-height: 1.6; color: #6c665e; margin: 0 0 1.5rem; }}
  @media (prefers-color-scheme: dark) {{ p {{ color: #a49e95; }} }}
  .bar {{ height: 3px; border-radius: 3px; background: #e3ded6; overflow: hidden; }}
  .bar i {{ display: block; height: 100%; width: 35%; background: #1f6f5c;
           animation: slide 1.4s ease-in-out infinite; }}
  @keyframes slide {{ 0% {{ transform: translateX(-100%) }}
                      100% {{ transform: translateX(340%) }} }}
  small {{ display: block; margin-top: 1.5rem; color: #97918a; font-size: .8rem; }}
</style></head>
<body><div class="card">
  <h1>⚖️ {title}</h1>
  <p>{message}</p>
  {'<div class="bar"><i></i></div>' if refresh else ''}
  <small>The demo sleeps when idle so it costs nothing to keep online.</small>
</div></body></html>"""
    return {"statusCode": 200,
            "headers": {"Content-Type": "text/html; charset=utf-8",
                        "Cache-Control": "no-store"},
            "body": html}

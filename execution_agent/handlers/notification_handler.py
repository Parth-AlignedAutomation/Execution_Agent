import json
import logging
import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart   import MIMEMultipart
from email.mime.text        import MIMEText

import requests

from execution_agent.handlers.base_handler import BaseHandler
from execution_agent.handlers.registry import registry

logger = logging.getLogger(__name__)


def _email_send(config: dict) -> str:
    recipients = config.get("to", [])
    subject     = config.get("subject", "Execution Report")
    message     = config.get("message", "Workflow completed.")
    attachments = config.get("attachments", [])
    smtp_host   = config.get("smtp_host") or os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port   = int(config.get("smtp_port") or os.getenv("SMTP_PORT", 587))
    smtp_user   = config.get("smtp_user") or os.getenv("SMTP_USER", "")
    smtp_pass   = config.get("smtp_pass") or os.getenv("SMTP_PASS", "")

    if not recipients:
        raise ValueError("Email: 'to' list is empty.")
    if not smtp_user:
        raise ValueError("Email: SMTP_USER env var is not set.")

    msg = MIMEMultipart()
    msg["From"]    = smtp_user
    msg["To"]      = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(message, "plain"))

    for path in attachments:
        if os.path.exists(path):
            with open(path, "rb") as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(path))
            part["Content-Disposition"] = f'attachment; filename="{os.path.basename(path)}"'
            msg.attach(part)

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, recipients, msg.as_string())

    return f"Email sent to {recipients}"


def _slack_send(config: dict) -> str:
    webhook_url = config.get("webhook_url") or os.getenv("SLACK_WEBHOOK_URL", "")
    message     = config.get("message", "Workflow completed.")
    username    = config.get("username", "Execution Agent")
    icon_emoji  = config.get("icon_emoji", ":robot_face:")
    channel     = config.get("channel", "")

    if not webhook_url:
        raise ValueError("Slack: 'webhook_url' or SLACK_WEBHOOK_URL is not set.")

    payload = {"text": message, "username": username, "icon_emoji": icon_emoji}
    if channel:
        payload["channel"] = channel

    response = requests.post(
        webhook_url,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    response.raise_for_status()
    return f"Slack message sent (channel: {channel or 'default'})"


def _teams_send(config: dict) -> str:
    webhook_url = config.get("webhook_url") or os.getenv("TEAMS_WEBHOOK_URL", "")
    title       = config.get("title",   "Workflow Notification")
    message     = config.get("message", "Workflow completed.")
    color       = config.get("color",   "0072C6")

    if not webhook_url:
        raise ValueError("Teams: 'webhook_url' or TEAMS_WEBHOOK_URL is not set.")

    payload = {
        "@type":      "MessageCard",
        "@context":   "http://schema.org/extensions",
        "themeColor": color,
        "summary":    title,
        "sections": [{"activityTitle": f"**{title}**", "activitySubtitle": message, "markdown": True}],
    }
    response = requests.post(
        webhook_url,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    response.raise_for_status()
    return f"Teams message sent: '{title}'"


NOTIFICATION_CHANNELS = {
    "email": {
        "send": _email_send,
    },
    "slack": {
        "send": _slack_send,
    },
    "teams": {
        "send": _teams_send,
    },
}

class NotificationHandler(BaseHandler):

    @property
    def name(self) -> str:
        return "notification"

    def execute(self, step: dict, state: dict) -> dict:
        channel_key = step.get("channel", "email").lower()

        if channel_key not in NOTIFICATION_CHANNELS:
            return {
                **state, "status": "FAILED",
                "error": (
                    f"[NotificationHandler] Unknown channel '{channel_key}'. "
                    f"Available: {list(NOTIFICATION_CHANNELS.keys())}"
                ),
            }

        channel_cfg = NOTIFICATION_CHANNELS[channel_key]

        try:
            logger.info("[NotificationHandler] Channel: %s", channel_key)
            confirmation = channel_cfg["send"](step)

            msg = f"[NotificationHandler] {confirmation}"
            logger.info(msg)

            return {
                **state,
                "logs":               state.get("logs", []) + [msg],
                "last_step_output":   msg,
                "current_step_index": state.get("current_step_index", 0) + 1,
                "error": None,
            }

        except Exception as exc:
            msg = f"[NotificationHandler] Failed: {exc}"
            logger.exception(msg)
            return {**state, "status": "FAILED", "error": msg}


# Auto-register
registry.register(NotificationHandler())
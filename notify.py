"""
notify.py — Send email notification when a stage completes.
Uses Gmail SMTP with an App Password (not your main password).

Setup (one time):
1. Go to myaccount.google.com → Security → 2-Step Verification → App passwords
2. Create app password for "Mail"
3. Set environment variable: set NOTIFY_PASSWORD=your_16char_app_password
4. Set your email:           set NOTIFY_EMAIL=your@gmail.com

Usage in main.py — add at end of each stage:
    from notify import notify
    notify("--stage train complete", "All 3 datasets trained. Run --stage unlearn next.")
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


def notify(subject: str, body: str = ""):
    email    = os.environ.get("xxx@gmail.com", "")
    password = os.environ.get("pass", "")

    if not email or not password:
        print(f"[notify] Skipped — set NOTIFY_EMAIL and NOTIFY_PASSWORD env vars")
        return

    try:
        msg = MIMEMultipart()
        msg["From"]    = email
        msg["To"]      = email
        msg["Subject"] = f"[NeurIPS Pipeline] {subject}"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_body = f"{body}\n\nTimestamp: {timestamp}\nMachine: RTX 4090"
        msg.attach(MIMEText(full_body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(email, password)
            server.send_message(msg)

        print(f"[notify] Email sent → {email}: {subject}")

    except Exception as e:
        print(f"[notify] Failed to send email: {e}")


if __name__ == "__main__":
    notify("Test notification", "If you got this, notifications are working.")
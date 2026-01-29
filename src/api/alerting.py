import os
import smtplib
from email.message import EmailMessage
from typing import Optional


def send_email_alert(subject: str, body: str) -> None:
    host = os.getenv("ALERT_SMTP_HOST")
    port = int(os.getenv("ALERT_SMTP_PORT", "587"))
    user = os.getenv("ALERT_SMTP_USER")
    password = os.getenv("ALERT_SMTP_PASSWORD")
    to_addr = os.getenv("ALERT_TO_EMAIL")
    from_addr = os.getenv("ALERT_FROM_EMAIL", user or "noreply@example.com")

    if not all([host, user, password, to_addr]):
        print("[ALERT] Email not configured. Would send:")
        print("Subject:", subject)
        print("Body:", body)
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)

    with smtplib.SMTP(host, port) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)


def alert_if_needed(
    *,
    bad_streak: int,
    predicted_label: str,
    proba_negative: Optional[float],
) -> None:
    threshold = int(os.getenv("ALERT_BAD_STREAK_THRESHOLD", "5"))
    if bad_streak < threshold:
        return

    subject = f"[Tweet Sentiment] {bad_streak} mauvaises prédictions consécutives"
    body = (
        f"Seuil atteint : {bad_streak} erreurs consécutives.\n"
        f"Dernière prédiction : {predicted_label} (proba_negative={proba_negative})\n"
        f"Action : vérifier dérive, seuil, ou régression modèle.\n"
    )
    send_email_alert(subject, body)

import smtplib
from email.mime.text import MIMEText
from config import Config
# utils.py
import re
import dns.resolver

def is_valid_email(email: str) -> bool:
    # Basic syntax check
    regex = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    if not re.match(regex, email):
        return False

    # MX record check
    domain = email.split("@")[1]
    try:
        mx_records = dns.resolver.resolve(domain, 'MX')
        if not mx_records:
            return False
    except Exception:
        return False

    return True


def send_signup_code(email: str, code: str):
    if not is_valid_email(email):
        return {"status": "error", "message": "Invalid email address."}

    subject = "Your Login Key"
    body = f"""
Hello,

Here’s your login key: {code}

Use this key whenever you want to access your account. 
If your session expires, you’ll get a new one.
"""

    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = Config.ZOHO_SENDER_EMAIL
    msg["To"] = email

    try:
        with smtplib.SMTP_SSL(Config.ZOHO_SMTP_SERVER, Config.ZOHO_SMTP_PORT) as server:
            server.login(Config.ZOHO_SENDER_EMAIL, Config.ZOHO_APP_PASSWORD)
            server.sendmail(Config.ZOHO_SENDER_EMAIL, email, msg.as_string())
        return {"status": "success", "message": f"Code sent to {email}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

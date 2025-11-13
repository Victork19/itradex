# /services/email_service.py
import logging
import re
import smtplib
from email.mime.text import MIMEText
from typing import Dict, Any, Optional
import asyncio  # Ensure imported
import threading  # NEW: For thread-local error handling if needed

from fastapi import BackgroundTasks
from config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

def is_valid_email(email: str) -> bool:
    """Validate email syntax and optional MX record."""
    # Basic syntax check
    regex = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    if not re.match(regex, email):
        return False

    # MX record check (requires dnspython: pip install dnspython)
    try:
        import dns.resolver
        domain = email.split("@")[1]
        mx_records = dns.resolver.resolve(domain, 'MX')
        if not mx_records:
            return False
    except ImportError:
        logger.warning("dnspython not installed; skipping MX check")
        return True
    except Exception:
        logger.warning(f"MX check failed for {email}; assuming valid")
        return True

    return True

async def send_email(  # CHANGED: Now async
    to_email: str,
    subject: str,
    body: str,
    background_tasks: Optional[BackgroundTasks] = None
) -> Dict[str, Any]:
    """
    Send an email via Zoho SMTP (async wrapper for sync SMTP).
    
    Args:
        to_email: Recipient email.
        subject: Email subject.
        body: Plain text body.
        background_tasks: Ignored for now (since we run blocking for status; add true async queuing later if needed).
    
    Returns:
        Dict with 'status' ('success' or 'error') and 'message'.
    """
    if not is_valid_email(to_email):
        return {"status": "error", "message": "Invalid email address."}

    if not all([settings.ZOHO_SMTP_SERVER, settings.ZOHO_SMTP_PORT, settings.ZOHO_SENDER_EMAIL, settings.ZOHO_APP_PASSWORD]):
        return {"status": "error", "message": "Email configuration missing."}

    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = settings.ZOHO_SENDER_EMAIL
    msg["To"] = to_email

    def _send_sync():
        try:
            with smtplib.SMTP_SSL(settings.ZOHO_SMTP_SERVER, settings.ZOHO_SMTP_PORT) as server:
                server.login(settings.ZOHO_SENDER_EMAIL, settings.ZOHO_APP_PASSWORD)
                server.sendmail(settings.ZOHO_SENDER_EMAIL, to_email, msg.as_string())
            logger.info(f"Email sent successfully to {to_email}: {subject}")
            return {"status": "success", "message": f"Email sent to {to_email}"}
        except Exception as e:
            logger.error(f"Email send failed to {to_email}: {str(e)}")
            return {"status": "error", "message": str(e)}

    # FIXED: Use run_in_executor to run sync code in thread without nesting loops
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _send_sync)
    
    
    return result
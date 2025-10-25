# /home/ukov/itrade/server/templates_config.py
from fastapi.templating import Jinja2Templates
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

# Initialize Jinja2Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Custom filters
def datetimeformat(value):
    if isinstance(value, datetime):
        try:
            return value.strftime("%b %-d, %Y")  # e.g., "Jan 2, 2025"
        except ValueError:
            return str(value)
    return str(value) if value else "N/A"

def format_currency(value):
    if value is None:
        return "0.00"
    try:
        return f"{float(value):,.2f}"
    except (ValueError, TypeError):
        return "0.00"

# Register filters
templates.env.filters["datetimeformat"] = datetimeformat
templates.env.filters["format_currency"] = format_currency
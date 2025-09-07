import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key")
    SQLALCHEMY_DATABASE_URI = "sqlite:///database.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET = os.getenv("JWT_SECRET", "jwt-secret")
    EMAIL_SENDER = os.getenv("EMAIL_SENDER", "youremail@example.com")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "yourpassword")
    RECAPTCHA_SECRET_KEY = os.getenv("RECAPTCHA_SECRET_KEY")
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    # Zoho SMTP
    ZOHO_SMTP_SERVER = os.getenv("ZOHO_SMTP_SERVER", "smtp.zoho.com")
    ZOHO_SMTP_PORT = int(os.getenv("ZOHO_SMTP_PORT", 465))
    ZOHO_SENDER_EMAIL = os.getenv("ZOHO_SENDER_EMAIL", "noreply@itradex.xyz")
    ZOHO_APP_PASSWORD = os.getenv("ZOHO_APP_PASSWORD")

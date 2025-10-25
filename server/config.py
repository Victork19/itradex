import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./test1.db")
    SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 60
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    REMEMBER_ME_REFRESH_TOKEN_EXPIRE_DAYS = 30
    IS_PRODUCTION = os.getenv("IS_PRODUCTION", "false").lower() == "true"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
    NOWPAYMENTS_API_KEY = os.getenv("NOWPAYMENTS_API_KEY")
    NOWPAYMENTS_IPN_SECRET = os.getenv("NOWPAYMENTS_IPN_SECRET")
    NOWPAYMENTS_BASE_URL = os.getenv("NOWPAYMENTS_BASE_URL")
    NOWPAYMENTS_EMAIL = os.getenv("NOWPAYMENTS_EMAIL")
    NOWPAYMENTS_PASSWORD = os.getenv("NOWPAYMENTS_PASSWORD")
    BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


settings = Settings()

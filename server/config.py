# config.py
import os
from functools import lru_cache
from urllib.parse import urlparse

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()   # loads .env (Docker compose also injects the same vars)


class Settings(BaseSettings):
    # ──────────────────────────────────────────────────────────────
    # 1. DATABASE (PostgreSQL ONLY)
    # ──────────────────────────────────────────────────────────────
    DATABASE_URL: str

    # ──────────────────────────────────────────────────────────────
    # 2. Auth / Tokens
    # ──────────────────────────────────────────────────────────────
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    REMEMBER_ME_REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # ──────────────────────────────────────────────────────────────
    # 3. Environment
    # ──────────────────────────────────────────────────────────────
    IS_PRODUCTION: bool = False

    # ──────────────────────────────────────────────────────────────
    # 4. External APIs
    # ──────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    GOOGLE_CLIENT_ID: str | None = None
    GOOGLE_CLIENT_SECRET: str | None = None
    NOWPAYMENTS_API_KEY: str | None = None
    NOWPAYMENTS_IPN_SECRET: str | None = None
    NOWPAYMENTS_BASE_URL: str | None = None
    NOWPAYMENTS_EMAIL: str | None = None
    NOWPAYMENTS_PASSWORD: str | None = None

    # ──────────────────────────────────────────────────────────────
    # 5. URLs
    # ──────────────────────────────────────────────────────────────
    BASE_URL: str = "https://itradex.xyz"
    REDIS_URL: str = "redis://redis:6379"

    # ──────────────────────────────────────────────────────────────
    # 6. **NEW** – reCAPTCHA & Email (the ones that were “extra”)
    # ──────────────────────────────────────────────────────────────
    RECAPTCHA_SECRET_KEY: str | None = None

    ZOHO_SMTP_SERVER: str | None = None
    ZOHO_SMTP_PORT: int | None = None
    ZOHO_SENDER_EMAIL: str | None = None
    ZOHO_APP_PASSWORD: str | None = None

    # ──────────────────────────────────────────────────────────────
    # 7. **NEW** – Postgres vars (used in DATABASE_URL template)
    # ──────────────────────────────────────────────────────────────
    POSTGRES_DB: str | None = None
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: str | None = None

    # ──────────────────────────────────────────────────────────────
    # 8. Pydantic config
    # ──────────────────────────────────────────────────────────────
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"          # ← **IMPORTANT** – ignore unknown vars

    # ──────────────────────────────────────────────────────────────
    # 9. Validate DB URL at import time
    # ──────────────────────────────────────────────────────────────
    def _validate_db_url(self) -> None:
        if not self.DATABASE_URL:
            raise RuntimeError("DATABASE_URL is required")
        p = urlparse(self.DATABASE_URL)
        if p.scheme not in ("postgresql+asyncpg", "postgresql"):
            raise ValueError(
                f"Invalid DB scheme '{p.scheme}'. Use 'postgresql+asyncpg://...'"
            )

    def __post_init__(self) -> None:
        self._validate_db_url()


@lru_cache
def get_settings() -> Settings:
    """Singleton – safe for FastAPI, Alembic, background tasks."""
    return Settings()
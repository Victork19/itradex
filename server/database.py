# server/database.py
from urllib.parse import urlparse
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

from config import get_settings

settings = get_settings()

# --------------------------------------------------------------
# 1. Enforce PostgreSQL
# --------------------------------------------------------------
if not settings.DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing – check .env or docker-compose")

parsed = urlparse(settings.DATABASE_URL)
if parsed.scheme not in ("postgresql+asyncpg", "postgresql"):
    raise ValueError(
        f"Invalid DB URL scheme '{parsed.scheme}'. "
        "Use 'postgresql+asyncpg://...' for async."
    )

# --------------------------------------------------------------
# 2. Async engine + session factory
# --------------------------------------------------------------
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()

# --------------------------------------------------------------
# 3. FastAPI dependency
# --------------------------------------------------------------
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# --------------------------------------------------------------
# 4. **CRITICAL**: Import ALL models to populate Base.metadata
# --------------------------------------------------------------
# This MUST be after Base = declarative_base()
from models import *  # ← Pulls in Waitlist, User, etc.
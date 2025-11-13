# alembic/env.py
import os
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Connection

from alembic import context

# ------------------------------------------------------------------
# 1. Your project imports
# ------------------------------------------------------------------
from config import get_settings
from database import Base  # ← pulls in ALL models

# ------------------------------------------------------------------
# 2. Settings
# ------------------------------------------------------------------
settings = get_settings()

# ------------------------------------------------------------------
# 3. Alembic needs a *sync* URL → strip asyncpg
# ------------------------------------------------------------------
if not settings.DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing")

sync_url = settings.DATABASE_URL.replace("+asyncpg", "")
context.config.set_main_option("sqlalchemy.url", sync_url)

# ------------------------------------------------------------------
# 4. Logging – safe (skip if no [formatters])
# ------------------------------------------------------------------
ini_path = context.config.config_file_name
if ini_path and Path(ini_path).exists():
    try:
        fileConfig(ini_path)
    except KeyError:
        pass  # No logging config → use defaults

# ------------------------------------------------------------------
# 5. Metadata for autogenerate
# ------------------------------------------------------------------
target_metadata = Base.metadata


# ------------------------------------------------------------------
# 6. Offline mode
# ------------------------------------------------------------------
def run_migrations_offline() -> None:
    url = context.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# ------------------------------------------------------------------
# 7. Online mode – **SYNC ENGINE ONLY**
# ------------------------------------------------------------------
def run_migrations_online() -> None:
    # Use sync engine – works with psycopg2-binary
    connectable = engine_from_config(
        context.config.get_section(context.config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


# ------------------------------------------------------------------
# 8. Execute
# ------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()   # ← **No asyncio.run()**
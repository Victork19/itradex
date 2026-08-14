# iTradeX

iTradeX (a.k.a. iTrade Journal) is a trading journal SaaS platform for forex and crypto traders. Users upload screenshots of their trades, an AI pipeline extracts the trade data via OCR/vision, and the app turns that into a structured journal with analytics, an AI trading coach, a subscription/marketplace system, and a beta/waitlist funnel.

The backend is a single FastAPI application that serves both a JSON API and server-rendered (Jinja2) HTML pages.

## Features

- **Trade journal** — log trades (entry/exit, SL/TP, position size, leverage, R:R, fees, tags, notes) and browse/edit/delete them via a paginated API (`server/router/journal.py`).
- **AI-powered trade extraction** — upload a trade screenshot and have entry/exit/SL/TP/direction/etc. auto-extracted via OCR (Tesseract) + an LLM pipeline, with batch upload support (`server/router/uploads.py`, `server/app_utils/uploads_utils.py`).
- **AI chat coach** — a chat assistant (text + voice transcribe/speak) for trading insights, gated by a monthly usage limit (`server/router/ai.py`).
- **Insights & dashboard** — computed performance stats (win rate, P&L, equity curve, top tickers, etc.) with CSV/data export (`server/router/insights.py`, `server/router/dashboard.py`).
- **Subscriptions & payments** — tiered plans (starter/pro/elite) billed in crypto via NOWPayments, plus a creator marketplace where eligible traders can sell subscriptions to their own trade feed, with automatic renewal invoicing via APScheduler (`server/router/payments.py`, `server/router/subscriptions.py`, `server/main.py`).
- **Trade Points (TP) economy** — a credit system that gates uploads, insights, and AI chats per plan, earnable via referrals (`server/app_utils/points.py`, `server/models/models.py`).
- **Referral program** — referral codes, tiers, and point rewards for inviting new users.
- **Beta / waitlist system** — a public waitlist with referral leaderboard, email verification, Google OAuth login, and admin-issued beta invite codes (`server/router/waitlist.py`).
- **Admin dashboard** — a full HTML admin panel for managing users, beta invites, waitlist, pricing/discounts, plan limits, payments, and marketplace approvals (`server/router/admin.py`).
- **Auth** — email/password (JWT access + refresh tokens, cookie-based) and Google OAuth, with email verification (`server/auth.py`, `server/router/users.py`, `server/services/email_service.py`).
- **Caching** — Redis-backed caching for expensive/frequently-read data (e.g. plan pricing).

## Tech stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (async) + [Jinja2](https://jinja.palletsprojects.com/) templates
- **Database**: PostgreSQL via SQLAlchemy (async, `asyncpg`) + Alembic migrations
- **Cache**: Redis
- **Auth**: JWT (`python-jose`), `passlib`/`bcrypt`, Google OAuth (`authlib`)
- **AI/OCR**: OpenAI API, Tesseract OCR (`pytesseract`)
- **Payments**: NOWPayments (crypto)
- **Jobs**: APScheduler (cron-based subscription renewals)
- **Rate limiting**: SlowAPI
- **Server**: Uvicorn
- **Containerization**: Docker + docker-compose (app, Postgres, Redis, Nginx, S3 backups)

## Project structure

```
.
├── Dockerfile                  # App image (Python 3.12 + Tesseract)
├── ngrok-v3-stable-linux-amd64.tgz
└── server/
    ├── main.py                 # FastAPI app: startup, middleware, page routes, router wiring
    ├── auth.py                 # Password hashing, JWT creation/validation, current-user deps
    ├── config.py                # Pydantic settings loaded from environment / .env
    ├── database.py              # SQLAlchemy async engine/session setup
    ├── redis_client.py          # Redis pool + cache helpers
    ├── templates_config.py      # Jinja2 environment configuration
    ├── docker-compose.yml       # app + nginx + postgres + redis + backups
    ├── nginx.conf
    ├── alembic/                 # DB migrations
    ├── models/
    │   ├── models.py            # SQLAlchemy ORM models (User, Trade, Subscription, ...)
    │   └── schemas.py           # Pydantic request/response schemas
    ├── router/                  # API/page routers, one module per domain
    │   ├── users.py             # Signup/login/logout, Google OAuth, email verification
    │   ├── journal.py           # Trade CRUD, marketplace trader eligibility/earnings
    │   ├── uploads.py           # Screenshot upload → AI extraction → save trades
    │   ├── insights.py          # Analytics/insights page + data + export
    │   ├── ai.py                # AI chat, voice transcribe/speak, TP spending
    │   ├── dashboard.py         # Dashboard page
    │   ├── profile.py           # Profile view/update, onboarding, subscription mgmt
    │   ├── subscriptions.py     # Subscription lifecycle (cancel/renew/retry)
    │   ├── payments.py          # NOWPayments checkout, webhook, verification
    │   ├── notifications.py     # In-app notifications
    │   ├── waitlist.py          # Public waitlist, referral leaderboard, beta access
    │   └── admin.py             # Admin dashboard and management endpoints
    ├── app_utils/                # Shared helpers (uploads/OCR, points, discounts, admin)
    ├── services/
    │   └── email_service.py     # Transactional email (Zoho SMTP)
    ├── templates/                # Jinja2 HTML templates (landing, dashboard, admin, etc.)
    ├── static/                   # Static assets (favicon, logo)
    └── requirements.txt
```

## Getting started

### Prerequisites

- Docker and Docker Compose (recommended), **or** Python 3.12, PostgreSQL, Redis, and Tesseract OCR installed locally.

### Environment variables

Create a `server/.env` file (see `server/config.py` for the full list). At minimum:

```env
DATABASE_URL=postgresql+asyncpg://user:password@db:5432/itrade
SECRET_KEY=change-me
REDIS_URL=redis://redis:6379

# Optional / feature-dependent
OPENAI_API_KEY=
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
NOWPAYMENTS_API_KEY=
NOWPAYMENTS_IPN_SECRET=
NOWPAYMENTS_BASE_URL=
ZOHO_SMTP_SERVER=
ZOHO_SMTP_PORT=
ZOHO_SENDER_EMAIL=
ZOHO_APP_PASSWORD=
RECAPTCHA_SECRET_KEY=
POSTGRES_DB=itrade
POSTGRES_USER=user
POSTGRES_PASSWORD=password
```

### Run with Docker Compose

```bash
cd server
docker compose up --build
```

This starts the FastAPI app (`:8000`), Postgres, Redis, Nginx (`:80`), and a scheduled S3 backup service. On startup, the app creates tables and seeds default pricing/discount rows automatically.

### Run locally without Docker

```bash
cd server
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Requires a running PostgreSQL and Redis instance reachable via `DATABASE_URL` / `REDIS_URL`, and Tesseract OCR installed on the host for upload extraction to work.

The app will be available at `http://localhost:8000`.

### Database migrations

```bash
cd server
alembic upgrade head
```

## API docs

Once running, interactive API docs are available at `/docs` (Swagger UI) and `/redoc`.

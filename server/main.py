from pathlib import Path
import logging
from typing import Optional
from collections import defaultdict
from datetime import datetime, timedelta, date
import json
import re
from fastapi import FastAPI, Request, Depends, Cookie, HTTPException, status, Form, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_, insert

# NEW: Redis imports
from redis.asyncio import Redis  # NEW: Import Redis type
from redis_client import init_redis, close_redis, redis_dependency, get_cache, set_cache

from database import Base, engine, get_session
from templates_config import templates
from models import models
from models.schemas import TradeResponse, ProfileUpdateRequest
import auth

# APScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Import get_profile directly for reuse in /profile route
from router.profile import get_profile
from router.profile import _compute_profile_stats
from auth import get_current_user_optional

# Routers
from router import (
    users, uploads, insights, journal, profile,
    admin, ai, payments, dashboard, subscriptions,
    notifications, waitlist
)

# --- NEW: Import required functions from payments ---
from router.payments import get_nowpayments_token, create_direct_invoice
from config import get_settings

# NEW: SlowAPI imports for rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

settings = get_settings()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("iTrade")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="iTrade Journal")

# NEW: Global Limiter setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# SessionMiddleware FIRST (before CORS, for better cookie/response handling)
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)

# CORS (after Session)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    logger.info("Mounted static directory: %s", STATIC_DIR)
else:
    logger.warning("Static directory not found: %s", STATIC_DIR)

# --- Seed Pricing & Discount ---
async def seed_pricing_and_discounts():
    async with engine.begin() as conn:
        pricing_plans = [
            {"plan": "pro", "interval": "monthly", "amount": 9.99},
            {"plan": "pro", "interval": "yearly", "amount": 99.00},
            {"plan": "elite", "interval": "monthly", "amount": 19.99},
            {"plan": "elite", "interval": "yearly", "amount": 199.00},
        ]
        inserted = []
        for p in pricing_plans:
            stmt = select(models.Pricing).where(
                models.Pricing.plan == p["plan"],
                models.Pricing.interval == p["interval"]
            )
            exists = await conn.execute(stmt)
            if not exists.scalar():
                await conn.execute(insert(models.Pricing).values(**p))
                inserted.append(p)
                logger.info(f"Seeded pricing: {p['plan']} {p['interval']} @ ${p['amount']}")

        stmt = select(models.Discount).where(models.Discount.id == 1)
        exists = await conn.execute(stmt)
        if not exists.scalar():
            await conn.execute(insert(models.Discount).values(
                id=1, enabled=True, percentage=10.0, expiry=None
            ))
            logger.info("Seeded default Discount (10% off, enabled indefinitely).")

        if inserted:
            logger.info(f"Seeded {len(inserted)} pricing rows.")
        else:
            logger.info("All pricing rows already exist.")

async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await seed_pricing_and_discounts()
    logger.info("Database models initialized and seeded")

# --- AUTO RENEWALS CRON JOB (MOVED HERE) ---
async def auto_generate_renewals(db: AsyncSession, redis: Redis):
    """Generate renewal invoices 3 days before next_billing_date."""
    due_date = datetime.utcnow() + timedelta(days=3)
    result = await db.execute(
        select(models.Subscription).where(
            models.Subscription.status == 'active',
            models.Subscription.next_billing_date <= due_date,
            models.Subscription.renewal_url.is_(None)
        )
    )
    subs = result.scalars().all()

    if not subs:
        logger.info("No subscriptions due for auto-renewal.")
        return

    token = await get_nowpayments_token(redis)
    for sub in subs:
        try:
            order_id = f"{sub.user_id}_{sub.plan_type}_renew"
            order_description = f"Auto-renewal for {sub.plan_type} (Sub ID: {sub.id})"
            success_params = f"&sub_id={sub.id}"

            invoice_url = await create_direct_invoice(
                amount=sub.amount_usd,
                order_id=order_id,
                order_description=order_description,
                token=token,
                success_params=success_params
            )

            # Create pending payment record
            db_payment = models.Payment(
                user_id=sub.user_id,
                subscription_id=sub.id,
                amount_usd=sub.amount_usd,
                status='generated',
                order_id=order_id,
                invoice_url=invoice_url
            )
            db.add(db_payment)

            # Update subscription
            sub.renewal_url = invoice_url
            sub.status = 'pending_renewal'  # Optional: custom status
            sub.updated_at = datetime.utcnow()

            await db.commit()
            logger.info(f"Auto-generated renewal invoice for sub {sub.id}: {invoice_url}")

            # TODO: Trigger in-app notification (e.g., DB flag, push, email via your system)

        except Exception as e:
            logger.error(f"Failed to generate renewal for sub {sub.id}: {e}")
            await db.rollback()

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    await init_models()
    await init_redis(getattr(settings, 'REDIS_URL', "redis://localhost:6379"))  # NEW: Initialize Redis

    # Start APScheduler
    scheduler = AsyncIOScheduler()
    # Patch: Run auto_generate_renewals with session via wrapper
    async def wrapped_renewals():
        async with get_session() as db:
            redis = redis_dependency()
            await auto_generate_renewals(db=db, redis=redis)
    scheduler.add_job(
        wrapped_renewals,
        trigger=CronTrigger(hour=2, minute=0),
        id="auto_renewals_wrapped",
        replace_existing=True
    )
    scheduler.start()
    logger.info("APScheduler started: auto-renewals scheduled at 2:00 AM UTC")

# NEW: Shutdown Event for Redis
@app.on_event("shutdown")
async def shutdown_event():
    await close_redis()

# --- Middleware (UPDATED: Validate token early for protected routes) ---
@app.middleware("http")
async def auth_redirect_middleware(request: Request, call_next):
    protected = [
        "/dashboard", "/insights", "/profile", "/plans", "/upload",
        "/journal", "/onboard", "/chat", "/subscriptions"
    ]
    # NEW: Exempt payment success page (public, no auth needed)
    if request.url.path.startswith("/payment-success"):
        return await call_next(request)
    if any(request.url.path.startswith(p) for p in protected):
        access_token = request.cookies.get("access_token")
        if not access_token:
            logger.info(f"Redirecting unauthenticated {request.url.path} -> /")
            return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        try:
            # Validate token without DB query (just decode for expiry/invalid check)
            payload = auth.decode_access_token(access_token)
            # If decode succeeds, proceed (user fetch happens in Depends later)
        except HTTPException:
            logger.info(f"Redirecting expired/invalid token for {request.url.path} -> /")
            return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    return await call_next(request)

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def root(
    request: Request,
    current_user: Optional[models.User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_session)
):
    # NEW: Redirect to onboarding if user is logged in but hasn't completed it
    if current_user and not current_user.trading_style:
        return RedirectResponse(url="/onboard", status_code=status.HTTP_303_SEE_OTHER)
    
    # Fetch/create beta config (consistent with users.py logic)
    beta_config = await users.get_beta_config(db)  # This creates/commits if missing, defaults to active=True

    context = {
        "request": request,
        "now": datetime.utcnow(),
        "is_logged_in": bool(current_user),
        "current_user": current_user,
        "success": "success" in request.query_params,
        "ref_code": request.query_params.get("ref"),
        "beta_config": beta_config  # Now guaranteed active if no row
    }
    return templates.TemplateResponse("hero.html", context)

@app.get("/auth", response_class=HTMLResponse)
async def auth_page(
    request: Request,
    current_user: Optional[models.User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_session)
):
    # If already logged in, redirect to dashboard
    if current_user:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    
    # Fetch/create beta config (consistent with users.py logic)
    beta_config = await users.get_beta_config(db)  # This creates/commits if missing, defaults to active=True

    context = {
        "request": request,
        "tab": request.query_params.get("tab", "signup"),
        "now": datetime.utcnow(),
        "is_logged_in": bool(current_user),
        "current_user": current_user,
        "success": "success" in request.query_params,
        "ref_code": request.query_params.get("ref"),
        "beta_config": beta_config  # Now guaranteed active if no row
    }
    return templates.TemplateResponse("index.html", context)

# NEW: Payment Success Page Route (public, verifies payment status)
@app.get("/payment-success", response_class=HTMLResponse)
async def payment_success_page(
    request: Request,
    NP_id: Optional[str] = Query(None),  # From query: ?NP_id=5078472979
    sub_id: Optional[str] = Query(None),
    trader_id: Optional[str] = Query(None),
    payment: str = Query("success"),  # Default from NowPayments
    db: AsyncSession = Depends(get_session)
):
    if payment != "success":
        raise HTTPException(status_code=400, detail="Invalid payment status")

    # Verify payment (public lookup—no auth needed)
    payment_status = "pending"  # Default
    user_email = None
    plan_type = None
    is_marketplace = False
    has_email = False  # NEW: Boolean for JS safety
    if NP_id:
        # Lookup Payment by nowpayments_payment_id (from webhook)
        result = await db.execute(
            select(models.Payment).where(models.Payment.nowpayments_payment_id == NP_id)
        )
        db_payment = result.scalar_one_or_none()
        if db_payment and db_payment.status in ["finished", "finished_auto", "finished_manual"]:
            payment_status = "confirmed"
            # Fetch user/sub for details
            if db_payment.user_id:
                user_result = await db.execute(select(models.User).where(models.User.id == db_payment.user_id))
                user = user_result.scalar_one_or_none()
                user_email = user.email if user else None
                has_email = bool(user_email)
            if db_payment.subscription_id:
                sub_result = await db.execute(
                    select(models.Subscription).where(models.Subscription.id == db_payment.subscription_id)
                )
                db_sub = sub_result.scalar_one_or_none()
                if db_sub:
                    plan_type = db_sub.plan_type
                    is_marketplace = bool(db_sub.trader_id)
        else:
            # Fallback: Check Subscription by sub_id (for initial/renew)
            if sub_id:
                try:
                    sub_int = int(sub_id)
                    sub_result = await db.execute(
                        select(models.Subscription).where(
                            models.Subscription.id == sub_int,
                            models.Subscription.status == "active"  # Assume webhook ran
                        )
                    )
                    db_sub = sub_result.scalar_one_or_none()
                    if db_sub:
                        payment_status = "confirmed"
                        user_result = await db.execute(select(models.User).where(models.User.id == db_sub.user_id))
                        user = user_result.scalar_one_or_none()
                        user_email = user.email if user else None
                        has_email = bool(user_email)
                        plan_type = db_sub.plan_type
                        is_marketplace = bool(db_sub.trader_id)
                except ValueError:
                    pass  # Invalid sub_id

    # If still pending, show "processing" (webhook might be delayed—poll or wait)
    if payment_status != "confirmed":
        logger.warning(f"Payment verification pending for NP_id={NP_id}, sub_id={sub_id}")

    context = {
        "request": request,
        "NP_id": NP_id,
        "sub_id": sub_id,
        "trader_id": trader_id,
        "payment_status": payment_status,
        "user_email": user_email,
        "has_email": has_email,  # NEW
        "plan_type": plan_type,
        "is_marketplace": is_marketplace,
        "now": datetime.utcnow(),
    }
    return templates.TemplateResponse("success.html", context)

@app.get("/onboard", response_class=HTMLResponse)
async def onboard_page(
    request: Request,
    current_user: Optional[models.User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_session)
):
    if not current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    if current_user.trading_style:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)

    initials = (
        "".join([n[0].upper() for n in current_user.full_name.split()[:2]]) if current_user.full_name
        else "U"
    )
    return templates.TemplateResponse("onboarding.html", {
        "request": request, "current_user": current_user,
        "initials": initials, "now": datetime.utcnow()
    })

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request, current_user: Optional[models.User] = Depends(get_current_user_optional)):
    if not current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    initials = (
        "".join([n[0].upper() for n in current_user.full_name.split()[:2]]) if current_user.full_name
        else "U"
    )
    return templates.TemplateResponse("upload.html", {
        "request": request, "current_user": current_user,
        "initials": initials, "now": datetime.utcnow()
    })

@app.get("/journal", response_class=HTMLResponse)
async def journal_page(request: Request, current_user: Optional[models.User] = Depends(get_current_user_optional)):
    if not current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    initials = (
        "".join([n[0].upper() for n in current_user.full_name.split()[:2]]) if current_user.full_name
        else "U"
    )
    return templates.TemplateResponse("journal.html", {
        "request": request, "current_user": current_user,
        "initials": initials, "now": datetime.utcnow()
    })

@app.get("/profile", response_class=HTMLResponse)
async def profile_page(
    request: Request,
    current_user: Optional[models.User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_session),
    redis: Redis = Depends(redis_dependency)
):
    if not current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    # UPDATED: Direct call to get_profile with redis (caching handled in API)
    profile_data = await get_profile(db=db, current_user=current_user, redis=redis)

    initials = (
        "".join([n[0].upper() for n in current_user.full_name.split()[:2]]) if current_user.full_name
        else "U"
    )

    return templates.TemplateResponse("profile.html", {
        "request": request, "current_user": current_user,
        "profile_data": profile_data, "initials": initials,
        "lifetime_pnl": profile_data.get('lifetime_pnl', 0),
        "win_rate": profile_data.get('win_rate', 0),
        "best_trade": profile_data.get('best_trade'),
        "worst_trade": profile_data.get('worst_trade'),
        "top_tickers": profile_data.get('top_tickers', []),
        "formatted_joined": current_user.created_at.strftime('%B %d, %Y') if current_user.created_at else '',
        "bio": profile_data.get('bio', ''), "trading_style": profile_data.get('trading_style', ''),
        "goals": profile_data.get('goals', ''), "now": datetime.utcnow()
    })

# --------------------------------------------------------------
#  REPLACE the existing @app.get("/plans") block with this one
# --------------------------------------------------------------
@app.get("/plans", response_class=HTMLResponse)
async def plans_page(
    request: Request,
    current_user: Optional[models.User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_session),
    redis: Redis = Depends(redis_dependency)  # NEW: Add Redis dependency
):
    if not current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    # NEW: Cache plans data globally (TTL 1 hour, as pricing/limits change rarely)
    cache_key = "plans_data"
    plans_data = await get_cache(redis, cache_key)
    if plans_data is None:
        # ------------------------------------------------------------------
        # 1. Pricing (still pulled from DB, fallback to defaults)
        # ------------------------------------------------------------------
        pricing = {
            "pro_monthly": 9.99, "pro_yearly": 99.00,
            "elite_monthly": 19.99, "elite_yearly": 199.00,
        }
        result = await db.execute(
            select(models.Pricing).where(
                models.Pricing.plan.in_(["pro", "elite"]),
                models.Pricing.interval.in_(["monthly", "yearly"]),
            )
        )
        for p in result.scalars():
            pricing[f"{p.plan}_{p.interval}"] = p.amount

        nested_pricing = {
            "pro": {"monthly": pricing["pro_monthly"], "yearly": pricing["pro_yearly"]},
            "elite": {"monthly": pricing["elite_monthly"], "yearly": pricing["elite_yearly"]},
        }

        # ------------------------------------------------------------------
        # 2. Discount
        # ------------------------------------------------------------------
        result = await db.execute(select(models.Discount).where(models.Discount.id == 1))
        db_discount = result.scalar_one_or_none()
        effective_discount = (
            db_discount.percentage
            if db_discount
            and db_discount.enabled
            and (not db_discount.expiry or db_discount.expiry > date.today())
            else 0.0
        )

        # ------------------------------------------------------------------
        # 3. ALL DYNAMIC LIMITS (the ones that were missing)
        # ------------------------------------------------------------------
        # Helper to fetch a limit by plan
        def _limit(records, plan, field):
            rec = next((r for r in records if r.plan == plan), None)
            return getattr(rec, field) if rec else 0

        # Initial TP (signup & upgrade)
        init_cfg = (await db.execute(select(models.InitialTpConfig).where(models.InitialTpConfig.id == 1))).scalar_one_or_none()
        starter_initial_tp = init_cfg.amount if init_cfg else 3

        upgrade_cfg = (await db.execute(select(models.UpgradeTpConfig))).scalars().all()
        pro_initial_tp   = next((c.amount for c in upgrade_cfg if c.plan == "pro"), 10)
        elite_initial_tp = next((c.amount for c in upgrade_cfg if c.plan == "elite"), 20)

        # Uploads
        uploads = (await db.execute(select(models.UploadLimits))).scalars().all()
        starter_uploads_monthly = _limit(uploads, "starter", "monthly_limit")
        pro_uploads_monthly     = _limit(uploads, "pro", "monthly_limit")
        elite_uploads_monthly   = _limit(uploads, "elite", "monthly_limit")

        # Insights
        insights = (await db.execute(select(models.InsightsLimits))).scalars().all()
        starter_insights_monthly = _limit(insights, "starter", "monthly_limit")
        pro_insights_monthly     = _limit(insights, "pro", "monthly_limit")
        elite_insights_monthly   = _limit(insights, "elite", "monthly_limit")

        # AI chats
        ai_chats = (await db.execute(select(models.AiChatLimits))).scalars().all()
        starter_ai_chats_monthly = _limit(ai_chats, "starter", "monthly_limit")
        pro_ai_chats_monthly     = _limit(ai_chats, "pro", "monthly_limit")
        elite_ai_chats_monthly   = _limit(ai_chats, "elite", "monthly_limit")

        # Referral TP
        ref_cfg = (await db.execute(select(models.BetaReferralTpConfig).where(models.BetaReferralTpConfig.id == 1))).scalar_one_or_none()
        starter_referral_tp = ref_cfg.starter_tp if ref_cfg else 5
        pro_referral_tp     = ref_cfg.pro_tp     if ref_cfg else 20
        elite_referral_tp   = ref_cfg.elite_tp   if ref_cfg else 45

        # Pack into dict for caching
        plans_data = {
            "pricing": {
                "pro_monthly": pricing["pro_monthly"],
                "pro_yearly": pricing["pro_yearly"],
                "elite_monthly": pricing["elite_monthly"],
                "elite_yearly": pricing["elite_yearly"],
            },
            "nested_pricing": nested_pricing,
            "effective_discount": effective_discount,
            # ---- DYNAMIC LIMITS ----
            "starter_initial_tp": starter_initial_tp,
            "pro_initial_tp": pro_initial_tp,
            "elite_initial_tp": elite_initial_tp,
            "starter_uploads_monthly": starter_uploads_monthly,
            "pro_uploads_monthly": pro_uploads_monthly,
            "elite_uploads_monthly": elite_uploads_monthly,
            "starter_insights_monthly": starter_insights_monthly,
            "pro_insights_monthly": pro_insights_monthly,
            "elite_insights_monthly": elite_insights_monthly,
            "starter_ai_chats_monthly": starter_ai_chats_monthly,
            "pro_ai_chats_monthly": pro_ai_chats_monthly,
            "elite_ai_chats_monthly": elite_ai_chats_monthly,
            "starter_referral_tp": starter_referral_tp,
            "pro_referral_tp": pro_referral_tp,
            "elite_referral_tp": elite_referral_tp,
        }
        await set_cache(redis, cache_key, plans_data, ttl=3600)  # 1 hour TTL
        logger.info("Cached plans data")

    # Unpack cached data
    pricing_obj = type("obj", (), plans_data["pricing"])
    nested_pricing = plans_data["nested_pricing"]
    effective_discount = plans_data["effective_discount"]
    # ... (unpack all limits similarly)
    starter_initial_tp = plans_data["starter_initial_tp"]
    pro_initial_tp = plans_data["pro_initial_tp"]
    elite_initial_tp = plans_data["elite_initial_tp"]
    starter_uploads_monthly = plans_data["starter_uploads_monthly"]
    pro_uploads_monthly = plans_data["pro_uploads_monthly"]
    elite_uploads_monthly = plans_data["elite_uploads_monthly"]
    starter_insights_monthly = plans_data["starter_insights_monthly"]
    pro_insights_monthly = plans_data["pro_insights_monthly"]
    elite_insights_monthly = plans_data["elite_insights_monthly"]
    starter_ai_chats_monthly = plans_data["starter_ai_chats_monthly"]
    pro_ai_chats_monthly = plans_data["pro_ai_chats_monthly"]
    elite_ai_chats_monthly = plans_data["elite_ai_chats_monthly"]
    starter_referral_tp = plans_data["starter_referral_tp"]
    pro_referral_tp = plans_data["pro_referral_tp"]
    elite_referral_tp = plans_data["elite_referral_tp"]

    # ------------------------------------------------------------------
    # 4. Current subscription (for “Current” badge)
    # ------------------------------------------------------------------
    result_sub = await db.execute(
        select(models.Subscription)
        .where(models.Subscription.user_id == current_user.id, models.Subscription.status == "active")
        .order_by(desc(models.Subscription.start_date))
    )
    current_sub = result_sub.scalars().first()

    # ------------------------------------------------------------------
    # 5. Avatar initials
    # ------------------------------------------------------------------
    initials = (
        "".join([n[0].upper() for n in current_user.full_name.split()[:2]])
        if current_user.full_name
        else "U"
    )

    # ------------------------------------------------------------------
    # 6. Render
    # ------------------------------------------------------------------
    return templates.TemplateResponse(
        "plans.html",
        {
            "request": request,
            "current_user": current_user,
            "initials": initials,
            # pricing
            "pricing": pricing_obj,
            "nested_pricing": nested_pricing,
            "effective_discount": effective_discount,
            "current_subscription": current_sub,
            # ---- DYNAMIC LIMITS ----
            "starter_initial_tp": starter_initial_tp,
            "pro_initial_tp": pro_initial_tp,
            "elite_initial_tp": elite_initial_tp,
            "starter_uploads_monthly": starter_uploads_monthly,
            "pro_uploads_monthly": pro_uploads_monthly,
            "elite_uploads_monthly": elite_uploads_monthly,
            "starter_insights_monthly": starter_insights_monthly,
            "pro_insights_monthly": pro_insights_monthly,
            "elite_insights_monthly": elite_insights_monthly,
            "starter_ai_chats_monthly": starter_ai_chats_monthly,
            "pro_ai_chats_monthly": pro_ai_chats_monthly,
            "elite_ai_chats_monthly": elite_ai_chats_monthly,
            "starter_referral_tp": starter_referral_tp,
            "pro_referral_tp": pro_referral_tp,
            "elite_referral_tp": elite_referral_tp,
            "now": datetime.utcnow(),
        },
    )



@app.post("/wallet-verify")
async def verify_and_set_wallet(
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Set/update the payout wallet address after basic validation. Requires trader status."""
    if not current_user.is_trader:
        raise HTTPException(status_code=403, detail="Access denied: Not a marketplace trader")

    body = await request.json()
    wallet = body.get("wallet", "").strip()

    if not wallet:
        raise HTTPException(status_code=400, detail="wallet address required")

    if not re.match(r'^0x[a-fA-F0-9]{40}$', wallet):
        raise HTTPException(status_code=400, detail="Invalid Ethereum wallet address")

    try:
        # Set or update wallet
        current_user.wallet_address = wallet
        current_user.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(current_user)
        
        logger.info(f"Wallet set for trader {current_user.id}: {wallet}")
        
        return {
            "verified": True,
            "message": "Wallet address set successfully"
        }
    except Exception as e:
        logger.error(f"Wallet setting failed for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to set wallet address")

# NEW: Static Pages Routes (public, no auth required)
@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    context = {"request": request, "now": datetime.utcnow()}
    return templates.TemplateResponse("privacy.html", context)

@app.get("/terms", response_class=HTMLResponse)
async def terms_page(request: Request):
    context = {"request": request, "now": datetime.utcnow()}
    return templates.TemplateResponse("terms.html", context)

@app.get("/contact", response_class=HTMLResponse)
async def contact_page(request: Request):
    context = {"request": request, "now": datetime.utcnow()}
    return templates.TemplateResponse("contact.html", context)

@app.get("/support", response_class=HTMLResponse)
async def support_page(request: Request):
    context = {"request": request, "now": datetime.utcnow()}
    return templates.TemplateResponse("support.html", context)

# --- Include Routers ---
app.include_router(users.router)
app.include_router(uploads.router)
app.include_router(insights.router)
app.include_router(journal.router)
app.include_router(profile.router)
app.include_router(admin.router)
app.include_router(ai.router)
app.include_router(payments.router)
app.include_router(dashboard.router)
app.include_router(subscriptions.router)
app.include_router(notifications.router)
app.include_router(waitlist.router, prefix="/waitlist")

# --- Redirects ---
@app.get("/subscriptions", response_class=RedirectResponse)
async def redirect_subscriptions():
    return RedirectResponse(url="/subscriptions/", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

@app.get("/subscription", response_class=RedirectResponse)
async def redirect_subscription():
    return RedirectResponse(url="/subscriptions/", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

@app.get("/waitlist", response_class=RedirectResponse)
async def redirect_waitlist():
    return RedirectResponse(url="/waitlist/", status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@app.get("/{full_path:path}")
async def catch_html_redirect(full_path: str):
    if full_path.endswith('.html'):
        clean = full_path[:-5]
        logger.info(f"Redirecting {full_path} → /{clean}")
        return RedirectResponse(url=f"/{clean}", status_code=status.HTTP_301_MOVED_PERMANENTLY)
    raise HTTPException(status_code=404, detail="Not Found")
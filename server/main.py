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

from database import Base, engine, get_session
from templates_config import templates
from models import models
from models.schemas import TradeResponse, ProfileUpdateRequest
import auth
from config import settings

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
    notifications
)

# --- NEW: Import required functions from payments ---
from router.payments import get_nowpayments_token, create_direct_invoice

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("iTrade")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="iTrade Journal")

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
async def auto_generate_renewals(db: AsyncSession):
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

    token = await get_nowpayments_token()
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

    # Start APScheduler
    scheduler = AsyncIOScheduler()
    # Patch: Run auto_generate_renewals with session via wrapper
    async def wrapped_renewals():
        async with get_session() as db:
            await auto_generate_renewals(db=db)
    scheduler.add_job(
        wrapped_renewals,
        trigger=CronTrigger(hour=2, minute=0),
        id="auto_renewals_wrapped",
        replace_existing=True
    )
    scheduler.start()
    logger.info("APScheduler started: auto-renewals scheduled at 2:00 AM UTC")

# --- Middleware (UPDATED: Exempt /payment-success from auth) ---
@app.middleware("http")
async def auth_redirect_middleware(request: Request, call_next):
    protected = [
        "/dashboard", "/insights", "/profile", "/plans", "/upload",
        "/journal", "/onboard", "/chat", "/subscriptions"
    ]
    # NEW: Exempt payment success page (public, no auth needed)
    if request.url.path.startswith("/payment-success"):
        return await call_next(request)
    if any(request.url.path.startswith(p) for p in protected) and not request.cookies.get("access_token"):
        logger.info(f"Redirecting unauthenticated {request.url.path} -> /")
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    return await call_next(request)

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def root(
    request: Request,
    current_user: Optional[models.User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_session)
):
    # Fetch beta config (with fallback)
    result = await db.execute(select(models.BetaConfig).where(models.BetaConfig.id == 1))
    beta_config = result.scalar_one_or_none()
    if not beta_config:
        beta_config = models.BetaConfig(
            id=1,
            is_active=False,  # Default: inactive (normal referral mode)
            required_for_signup=False,
            award_points_on_use=3
        )

    context = {
        "request": request,
        "tab": request.query_params.get("tab", "signup"),
        "now": datetime.utcnow(),
        "is_logged_in": bool(current_user),
        "current_user": current_user,
        "success": "success" in request.query_params,
        "ref_code": request.query_params.get("ref"),
        "beta_config": beta_config  # NEW: Pass to template for conditional rendering
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
    db: AsyncSession = Depends(get_session)
):
    if not current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    profile_data = await get_profile(db=db, current_user=current_user)
    profile_data["computed_daily_limit"] = round(
        (profile_data.get("daily_loss_percent", 5) / 100) * profile_data.get("account_balance", 10000)
    )

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

@app.get("/plans", response_class=HTMLResponse)
async def plans_page(
    request: Request,
    current_user: Optional[models.User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_session)
):
    if not current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    initials = (
        "".join([n[0].upper() for n in current_user.full_name.split()[:2]]) if current_user.full_name
        else "U"
    )

    pricing = {'pro_monthly': 9.99, 'pro_yearly': 99.00, 'elite_monthly': 19.99, 'elite_yearly': 199.00}
    result = await db.execute(select(models.Pricing).where(
        models.Pricing.plan.in_(['pro', 'elite']),
        models.Pricing.interval.in_(['monthly', 'yearly'])
    ))
    for p in result.scalars():
        pricing[f"{p.plan}_{p.interval}"] = p.amount

    discount = {'enabled': False, 'percentage': 0.0, 'expiry': None}
    result = await db.execute(select(models.Discount).where(models.Discount.id == 1))
    db_discount = result.scalar_one_or_none()
    if db_discount:
        discount.update({'enabled': db_discount.enabled, 'percentage': db_discount.percentage, 'expiry': db_discount.expiry})

    effective_discount = db_discount.percentage if db_discount and db_discount.enabled and (not db_discount.expiry or db_discount.expiry > date.today()) else 0.0

    nested_pricing = {
        'pro': {'monthly': pricing['pro_monthly'], 'yearly': pricing['pro_yearly']},
        'elite': {'monthly': pricing['elite_monthly'], 'yearly': pricing['elite_yearly']}
    }

    result_sub = await db.execute(
        select(models.Subscription).where(
            models.Subscription.user_id == current_user.id,
            models.Subscription.status == 'active'
        ).order_by(desc(models.Subscription.start_date))
    )
    current_sub = result_sub.scalars().first()

    return templates.TemplateResponse("plans.html", {
        "request": request, "current_user": current_user, "initials": initials,
        "pricing": pricing, "nested_pricing": nested_pricing,
        "discount": discount, "effective_discount": effective_discount,
        "current_subscription": current_sub, "now": datetime.utcnow()
    })

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

# --- Redirects ---
@app.get("/subscriptions", response_class=RedirectResponse)
async def redirect_subscriptions():
    return RedirectResponse(url="/subscriptions/", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

@app.get("/subscription", response_class=RedirectResponse)
async def redirect_subscriptions():
    return RedirectResponse(url="/subscriptions/", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

@app.get("/{full_path:path}")
async def catch_html_redirect(full_path: str):
    if full_path.endswith('.html'):
        clean = full_path[:-5]
        logger.info(f"Redirecting {full_path} → /{clean}")
        return RedirectResponse(url=f"/{clean}", status_code=status.HTTP_301_MOVED_PERMANENTLY)
    raise HTTPException(status_code=404, detail="Not Found")
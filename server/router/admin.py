# Updated server/router/admin.py
import logging
import secrets
import string
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, Request, Depends, HTTPException, status, Form, Body, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update, or_, desc
from sqlalchemy.orm import joinedload

from templates_config import templates
from models.models import (
    User, Trade, Subscription, Payment, Pricing, Discount, EligibilityConfig, UploadLimits, Notification, InsightsLimits,
    Referral, PointTransaction, InitialTpConfig, UpgradeTpConfig, AiChatLimits, BetaInvite, BetaConfig, BetaReferralTpConfig
)
from database import get_session
import auth

from app_utils.admin_utils import (
    compute_admin_stats, get_all_configs, get_revenue_metrics,
    get_recent_users_with_stats, get_marketplace_traders_with_stats, get_pending_traders_with_stats,
    get_recent_trades_list, get_recent_initiated_platform_list, get_recent_initiated_marketplace_list,
    get_recent_partial_platform_list, get_recent_partial_marketplace_list,
    get_recent_referrals_list, get_recent_points_list,
    get_user_details_data,
    get_users_list, get_referrals_list, get_points_list, get_payments_list
)

logger = logging.getLogger("iTrade")

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("", response_class=HTMLResponse)
async def admin_page(
    request: Request,
    search: Optional[str] = Query(None, description="Search users by email or name"),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    if current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    initials = ""
    if current_user.full_name:
        names = current_user.full_name.split()
        if len(names) >= 2:
            initials = names[0][0].upper() + names[-1][0].upper()
        elif len(names) == 1:
            initials = names[0][0].upper() * 2
    else:
        initials = "U"

    # Fetch data using utils
    stats = await compute_admin_stats(db)
    configs = await get_all_configs(db)
    revenue = await get_revenue_metrics(db)
    recent_users = await get_recent_users_with_stats(db, search)
    marketplace_traders = await get_marketplace_traders_with_stats(db)
    pending_traders = await get_pending_traders_with_stats(db)
    recent_trades = await get_recent_trades_list(db)
    recent_initiated_platform = await get_recent_initiated_platform_list(db)
    recent_initiated_marketplace = await get_recent_initiated_marketplace_list(db)
    recent_partial_platform = await get_recent_partial_platform_list(db)
    recent_partial_marketplace = await get_recent_partial_marketplace_list(db)
    recent_referrals = await get_recent_referrals_list(db)
    recent_points = await get_recent_points_list(db)

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "current_user": current_user,
            "initials": initials,
            **stats,
            **configs,
            **revenue,
            "recent_users": recent_users,
            "marketplace_traders": marketplace_traders,
            "pending_traders": pending_traders,
            "recent_trades": recent_trades,
            "recent_referrals": recent_referrals,
            "recent_points": recent_points,
            "recent_initiated_platform": recent_initiated_platform,
            "recent_initiated_marketplace": recent_initiated_marketplace,
            "recent_partial_platform": recent_partial_platform,
            "recent_partial_marketplace": recent_partial_marketplace,
            "search": search or "",
            'now': datetime.now(),
        }
    )


# ───── GET USER DETAILS (FULL) ─────
@router.get("/user/{user_id}", response_class=JSONResponse)
async def get_user_details(
    user_id: int,
    db: AsyncSession = Depends(get_session)
):
    data = await get_user_details_data(db, user_id)
    return data


def generate_code(length=8):
    """Helper to generate a unique beta invite code."""
    return ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(length))


# ───── NEW: TOGGLE BETA MODE ─────
@router.post("/toggle-beta-mode")
async def toggle_beta_mode(
    is_active: bool = Form(False),  # FIXED: Default to False to handle unchecked checkbox
    required_for_signup: bool = Form(False),
    award_points_on_use: int = Form(3),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(auth.get_current_user)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    if award_points_on_use < 0:
        raise HTTPException(status_code=400, detail="Award points cannot be negative")

    result = await db.execute(select(BetaConfig).where(BetaConfig.id == 1))
    config = result.scalar_one_or_none()
    if config:
        config.is_active = is_active
        config.required_for_signup = required_for_signup
        config.award_points_on_use = award_points_on_use
    else:
        config = BetaConfig(
            id=1,
            is_active=is_active,
            required_for_signup=required_for_signup,
            award_points_on_use=award_points_on_use
        )
        db.add(config)

    await db.commit()

    status_str = "activated" if is_active else "deactivated"
    req_str = "required" if required_for_signup else "optional"
    logger.info(f"Admin {status_str} beta mode (signup code {req_str}, {award_points_on_use} TP award)")
    return JSONResponse({
        "success": True,
        "message": f"Beta mode {status_str}. Signup code is {'required' if required_for_signup else 'optional'}. Award: {award_points_on_use} TP."
    })


# ───── NEW: GENERATE BETA INVITE CODES FOR USER ─────
@router.post("/generate-beta-invites/{user_id}")
async def admin_generate_beta_invites(
    user_id: int,
    count: int = Form(3, ge=1, le=10),  # Limit to reasonable number
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(auth.get_current_user)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    target_user = await db.get(User, user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")

    created_codes = []
    for _ in range(count):
        code = generate_code()
        # Ensure unique
        while (await db.execute(select(BetaInvite).where(BetaInvite.code == code))).first():
            code = generate_code()
        invite = BetaInvite(
            owner_id=user_id,
            code=code,
            created_at=datetime.utcnow(),
            used_by_id=None,
            used_at=None
        )
        db.add(invite)
        created_codes.append(code)

    await db.commit()

    logger.info(f"Admin generated {count} beta invites for user {user_id}: {created_codes}")
    return JSONResponse({
        "success": True,
        "message": f"Generated {count} new beta invite codes: {', '.join(created_codes)}"
    })


# ───── NEW: GENERATE BETA POOL CODES (Global access pool) ─────
@router.post("/generate-beta-pool")
async def admin_generate_beta_pool(
    count: int = Form(10, ge=1, le=100),  # Reasonable limit for pool
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(auth.get_current_user)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    created_codes = []
    for _ in range(count):
        code = generate_code()
        # Ensure unique
        while (await db.execute(select(BetaInvite).where(BetaInvite.code == code))).first():
            code = generate_code()
        invite = BetaInvite(
            owner_id=0,  # FIXED: Global pool: special owner_id=0 (no real user, treated as None in queries)
            code=code,
            created_at=datetime.utcnow(),
            used_by_id=None,
            used_at=None
        )
        db.add(invite)
        created_codes.append(code)

    await db.commit()

    logger.info(f"Admin generated {count} beta pool invites: {created_codes}")
    return JSONResponse({
        "success": True,
        "message": f"Generated {count} beta pool codes: {', '.join(created_codes)}"
    })


# ───── BULK USER MANAGEMENT ─────
@router.get("/users")
async def list_users(
    search: Optional[str] = Query(None),
    plan: Optional[str] = Query(None),
    is_trader: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(auth.get_current_user)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")
    
    data = await get_users_list(db, search, plan, is_trader, limit, offset)
    return data


# ───── BULK REFERRAL MANAGEMENT ─────
@router.get("/referrals")
async def list_referrals(
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(auth.get_current_user)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")
    
    data = await get_referrals_list(db, search, status, limit, offset)
    return data


# ───── NEW: BULK BETA INVITES MANAGEMENT ─────
@router.get("/beta-invites")
async def list_beta_invites(
    search: Optional[str] = Query(None),
    used: Optional[bool] = Query(None),
    owner_id: Optional[int] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(auth.get_current_user)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")
    
    query = select(BetaInvite).options(joinedload(BetaInvite.owner), joinedload(BetaInvite.used_by))
    if search:
        query = query.where(or_(
            BetaInvite.code.ilike(f"%{search}%"),
            BetaInvite.owner.has(User.email.ilike(f"%{search}%")),
            BetaInvite.used_by.has(User.email.ilike(f"%{search}%"))
        ))
    if used is not None:
        if used:
            query = query.where(BetaInvite.used_by_id.is_not(None))
        else:
            query = query.where(BetaInvite.used_by_id.is_(None))
    if owner_id:
        query = query.where(BetaInvite.owner_id == owner_id)
    query = query.order_by(desc(BetaInvite.created_at))
    result = await db.execute(query.limit(limit).offset(offset))
    invites = result.scalars().all()
    total_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total_count = total_result.scalar()

    # Serialize
    serialized = []
    for invite in invites:
        used_email = invite.used_by.email if invite.used_by else None
        owner_email = invite.owner.email if invite.owner else "Global Pool"
        serialized.append({
            "id": invite.id,
            "code": invite.code,
            "owner_id": invite.owner_id,
            "owner_email": owner_email,
            "used_by_id": invite.used_by_id,
            "used_by_email": used_email,
            "created_at": invite.created_at.isoformat(),
            "used_at": invite.used_at.isoformat() if invite.used_at else None
        })
    return {"invites": serialized, "total": total_count}


# ───── BULK POINT TRANSACTIONS ─────
@router.get("/points")
async def list_points(
    search: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    user_id: Optional[int] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(auth.get_current_user)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")
    
    data = await get_points_list(db, search, type, user_id, limit, offset)
    return data


# ───── ADMIN ADJUST POINTS (Add/Remove) ─────
@router.post("/points/adjust")
async def admin_adjust_points(
    user_id: int = Form(...),
    amount: int = Form(...),
    reason: str = Form("Admin adjustment"),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(auth.get_current_user)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")
    
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.trade_points += amount
    user.trade_points = max(0, user.trade_points)
    
    tx = PointTransaction(
        user_id=user_id,
        type="admin_adjust",
        amount=amount,
        description=f"{reason} ({'Added' if amount > 0 else 'Deducted'} {abs(amount)} TP)"
    )
    db.add(tx)
    await db.commit()
    await db.refresh(user)
    
    logger.info(f"Admin adjusted {amount} TP for user {user_id}: {reason}")
    return JSONResponse({
        "success": True,
        "message": f"{'Added' if amount > 0 else 'Deducted'} {abs(amount)} TP for user {user_id}. New balance: {user.trade_points}",
        "new_balance": user.trade_points
    })


# ───── LIST ALL PAYMENTS WITH FILTERS ─────
@router.get("/payments")
async def list_payments(
    status: Optional[str] = Query(None),
    user_id: Optional[int] = Query(None),
    crypto_currency: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(auth.get_current_user)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")
    
    data = await get_payments_list(db, status, user_id, crypto_currency, limit, offset)
    return data


# ───── REFUND/CANCEL PAYMENT ─────
@router.post("/payments/refund/{payment_id}")
async def admin_refund_payment(
    payment_id: str,
    reason: str = Form("Admin refund"),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(auth.get_current_user)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")
    
    payment = await db.get(Payment, payment_id, Payment.nowpayments_payment_id)
    if not payment:
        raise HTTPException(status_code=404, detail="Payment not found")
    
    if payment.status not in ['finished', 'partially_paid']:
        raise HTTPException(status_code=400, detail="Only completed payments can be refunded")
    
    payment.status = 'refunded'
    payment.notes = reason
    await db.commit()
    
    if payment.subscription_id:
        sub = await db.get(Subscription, payment.subscription_id)
        if sub:
            sub.status = 'paused'
            await db.commit()
    
    logger.info(f"Admin refunded payment {payment_id}: {reason}")
    return JSONResponse({
        "success": True,
        "message": f"Payment {payment_id} refunded and marked as 'refunded'. Linked subscription paused if applicable."
    })


# ───── SEND BULK NOTIFICATION ─────
@router.post("/notifications/send_bulk")
async def admin_send_bulk_notification(
    title: str = Form(...),
    message: str = Form(...),
    user_ids: List[int] = Body(...),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(auth.get_current_user)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not user_ids:
        user_ids = [u.id for u in (await db.execute(select(User))).scalars().all()]
    
    created_count = 0
    for uid in user_ids:
        notif = Notification(
            user_id=uid,
            title=title,
            message=message,
            type="admin_broadcast"
        )
        db.add(notif)
        created_count += 1
    
    await db.commit()
    logger.info(f"Admin sent bulk notification '{title}' to {created_count} users")
    return JSONResponse({
        "success": True,
        "message": f"Bulk notification sent to {created_count} users",
        "title": title
    })


# ───── GENERATE FAKE DATA (for testing) ─────
@router.post("/generate_fake_data")
async def admin_generate_fake_data(
    num_users: int = Form(10, ge=1, le=100),
    num_trades_per_user: int = Form(5, ge=1, le=20),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(auth.get_current_user)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")
    
    from faker import Faker
    fake = Faker()
    
    created_users = 0
    created_trades = 0
    
    for _ in range(num_users):
        new_user = User(
            username=fake.user_name(),
            full_name=fake.name(),
            email=fake.email(),
            password_hash="fake_hash",
            referral_code=fake.uuid4()[:8],
            plan="starter",
            created_at=datetime.utcnow() - timedelta(days=fake.random_int(1, 365))
        )
        db.add(new_user)
        created_users += 1
        
        for __ in range(num_trades_per_user):
            new_trade = Trade(
                owner_id=new_user.id,
                symbol=fake.random_element(['EUR/USD', 'BTC/USD', 'GBP/JPY']),
                entry_price=fake.pydecimal(left_digits=3, right_digits=4, positive=True),
                exit_price=fake.pydecimal(left_digits=3, right_digits=4, positive=True),
                pnl=fake.pydecimal(left_digits=1, right_digits=2, positive=fake.boolean()),
                created_at=new_user.created_at + timedelta(days=fake.random_int(1, 365))
            )
            db.add(new_trade)
            created_trades += 1
    
    await db.commit()
    
    logger.info(f"Generated {created_users} fake users and {created_trades} trades")
    return JSONResponse({
        "success": True,
        "message": f"Generated {created_users} users and {created_trades} trades. Refresh dashboard."
    })


# ───── UPDATE PRICING ─────
@router.post("/update_pricing")
async def update_pricing(
    pro_monthly: float = Form(...), pro_yearly: float = Form(...),
    elite_monthly: float = Form(...), elite_yearly: float = Form(...),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    prices = [
        {'plan': 'pro', 'interval': 'monthly', 'amount': pro_monthly},
        {'plan': 'pro', 'interval': 'yearly', 'amount': pro_yearly},
        {'plan': 'elite', 'interval': 'monthly', 'amount': elite_monthly},
        {'plan': 'elite', 'interval': 'yearly', 'amount': elite_yearly},
    ]

    for p in prices:
        await db.execute(update(Pricing).where(
            Pricing.plan == p['plan'], Pricing.interval == p['interval']
        ).values(amount=p['amount']))
        if not (await db.execute(select(Pricing).where(
            Pricing.plan == p['plan'], Pricing.interval == p['interval']
        ))).scalar():
            db.add(Pricing(**p))
    await db.commit()
    return JSONResponse({"success": True, "message": "Pricing updated"})


# ───── UPDATE PLATFORM DISCOUNT (id=1) ─────
@router.post("/update_discount")
async def update_discount(
    discount_enabled: bool = Form(False),
    discount_percentage: float = Form(0.0),
    discount_expiry: str = Form(None),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    expiry_date = None
    if discount_expiry:
        try:
            expiry_date = datetime.strptime(discount_expiry, '%Y-%m-%d').date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid expiry date format")

    result = await db.execute(select(Discount).where(Discount.id == 1))
    db_discount = result.scalar_one_or_none()

    if db_discount:
        db_discount.enabled = discount_enabled
        db_discount.percentage = discount_percentage if discount_enabled else 0.0
        db_discount.expiry = expiry_date if discount_enabled else None
    else:
        db_discount = Discount(id=1, enabled=discount_enabled, percentage=discount_percentage if discount_enabled else 0.0, expiry=expiry_date)
        db.add(db_discount)

    await db.commit()
    return JSONResponse({"success": True, "message": "Discount updated"})


# ───── UPDATE MARKETPLACE DISCOUNT (id=2) ─────
@router.post("/update_marketplace_discount")
async def update_marketplace_discount(
    marketplace_discount_enabled: bool = Form(False),
    marketplace_discount_percentage: float = Form(0.0),
    marketplace_discount_expiry: str = Form(None),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    expiry_date = None
    if marketplace_discount_expiry:
        try:
            expiry_date = datetime.strptime(marketplace_discount_expiry, '%Y-%m-%d').date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid expiry date format")

    result = await db.execute(select(Discount).where(Discount.id == 2))
    db_discount = result.scalar_one_or_none()

    if db_discount:
        db_discount.enabled = marketplace_discount_enabled
        db_discount.percentage = marketplace_discount_percentage if marketplace_discount_enabled else 0.0
        db_discount.expiry = expiry_date if marketplace_discount_enabled else None
    else:
        db_discount = Discount(id=2, enabled=marketplace_discount_enabled, percentage=marketplace_discount_percentage if marketplace_discount_enabled else 0.0, expiry=expiry_date)
        db.add(db_discount)

    await db.commit()
    return JSONResponse({"success": True, "message": "Marketplace discount updated"})


# ───── UPDATE ELIGIBILITY (UPDATED: Include trader_share_percent) ─────
@router.post("/update_eligibility")
async def update_eligibility(
    min_trades: int = Form(50),
    min_win_rate: float = Form(80.0),
    max_marketplace_price: float = Form(99.99),
    trader_share_percent: float = Form(70.0),  # NEW: Trader's share (0-100)
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    if min_trades < 1 or min_win_rate < 0 or min_win_rate > 100 or max_marketplace_price < 0 or trader_share_percent < 0 or trader_share_percent > 100:
        raise HTTPException(status_code=400, detail="Invalid thresholds")

    result = await db.execute(select(EligibilityConfig).where(EligibilityConfig.id == 1))
    db_config = result.scalar_one_or_none()
    if not db_config:
        db_config = EligibilityConfig(id=1)
        db.add(db_config)

    db_config.min_trades = min_trades
    db_config.min_win_rate = min_win_rate
    db_config.max_marketplace_price = max_marketplace_price
    db_config.trader_share_percent = trader_share_percent  # NEW: Update split
    await db.commit()

    return JSONResponse({
        "success": True,
        "message": f"Eligibility updated: {min_trades} trades, {min_win_rate}% win rate, ${max_marketplace_price} max price, Trader share: {trader_share_percent}%"
    })


# ───── UPDATE UPLOAD LIMITS ─────
@router.post("/update_upload_limits")
async def update_upload_limits(
    starter_monthly: int = Form(2),
    starter_batch: int = Form(3),
    pro_monthly: int = Form(29),
    pro_batch: int = Form(10),
    elite_monthly: int = Form(1000),
    elite_batch: int = Form(10),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    plans_data = [
        {'plan': 'starter', 'monthly': starter_monthly, 'batch': starter_batch},
        {'plan': 'pro', 'monthly': pro_monthly, 'batch': pro_batch},
        {'plan': 'elite', 'monthly': elite_monthly, 'batch': elite_batch},
    ]

    for data in plans_data:
        await db.execute(update(UploadLimits).where(
            UploadLimits.plan == data['plan']
        ).values(
            monthly_limit=data['monthly'],
            batch_limit=data['batch']
        ))
        if not (await db.execute(select(UploadLimits).where(UploadLimits.plan == data['plan']))).scalar():
            db.add(UploadLimits(**data))

    await db.commit()
    return JSONResponse({"success": True, "message": "Upload limits updated"})


# ───── UPDATE INSIGHTS LIMITS ─────
@router.post("/update_insights_limits")
async def update_insights_limits(
    starter_monthly: int = Form(3),
    pro_monthly: int = Form(999),
    elite_monthly: int = Form(999),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    plans_data = [
        {'plan': 'starter', 'monthly': starter_monthly},
        {'plan': 'pro', 'monthly': pro_monthly},
        {'plan': 'elite', 'monthly': elite_monthly},
    ]

    for data in plans_data:
        await db.execute(update(InsightsLimits).where(
            InsightsLimits.plan == data['plan']
        ).values(monthly_limit=data['monthly']))
        if not (await db.execute(select(InsightsLimits).where(InsightsLimits.plan == data['plan']))).scalar():
            db.add(InsightsLimits(plan=data['plan'], monthly_limit=data['monthly']))

    await db.commit()
    return JSONResponse({"success": True, "message": "Insights generation limits updated"})


# ───── UPDATE AI CHAT LIMITS (NEW: Admin-configurable AI chat monthly limits and TP cost) ─────
@router.post("/update_ai_chat_limits")
async def update_ai_chat_limits(
    starter_monthly: int = Form(5, ge=0),
    starter_tp: int = Form(1, ge=0),
    pro_monthly: int = Form(25, ge=0),
    pro_tp: int = Form(0, ge=0),
    elite_monthly: int = Form(50, ge=0),
    elite_tp: int = Form(0, ge=0),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    plans_data = [
        {'plan': 'starter', 'monthly': starter_monthly, 'tp': starter_tp},
        {'plan': 'pro', 'monthly': pro_monthly, 'tp': pro_tp},
        {'plan': 'elite', 'monthly': elite_monthly, 'tp': elite_tp},
    ]

    for data in plans_data:
        await db.execute(update(AiChatLimits).where(
            AiChatLimits.plan == data['plan']
        ).values(
            monthly_limit=data['monthly'],
            tp_cost=data['tp']
        ))
        if not (await db.execute(select(AiChatLimits).where(AiChatLimits.plan == data['plan']))).scalar():
            db.add(AiChatLimits(
                plan=data['plan'],
                monthly_limit=data['monthly'],
                tp_cost=data['tp']
            ))

    await db.commit()
    return JSONResponse({"success": True, "message": "AI Chat limits updated"})


# ───── UPDATE INITIAL TP GRANT (NEW: Admin-configurable signup TP amount) ─────
@router.post("/update_initial_tp")
async def update_initial_tp(
    amount: int = Form(3, ge=0),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    if amount < 0:
        raise HTTPException(status_code=400, detail="Amount cannot be negative")

    result = await db.execute(select(InitialTpConfig).where(InitialTpConfig.id == 1))
    db_config = result.scalar_one_or_none()
    if not db_config:
        db_config = InitialTpConfig(id=1)
        db.add(db_config)

    db_config.amount = amount
    await db.commit()

    return JSONResponse({
        "success": True,
        "message": f"Initial TP grant updated to {amount} TP per new user signup"
    })


# ───── UPDATE UPGRADE TP CONFIG (NEW: Admin-configurable TP for plan upgrades) ─────
@router.post("/update_upgrade_tp")
async def update_upgrade_tp(
    pro_amount: int = Form(10, ge=0),
    elite_amount: int = Form(20, ge=0),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    if pro_amount < 0 or elite_amount < 0:
        raise HTTPException(status_code=400, detail="Amounts cannot be negative")

    configs = [
        {'id': 1, 'plan': 'pro', 'amount': pro_amount},
        {'id': 2, 'plan': 'elite', 'amount': elite_amount},
    ]

    for c in configs:
        await db.execute(update(UpgradeTpConfig).where(
            UpgradeTpConfig.id == c['id']
        ).values(amount=c['amount']))
        if not (await db.execute(select(UpgradeTpConfig).where(
            UpgradeTpConfig.id == c['id']
        ))).scalar():
            db.add(UpgradeTpConfig(**c))

    await db.commit()

    return JSONResponse({
        "success": True,
        "message": f"Upgrade TP updated: Pro {pro_amount} TP, Elite {elite_amount} TP"
    })


# ───── UPDATE BETA REFERRAL TP BONUS (NEW: Admin-configurable TP for beta referrals per plan) ─────
@router.post("/update_beta_referral_tp")
async def update_beta_referral_tp(
    starter_amount: int = Form(5, ge=0),
    pro_amount: int = Form(20, ge=0),
    elite_amount: int = Form(45, ge=0),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    if starter_amount < 0 or pro_amount < 0 or elite_amount < 0:
        raise HTTPException(status_code=400, detail="Amounts cannot be negative")

    result = await db.execute(select(BetaReferralTpConfig).where(BetaReferralTpConfig.id == 1))
    db_config = result.scalar_one_or_none()
    if not db_config:
        db_config = BetaReferralTpConfig(
            id=1,
            starter_tp=starter_amount,
            pro_tp=pro_amount,
            elite_tp=elite_amount
        )
        db.add(db_config)
    else:
        db_config.starter_tp = starter_amount
        db_config.pro_tp = pro_amount
        db_config.elite_tp = elite_amount

    await db.commit()

    return JSONResponse({
        "success": True,
        "message": f"Beta referral TP bonuses updated: Starter {starter_amount} TP, Pro {pro_amount} TP, Elite {elite_amount} TP"
    })


# ───── UPDATE USER PLAN ─────
@router.post("/update_plan/{user_id}")
async def update_user_plan(
    user_id: int,
    plan: str = Form(...),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    if plan not in ["starter", "pro", "elite"]:
        raise HTTPException(status_code=400, detail="Invalid plan")

    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.plan = plan
    user.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(user)

    return JSONResponse({"success": True, "message": f"Plan updated to {plan}"})


# ───── UPDATE MARKETPLACE PRICE ─────
@router.post("/update_marketplace_price/{user_id}")
async def update_marketplace_price(
    user_id: int,
    price: float = Form(...),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    if price < 0:
        raise HTTPException(status_code=400, detail="Price cannot be negative")

    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.is_trader:
        raise HTTPException(status_code=400, detail="User is not a trader")

    user.marketplace_price = price
    user.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(user)

    return JSONResponse({"success": True, "message": f"Price updated to ${price}"})


# ───── TOGGLE TRADER ─────
@router.post("/toggle_trader/{user_id}")
async def toggle_trader(
    user_id: int,
    desired_is_trader: bool = Form(...),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session),
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    config = (await db.execute(select(EligibilityConfig).where(EligibilityConfig.id == 1))).scalar_one_or_none()
    min_trades = config.min_trades if config else 50
    min_win_rate = config.min_win_rate if config else 80.0

    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    trade_query = select(Trade).where(Trade.owner_id == user_id)
    total = (await db.execute(select(func.count()).select_from(trade_query.subquery()))).scalar() or 0
    wins_count = (await db.execute(select(func.count()).select_from(trade_query.where(Trade.pnl > 0).subquery()))).scalar() or 0
    win_rate = round((wins_count / total * 100), 1) if total > 0 else 0.0
    user.win_rate = win_rate

    eligible = total >= min_trades and win_rate >= min_win_rate

    if desired_is_trader and not eligible:
        raise HTTPException(status_code=400, detail=f"User ineligible: Needs {min_trades - total} more trades and/or {min_win_rate - win_rate:.1f}% higher win rate.")

    was_trader = user.is_trader
    user.is_trader = desired_is_trader
    user.is_trader_pending = False
    user.updated_at = datetime.utcnow()

    if was_trader and not desired_is_trader:
        removal_notif = Notification(
            user_id=user.id,
            title="Removed from Marketplace",
            message="You have been removed from the marketplace. Please contact support if you believe this is an error.",
            type="removal"
        )
        db.add(removal_notif)
        logger.info(f"Removed trader {user.id} from marketplace and sent notification")

    await db.commit()
    await db.refresh(user)

    return JSONResponse({
        "success": True,
        "is_trader": user.is_trader,
        "win_rate": win_rate,
        "total_trades": total,
        "eligible": eligible,
        "min_trades": min_trades,
        "min_win_rate": min_win_rate,
    })


# ───── CREATE TEST MARKETPLACE SUBSCRIPTION ─────
@router.post("/create_test_subscription")
async def create_test_subscription(
    user_id: int = Form(...),
    trader_id: int = Form(...),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")
    
    existing = await db.execute(
        select(Subscription).where(
            Subscription.user_id == user_id,
            Subscription.trader_id == trader_id,
            Subscription.status == 'active'
        )
    )
    if existing.scalar():
        raise HTTPException(status_code=400, detail="Test subscription already active")
    
    user = await db.get(User, user_id)
    trader = await db.get(User, trader_id)
    if not user or not trader:
        raise HTTPException(status_code=404, detail="User or Trader not found")
    if not trader.is_trader:
        raise HTTPException(status_code=400, detail="Trader not eligible")
    
    test_sub = Subscription(
        user_id=user_id,
        trader_id=trader_id,
        plan_type=f"test_marketplace_{trader_id}_monthly",
        interval_days=30,
        amount_usd=0.0,
        status='active',
        start_date=datetime.utcnow(),
        next_billing_date=datetime.utcnow() + timedelta(days=30),
        order_id=f"test_{user_id}_{trader_id}",
        order_description=f"Test sub to {trader.full_name or trader.username}",
        renewal_url=None
    )
    db.add(test_sub)
    await db.commit()
    await db.refresh(test_sub)
    
    logger.info(f"Created test sub {test_sub.id} for user {user_id} to trader {trader_id}")
    return JSONResponse({
        "success": True,
        "message": f"Test access granted to {trader.full_name or 'Trader'}. User can now view journal via /journal?source=trader&trader_id={trader_id}",
        "sub_id": test_sub.id
    })


# ───── DELETE TEST MARKETPLACE SUBSCRIPTION ─────
@router.delete("/delete_test_subscription/{user_id}/{trader_id}")
async def delete_test_subscription(
    user_id: int,
    trader_id: int,
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")
    
    result = await db.execute(
        select(Subscription).where(
            Subscription.user_id == user_id,
            Subscription.trader_id == trader_id,
            Subscription.status == 'active'
        )
    )
    sub = result.scalar_one_or_none()
    if not sub:
        raise HTTPException(status_code=404, detail="No active test subscription found")
    
    await db.delete(sub)
    await db.commit()
    
    logger.info(f"Deleted test sub {sub.id} for user {user_id} to trader {trader_id}")
    return JSONResponse({
        "success": True,
        "message": f"Test access revoked for trader {trader_id}."
    })


# ───── APPROVE TRADER APPLICATION ─────
@router.post("/approve_trader/{user_id}")
async def approve_trader(
    user_id: int,
    approve: bool = Form(...),
    reason: Optional[str] = Form(None),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    applicant = await db.get(User, user_id)
    if not applicant or not applicant.is_trader_pending:
        raise HTTPException(status_code=404, detail="No pending application found")

    applicant.is_trader_pending = False
    if approve:
        applicant.is_trader = True
        logger.info(f"Approved trader application for user {applicant.id}")
        message = f"Approved: {applicant.full_name or applicant.email}"
        
        approval_notif = Notification(
            user_id=applicant.id,
            title="Trader Application Approved!",
            message=f"Congratulations! Your application to become a marketplace trader has been approved. You can now set your subscription price and start earning. Visit your profile to get started.",
            type="approval"
        )
        db.add(approval_notif)
    else:
        applicant.is_trader = False
        applicant.marketplace_price = None
        logger.info(f"Rejected trader application for user {applicant.id}: {reason}")
        message = f"Rejected: {applicant.full_name or applicant.email} ({reason})"
        
        rejection_notif = Notification(
            user_id=applicant.id,
            title="Trader Application Update",
            message=f"Your application to become a marketplace trader has been reviewed. Unfortunately, it was not approved at this time. Reason: {reason or 'Did not meet current eligibility criteria.'} Feel free to improve your stats and reapply!",
            type="rejection"
        )
        db.add(rejection_notif)

    applicant.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(applicant)

    return JSONResponse({"success": True, "message": message})


# ───── MANUAL COMPLETE PAYMENT ─────
@router.post("/complete-payment")
async def manual_complete_payment(
    payload: dict = Body(...),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    payment_id = payload.get("payment_id")
    reason = payload.get("reason", "Test/manual override")

    result = await db.execute(select(Payment).where(Payment.nowpayments_payment_id == payment_id))
    db_payment = result.scalar_one_or_none()
    if not db_payment:
        raise HTTPException(status_code=404, detail="Payment not found")
    
    if db_payment.status not in ["partially_paid", "failed"]:
        raise HTTPException(status_code=400, detail="Only partial/failed payments can be completed")
    
    if not db_payment.subscription_id:
        raise HTTPException(status_code=400, detail="No linked subscription")
    
    result = await db.execute(select(Subscription).where(Subscription.id == db_payment.subscription_id))
    db_sub = result.scalar_one_or_none()
    if not db_sub:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    db_payment.status = "finished_manual"
    db_payment.notes = reason
    db_payment.updated_at = datetime.utcnow()
    
    db_sub.status = "active"
    if "_renew" in (db_payment.order_id or ""):
        db_sub.next_billing_date += timedelta(days=db_sub.interval_days)
    else:
        db_sub.next_billing_date = db_sub.start_date + timedelta(days=db_sub.interval_days)
    db_sub.renewal_url = None
    db_sub.updated_at = datetime.utcnow()
    
    result = await db.execute(select(User).where(User.id == db_sub.user_id))
    user = result.scalar_one_or_none()
    if user and db_sub.trader_id is None:  # Only update for PLATFORM subs (no trader_id)
        # Normalize to base plan (e.g., 'pro_monthly' -> 'pro' for AiChatLimits matching)
        base_plan = db_sub.plan_type.lower().split('_')[0] if '_' in db_sub.plan_type.lower() else db_sub.plan_type.lower()
        user.plan = base_plan  # e.g., 'pro' instead of 'pro_monthly'
        user.updated_at = datetime.utcnow()
    
    await db.commit()
    
    logger.info(f"Manually completed payment {payment_id} for sub {db_sub.id}: {reason}")
    return {"success": True, "message": f"Payment {payment_id} completed! Subscription {db_sub.id} activated for user {db_sub.user_id}."}
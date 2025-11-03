# router/admin.py
import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Request, Depends, HTTPException, status, Form, Body
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, update
from sqlalchemy.orm import joinedload
from typing import Optional

from templates_config import templates
from models.models import (
    User, Trade, Subscription, Payment, Pricing, Discount, EligibilityConfig, UploadLimits
)
from database import get_session
import auth

logger = logging.getLogger("iTrade")

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("", response_class=HTMLResponse)
async def admin_page(
    request: Request,
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

    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    one_month_ago = now - timedelta(days=30)

    # ───── Stats ─────
    total_users = (await db.execute(select(func.count()).select_from(User))).scalar() or 0
    new_users = (await db.execute(select(func.count()).select_from(User).where(User.created_at >= week_ago))).scalar() or 0
    total_trades = (await db.execute(select(func.count()).select_from(Trade))).scalar() or 0
    avg_trades_per_user = round(total_trades / total_users, 1) if total_users else 0

    plan_counts = {'starter': 0, 'pro': 0, 'elite': 0}
    if total_users > 0:
        result_plans = await db.execute(select(User.plan, func.count()).group_by(User.plan))
        for plan, count in result_plans.all():
            if plan:
                plan_counts[plan] = count

    # ───── Pricing ─────
    pricing = {'pro_monthly': 9.99, 'pro_yearly': 99.0, 'elite_monthly': 19.99, 'elite_yearly': 199.0}
    result_pricing = await db.execute(select(Pricing).where(
        Pricing.plan.in_(['pro', 'elite']),
        Pricing.interval.in_(['monthly', 'yearly'])
    ))
    for p in result_pricing.scalars().all():
        key = f"{p.plan}_{p.interval}"
        pricing[key] = p.amount

    # ───── Platform Discount (id=1) ─────
    discount = {'enabled': False, 'percentage': 0.0, 'expiry': ''}
    result_discount = await db.execute(select(Discount).where(Discount.id == 1))
    db_discount = result_discount.scalar_one_or_none()
    if db_discount:
        discount['enabled'] = db_discount.enabled
        discount['percentage'] = db_discount.percentage
        discount['expiry'] = db_discount.expiry.strftime('%Y-%m-%d') if db_discount.expiry else ''
    else:
        db_discount = Discount(id=1, enabled=False, percentage=0.0, expiry=None)
        db.add(db_discount)
        await db.commit()

    # ───── Marketplace Discount (id=2) ─────
    marketplace_discount = {'enabled': False, 'percentage': 0.0, 'expiry': ''}
    result_marketplace_discount = await db.execute(select(Discount).where(Discount.id == 2))
    db_marketplace_discount = result_marketplace_discount.scalar_one_or_none()
    if db_marketplace_discount:
        marketplace_discount['enabled'] = db_marketplace_discount.enabled
        marketplace_discount['percentage'] = db_marketplace_discount.percentage
        marketplace_discount['expiry'] = db_marketplace_discount.expiry.strftime('%Y-%m-%d') if db_marketplace_discount.expiry else ''
    else:
        db_marketplace_discount = Discount(id=2, enabled=False, percentage=0.0, expiry=None)
        db.add(db_marketplace_discount)
        await db.commit()

    # ───── Eligibility Config ─────
    result_config = await db.execute(select(EligibilityConfig).where(EligibilityConfig.id == 1))
    db_config = result_config.scalar_one_or_none()
    if not db_config:
        db_config = EligibilityConfig(id=1, min_trades=50, min_win_rate=80.0, max_marketplace_price=99.99)
        db.add(db_config)
        await db.commit()

    eligibility = {
        'min_trades': db_config.min_trades,
        'min_win_rate': db_config.min_win_rate,
        'max_marketplace_price': db_config.max_marketplace_price,  # NEW
    }

    # ───── Upload Limits ─────
    upload_limits = {}
    plans = ['starter', 'pro', 'elite']
    for plan in plans:
        result_limit = await db.execute(select(UploadLimits).where(UploadLimits.plan == plan))
        db_limit = result_limit.scalar_one_or_none()
        if not db_limit:
            # Initialize with updated defaults
            if plan == 'starter':
                monthly_default = 2
                batch_default = 3
            elif plan == 'pro':
                monthly_default = 29
                batch_default = 10
            else:  # elite
                monthly_default = 1000
                batch_default = 10
            db_limit = UploadLimits(
                plan=plan,
                monthly_limit=monthly_default,
                batch_limit=batch_default
            )
            db.add(db_limit)
        upload_limits[plan] = {
            'monthly_limit': db_limit.monthly_limit,
            'batch_limit': db_limit.batch_limit
        }
    await db.commit()

    # ───── Revenue (FIXED: Use 'finished' instead of 'paid'; case-insensitive)
    active_subscribers = (await db.execute(select(func.count()).select_from(Subscription).where(Subscription.status == 'active'))).scalar() or 0
    active_subs = (await db.execute(select(Subscription).where(Subscription.status == 'active'))).scalars().all()
    mrr = sum(sub.amount_usd / (12 if sub.interval_days > 30 else 1) for sub in active_subs)

    # FIXED: Query for 'finished' (success status) case-insensitively
    monthly_revenue = (await db.execute(select(func.sum(Payment.amount_usd)).select_from(Payment).where(
        func.lower(Payment.status) == 'finished',  # Changed from 'paid' to 'finished'
        Payment.paid_at >= one_month_ago
    ))).scalar() or 0.0
    arpu = monthly_revenue / total_users if total_users else 0.0

    past_active_subs_count = (await db.execute(select(func.count()).select_from(Subscription).where(
        Subscription.status == 'active', Subscription.start_date <= one_month_ago
    ))).scalar() or 0
    canceled_last_month = (await db.execute(select(func.count()).select_from(Subscription).where(
        Subscription.status == 'canceled', Subscription.updated_at >= one_month_ago
    ))).scalar() or 0
    user_churn_rate = (canceled_last_month / past_active_subs_count * 100) if past_active_subs_count else 0.0

    past_active_subs = (await db.execute(select(Subscription).where(
        Subscription.status == 'active', Subscription.start_date <= one_month_ago
    ))).scalars().all()
    past_mrr = sum(sub.amount_usd / (12 if sub.interval_days > 30 else 1) for sub in past_active_subs)
    lost_mrr = max(0, past_mrr - mrr)
    revenue_churn_rate = (lost_mrr / past_mrr * 100) if past_mrr else 0.0

    # ───── NEW: Admin's Own Plan Details ─────
    admin_plan_display = current_user.plan.lower() if current_user.plan else 'starter'
    if 'pro' in admin_plan_display:
        admin_plan_display = 'Pro'
    elif 'elite' in admin_plan_display:
        admin_plan_display = 'Elite'
    else:
        admin_plan_display = 'Starter'

    # Fetch admin's active subscription - FIXED: Use .first() to avoid MultipleResultsFound
    admin_active_sub_result = await db.execute(
        select(Subscription).where(
            Subscription.user_id == current_user.id,
            Subscription.status == 'active'
        ).order_by(desc(Subscription.start_date)).limit(1)
    )
    admin_active_sub = admin_active_sub_result.scalar_one_or_none()

    # ───── Recent Users ─────
    recent_users = (await db.execute(select(User).order_by(desc(User.created_at)).limit(10))).scalars().all()
    recent_users_with_stats = []
    for user in recent_users:
        trade_query = select(Trade).where(Trade.owner_id == user.id)
        total_res = await db.execute(select(func.count()).select_from(trade_query.subquery()))
        trade_count = total_res.scalar() or 0
        wins_res = await db.execute(select(func.count()).select_from(trade_query.where(Trade.pnl > 0).subquery()))
        wins_count = wins_res.scalar() or 0
        win_rate = round((wins_count / trade_count * 100), 1) if trade_count > 0 else 0.0
        joined_formatted = user.created_at.strftime('%b %d, %Y') if user.created_at else 'N/A'
        plan = getattr(user, 'plan', 'starter')
        
        # UPDATED: Fetch active test marketplace subs (amount=0)
        test_subs_query = select(Subscription.trader_id, User.full_name).select_from(Subscription).join(User, Subscription.trader_id == User.id).where(
            Subscription.user_id == user.id,
            Subscription.status == 'active',
            Subscription.amount_usd == 0.0
        )
        result_test_subs = await db.execute(test_subs_query)
        active_test_subs = result_test_subs.all()
        sub_list = [
            {'trader_id': row[0], 'trader_name': row[1] or f'Trader {row[0]}'}
            for row in active_test_subs
        ]
        sub_count = len(sub_list)
        
        recent_users_with_stats.append({
            'id': user.id,
            'email': user.email,
            'full_name': user.full_name or 'N/A',
            'joined': joined_formatted,
            'trades': trade_count,
            'plan': plan.upper() if plan else 'STARTER',
            'is_trader': getattr(user, 'is_trader', False),
            'is_trader_pending': getattr(user, 'is_trader_pending', False),  # NEW
            'win_rate': win_rate,
            'sub_count': sub_count,
            'sub_list': sub_list,  # NEW: List of active test subs
        })

    # ───── Marketplace Traders ─────
    marketplace_traders = (await db.execute(
        select(User).where(User.is_trader == True).order_by(desc(User.created_at)).limit(20)
    )).scalars().all()
    marketplace_traders_with_stats = []
    for trader in marketplace_traders:
        trade_query = select(func.count(Trade.id)).where(Trade.owner_id == trader.id)
        total_res = await db.execute(trade_query)
        trade_count = total_res.scalar() or 0
        marketplace_traders_with_stats.append({
            'id': trader.id,
            'name': trader.full_name or f'Trader {trader.id}',
            'win_rate': round(trader.win_rate or 0, 1),
            'trades': trade_count,
            'price': trader.marketplace_price or 19.99
        })

    # ───── NEW: Pending Trader Applications ─────
    pending_traders = (await db.execute(
        select(User).where(User.is_trader_pending == True).order_by(desc(User.created_at)).limit(10)
    )).scalars().all()
    pending_traders_with_stats = []
    for trader in pending_traders:
        trade_query = select(func.count(Trade.id)).where(Trade.owner_id == trader.id)
        total_res = await db.execute(trade_query)
        trade_count = total_res.scalar() or 0
        wins_res = await db.execute(select(func.count(Trade.id)).where(Trade.owner_id == trader.id, Trade.pnl > 0))
        wins_count = wins_res.scalar() or 0
        win_rate = round((wins_count / trade_count * 100), 1) if trade_count > 0 else 0.0
        pending_traders_with_stats.append({
            'id': trader.id,
            'name': trader.full_name or f'Trader {trader.id}',
            'email': trader.email,
            'win_rate': win_rate,
            'trades': trade_count,
            'applied_at': trader.updated_at.strftime('%b %d, %Y %H:%M') if trader.updated_at else 'N/A'
        })

    # ───── Recent Trades ─────
    recent_trades = (await db.execute(select(Trade).order_by(desc(Trade.created_at)).limit(10))).scalars().all()
    recent_trades_list = []
    for trade in recent_trades:
        user = (await db.execute(select(User).where(User.id == trade.owner_id))).scalar()
        user_name = user.full_name if user else 'Unknown'
        created_formatted = trade.created_at.strftime('%b %d, %Y') if trade.created_at else 'N/A'
        recent_trades_list.append({
            'id': trade.id,
            'user_name': user_name,
            'symbol': trade.symbol or 'N/A',
            'pnl': trade.pnl or 0,
            'created': created_formatted
        })

    # ───── NEW: Recent Partial Payments (FIXED: Case-insensitive query)
    partial_payments = (await db.execute(
        select(Payment).where(func.lower(Payment.status) == 'partially_paid').order_by(desc(Payment.created_at)).limit(10)
    )).scalars().all()
    recent_partial_payments = []
    for p in partial_payments:
        sub_result = await db.execute(select(Subscription).where(Subscription.id == p.subscription_id))
        sub = sub_result.scalar_one_or_none()
        if not sub:
            continue
        user = await db.get(User, sub.user_id)
        trader = await db.get(User, sub.trader_id) if sub.trader_id else None
        trader_name = trader.full_name or trader.username if trader else None
        is_marketplace = bool(sub.trader_id)
        created_formatted = p.created_at.strftime('%b %d, %Y %H:%M') if p.created_at else 'N/A'
        recent_partial_payments.append({
            'id': p.id,
            'nowpayments_payment_id': p.nowpayments_payment_id,
            'user_email': user.email if user else 'Unknown',
            'user_id': sub.user_id,
            'trader_name': trader_name,
            'is_marketplace': is_marketplace,
            'amount_usd': p.amount_usd,
            'amount_paid_crypto': p.amount_paid_crypto,
            'crypto_currency': p.crypto_currency,
            'created': created_formatted,
        })

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "current_user": current_user,
            "initials": initials,
            "total_users": total_users,
            "new_users": new_users,
            "total_trades": total_trades,
            "avg_trades_per_user": avg_trades_per_user,
            "plan_counts": plan_counts,
            "recent_users": recent_users_with_stats,
            "marketplace_traders": marketplace_traders_with_stats,
            "pending_traders": pending_traders_with_stats,
            "recent_trades": recent_trades_list,
            "recent_partial_payments": recent_partial_payments,
            "pricing": pricing,
            "discount": discount,
            "marketplace_discount": marketplace_discount,
            "eligibility": eligibility,
            "upload_limits": upload_limits,
            "mrr": round(mrr, 2),
            "monthly_revenue": round(monthly_revenue, 2),
            "arpu": round(arpu, 2),
            "active_subscribers": active_subscribers,
            "user_churn_rate": round(user_churn_rate, 1),
            "revenue_churn_rate": round(revenue_churn_rate, 1),
            "admin_plan": admin_plan_display,
            "admin_sub": admin_active_sub,
        }
    )


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


# ───── UPDATE ELIGIBILITY ─────
@router.post("/update_eligibility")
async def update_eligibility(
    min_trades: int = Form(50),
    min_win_rate: float = Form(80.0),
    max_marketplace_price: float = Form(99.99),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    if min_trades < 1 or min_win_rate < 0 or min_win_rate > 100 or max_marketplace_price < 0:
        raise HTTPException(status_code=400, detail="Invalid thresholds")

    result = await db.execute(select(EligibilityConfig).where(EligibilityConfig.id == 1))
    db_config = result.scalar_one_or_none()
    if not db_config:
        db_config = EligibilityConfig(id=1)
        db.add(db_config)

    db_config.min_trades = min_trades
    db_config.min_win_rate = min_win_rate
    db_config.max_marketplace_price = max_marketplace_price
    await db.commit()

    return JSONResponse({
        "success": True,
        "message": f"Eligibility updated: {min_trades} trades, {min_win_rate}% win rate, ${max_marketplace_price} max price"
    })


# ───── NEW: UPDATE UPLOAD LIMITS ─────
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

    user.is_trader = desired_is_trader
    user.is_trader_pending = False  # Clear pending on toggle
    user.updated_at = datetime.utcnow()
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

# ───── NEW: CREATE TEST MARKETPLACE SUBSCRIPTION ─────
@router.post("/create_test_subscription")
async def create_test_subscription(
    user_id: int = Form(...),
    trader_id: int = Form(...),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Check if sub already exists
    existing = await db.execute(
        select(Subscription).where(
            Subscription.user_id == user_id,
            Subscription.trader_id == trader_id,
            Subscription.status == 'active'
        )
    )
    if existing.scalar():
        raise HTTPException(status_code=400, detail="Test subscription already active")
    
    # Verify user and trader exist
    user = await db.get(User, user_id)
    trader = await db.get(User, trader_id)
    if not user or not trader:
        raise HTTPException(status_code=404, detail="User or Trader not found")
    if not trader.is_trader:
        raise HTTPException(status_code=400, detail="Trader not eligible")
    
    # Create fake active sub (monthly, $0 for test)
    test_sub = Subscription(
        user_id=user_id,
        trader_id=trader_id,
        plan_type=f"test_marketplace_{trader_id}_monthly",
        interval_days=30,
        amount_usd=0.0,  # Free for test
        status='active',
        start_date=datetime.utcnow(),
        next_billing_date=datetime.utcnow() + timedelta(days=30),
        order_id=f"test_{user_id}_{trader_id}",
        order_description=f"Test sub to {trader.full_name or trader.username}",
        renewal_url=None  # No real payment
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

# ───── NEW: DELETE TEST MARKETPLACE SUBSCRIPTION ─────
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

# ───── NEW: Approve/Reject Trader Application ─────
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
    else:
        applicant.is_trader = False
        applicant.marketplace_price = None
        logger.info(f"Rejected trader application for user {applicant.id}: {reason}")
        message = f"Rejected: {applicant.full_name or applicant.email} ({reason})"

    applicant.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(applicant)

    return JSONResponse({"success": True, "message": message})

# ───── NEW: Manual Complete Payment ─────
@router.post("/complete-payment")
async def manual_complete_payment(
    payment_id: str = Body(...),  # NowPayments payment_id
    reason: str = Body("Test/manual override"),
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or current_user.email != "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")

    # Fetch payment
    result = await db.execute(select(Payment).where(Payment.nowpayments_payment_id == payment_id))
    db_payment = result.scalar_one_or_none()
    if not db_payment:
        raise HTTPException(status_code=404, detail="Payment not found")
    
    # Verify status
    if db_payment.status not in ["partially_paid", "failed"]:
        raise HTTPException(status_code=400, detail="Only partial/failed payments can be completed")
    
    # Link to sub if needed
    if not db_payment.subscription_id:
        raise HTTPException(status_code=400, detail="No linked subscription")
    
    result = await db.execute(select(Subscription).where(Subscription.id == db_payment.subscription_id))
    db_sub = result.scalar_one_or_none()
    if not db_sub:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    # Complete payment
    db_payment.status = "finished_manual"
    db_payment.notes = reason
    db_payment.updated_at = datetime.utcnow()
    
    # Activate sub
    db_sub.status = "active"
    if "_renew" in (db_payment.order_id or ""):
        db_sub.next_billing_date += timedelta(days=db_sub.interval_days)
    else:
        db_sub.next_billing_date = db_sub.start_date + timedelta(days=db_sub.interval_days)
    db_sub.renewal_url = None
    db_sub.updated_at = datetime.utcnow()
    
    # Update user
    result = await db.execute(select(User).where(User.id == db_sub.user_id))
    user = result.scalar_one_or_none()
    if user:
        user.plan = db_sub.plan_type
        user.updated_at = datetime.utcnow()
    
    await db.commit()
    
    logger.info(f"Manually completed payment {payment_id} for sub {db_sub.id}: {reason}")
    return {"success": True, "message": f"Payment {payment_id} completed! Subscription {db_sub.id} activated for user {db_sub.user_id}."}
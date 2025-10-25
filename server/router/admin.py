import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Request, Depends, HTTPException, status, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, update

from templates_config import templates
from models import models
from database import get_session
import auth

logger = logging.getLogger("iTrade")

router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("", response_class=HTMLResponse)
async def admin_page(
    request: Request,
    current_user: models.User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    if not current_user.email == "ukovictor8@gmail.com":  # Placeholder: replace with real admin logic
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
    
    result_users = await db.execute(select(func.count()).select_from(models.User))
    total_users = result_users.scalar() or 0
    
    result_new_users = await db.execute(
        select(func.count()).select_from(models.User).where(models.User.created_at >= week_ago)
    )
    new_users = result_new_users.scalar() or 0
    
    result_trades = await db.execute(select(func.count()).select_from(models.Trade))
    total_trades = result_trades.scalar() or 0
    
    avg_trades_per_user = round(total_trades / total_users, 1) if total_users else 0
    
    plan_counts = {'starter': 0, 'pro': 0, 'elite': 0}
    if total_users > 0:
        result_plans = await db.execute(
            select(models.User.plan, func.count()).group_by(models.User.plan).select_from(models.User)
        )
        for plan, count in result_plans.all():
            if plan:
                plan_counts[plan] = count
    
    pricing = {
        'pro_monthly': 9.99,
        'pro_yearly': 99.0,
        'elite_monthly': 19.99,
        'elite_yearly': 199.0
    }
    result_pricing = await db.execute(
        select(models.Pricing).where(
            models.Pricing.plan.in_(['pro', 'elite']),
            models.Pricing.interval.in_(['monthly', 'yearly'])
        )
    )
    for p in result_pricing.scalars().all():
        key = f"{p.plan}_{p.interval}"
        pricing[key] = p.amount

    discount = {
        'enabled': False,
        'percentage': 0.0,
        'expiry': ''
    }
    result_discount = await db.execute(
        select(models.Discount).where(models.Discount.id == 1)
    )
    db_discount = result_discount.scalar_one_or_none()
    if db_discount:
        discount['enabled'] = db_discount.enabled
        discount['percentage'] = db_discount.percentage
        discount['expiry'] = db_discount.expiry.strftime('%Y-%m-%d') if db_discount.expiry else ''

    result_active_subs = await db.execute(
        select(func.count()).select_from(models.Subscription).where(models.Subscription.status == 'active')
    )
    active_subscribers = result_active_subs.scalar() or 0

    result_active_subs_data = await db.execute(
        select(models.Subscription).where(models.Subscription.status == 'active')
    )
    active_subs = result_active_subs_data.scalars().all()
    mrr = 0.0
    for sub in active_subs:
        if sub.interval_days == 30:
            mrr += sub.amount_usd
        else:
            mrr += sub.amount_usd / 12

    result_monthly_rev = await db.execute(
        select(func.sum(models.Payment.amount_usd)).select_from(models.Payment).where(
            models.Payment.status == 'paid',
            models.Payment.paid_at >= one_month_ago
        )
    )
    monthly_revenue = result_monthly_rev.scalar() or 0.0

    arpu = monthly_revenue / total_users if total_users else 0.0

    result_past_active_count = await db.execute(
        select(func.count()).select_from(models.Subscription).where(
            models.Subscription.status == 'active',
            models.Subscription.start_date <= one_month_ago
        )
    )
    past_active_subs_count = result_past_active_count.scalar() or 0

    result_canceled_last_month = await db.execute(
        select(func.count()).select_from(models.Subscription).where(
            models.Subscription.status == 'canceled',
            models.Subscription.updated_at >= one_month_ago
        )
    )
    canceled_last_month = result_canceled_last_month.scalar() or 0

    user_churn_rate = (canceled_last_month / past_active_subs_count * 100) if past_active_subs_count else 0.0

    result_past_active_subs_data = await db.execute(
        select(models.Subscription).where(
            models.Subscription.status == 'active',
            models.Subscription.start_date <= one_month_ago
        )
    )
    past_active_subs = result_past_active_subs_data.scalars().all()
    past_mrr = 0.0
    for sub in past_active_subs:
        if sub.interval_days == 30:
            past_mrr += sub.amount_usd
        else:
            past_mrr += sub.amount_usd / 12

    lost_mrr = max(0, past_mrr - mrr)
    revenue_churn_rate = (lost_mrr / past_mrr * 100) if past_mrr else 0.0

    recent_users_result = await db.execute(
        select(models.User).order_by(desc(models.User.created_at)).limit(10)
    )
    recent_users = recent_users_result.scalars().all()
    
    recent_users_with_stats = []
    for user in recent_users:
        trade_count_result = await db.execute(
            select(func.count()).select_from(models.Trade).where(models.Trade.owner_id == user.id)
        )
        trade_count = trade_count_result.scalar() or 0
        joined_formatted = user.created_at.strftime('%b %d, %Y') if user.created_at else 'N/A'
        plan = getattr(user, 'plan', 'starter')
        recent_users_with_stats.append({
            'id': user.id,
            'email': user.email,
            'full_name': user.full_name or 'N/A',
            'joined': joined_formatted,
            'trades': trade_count,
            'plan': plan.upper() if plan else 'STARTER'
        })
    
    recent_trades_result = await db.execute(
        select(models.Trade).order_by(desc(models.Trade.created_at)).limit(10)
    )
    recent_trades = recent_trades_result.scalars().all()
    
    recent_trades_list = []
    for trade in recent_trades:
        user_result = await db.execute(select(models.User).where(models.User.id == trade.owner_id))
        user = user_result.scalar()
        user_name = user.full_name if user else 'Unknown'
        created_formatted = trade.created_at.strftime('%b %d, %Y') if trade.created_at else 'N/A'
        recent_trades_list.append({
            'id': trade.id,
            'user_name': user_name,
            'symbol': trade.symbol or 'N/A',
            'pnl': trade.pnl or 0,
            'created': created_formatted
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
            "recent_trades": recent_trades_list,
            "pricing": pricing,
            "discount": discount,
            "mrr": mrr,
            "monthly_revenue": monthly_revenue,
            "arpu": arpu,
            "active_subscribers": active_subscribers,
            "user_churn_rate": user_churn_rate,
            "revenue_churn_rate": revenue_churn_rate,
        }
    )

@router.post("/update_pricing")
async def update_pricing(
    pro_monthly: float = Form(...),
    pro_yearly: float = Form(...),
    elite_monthly: float = Form(...),
    elite_yearly: float = Form(...),
    current_user: models.User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or not current_user.email == "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")
    
    prices = [
        {'plan': 'pro', 'interval': 'monthly', 'amount': pro_monthly},
        {'plan': 'pro', 'interval': 'yearly', 'amount': pro_yearly},
        {'plan': 'elite', 'interval': 'monthly', 'amount': elite_monthly},
        {'plan': 'elite', 'interval': 'yearly', 'amount': elite_yearly},
    ]
    
    for p in prices:
        stmt = update(models.Pricing).where(
            models.Pricing.plan == p['plan'],
            models.Pricing.interval == p['interval']
        ).values(amount=p['amount'])
        await db.execute(stmt)
        result = await db.execute(
            select(models.Pricing).where(
                models.Pricing.plan == p['plan'],
                models.Pricing.interval == p['interval']
            )
        )
        if not result.scalar():
            new_price = models.Pricing(**p)
            db.add(new_price)
    await db.commit()
    
    return JSONResponse({"success": True, "message": "Pricing updated"})

@router.post("/update_discount")
async def update_discount(
    discount_enabled: bool = Form(False),
    discount_percentage: float = Form(0.0),
    discount_expiry: str = Form(None),
    current_user: models.User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or not current_user.email == "victor@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")
    
    expiry_date = None
    if discount_expiry:
        try:
            expiry_date = datetime.strptime(discount_expiry, '%Y-%m-%d').date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid expiry date format")
    
    result = await db.execute(select(models.Discount).where(models.Discount.id == 1))
    db_discount = result.scalar_one_or_none()
    
    if db_discount:
        db_discount.enabled = discount_enabled
        db_discount.percentage = discount_percentage if discount_enabled else 0.0
        db_discount.expiry = expiry_date if discount_enabled else None
    else:
        db_discount = models.Discount(
            id=1,
            enabled=discount_enabled,
            percentage=discount_percentage if discount_enabled else 0.0,
            expiry=expiry_date if discount_enabled else None
        )
        db.add(db_discount)
    
    await db.commit()
    return JSONResponse({"success": True, "message": "Discount updated"})

@router.post("/update_plan/{user_id}")
async def update_user_plan(
    user_id: int,
    plan: str = Form(...),
    current_user: models.User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user or not current_user.email == "ukovictor8@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")
    
    if plan not in ["starter", "pro", "elite"]:
        raise HTTPException(status_code=400, detail="Invalid plan")
    
    result = await db.execute(select(models.User).where(models.User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.plan = plan
    user.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(user)
    
    return JSONResponse({"success": True, "message": f"Plan updated to {plan}"})
# server/app_utils/admin_utils.py
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from sqlalchemy import select, func, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from models.models import (
    User, Trade, Subscription, Payment, Pricing, Discount, EligibilityConfig, UploadLimits, Notification, InsightsLimits,
    Referral, PointTransaction, InitialTpConfig, UpgradeTpConfig, AiChatLimits, BetaInvite, BetaConfig, BetaReferralTpConfig
)

logger = logging.getLogger("iTrade")


async def compute_admin_stats(db: AsyncSession) -> Dict[str, Any]:
    """Compute core admin statistics."""
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)

    total_users = (await db.execute(select(func.count()).select_from(User))).scalar() or 0
    new_users = (await db.execute(select(func.count()).select_from(User).where(User.created_at >= week_ago))).scalar() or 0
    total_trades = (await db.execute(select(func.count()).select_from(Trade))).scalar() or 0
    avg_trades_per_user = round(total_trades / total_users, 1) if total_users else 0

    total_referrals = (await db.execute(select(func.count()).select_from(Referral))).scalar() or 0
    total_points_issued = (await db.execute(select(func.sum(PointTransaction.amount)).where(PointTransaction.amount > 0))).scalar() or 0

    total_beta_invites = (await db.execute(select(func.count()).select_from(BetaInvite))).scalar() or 0
    used_beta_invites = (await db.execute(select(func.count()).select_from(BetaInvite).where(BetaInvite.used_by_id.is_not(None)))).scalar() or 0

    return {
        "total_users": total_users,
        "new_users": new_users,
        "total_trades": total_trades,
        "avg_trades_per_user": avg_trades_per_user,
        "total_referrals": total_referrals,
        "total_points_issued": total_points_issued,
        "total_beta_invites": total_beta_invites,
        "used_beta_invites": used_beta_invites,
    }


async def get_all_configs(db: AsyncSession) -> Dict[str, Any]:
    """Fetch and initialize all admin configurations."""
    # Beta Config
    beta_config = await db.execute(select(BetaConfig).where(BetaConfig.id == 1))
    db_beta_config = beta_config.scalar_one_or_none()
    if not db_beta_config:
        db_beta_config = BetaConfig(id=1, is_active=True, required_for_signup=True, award_points_on_use=3)
        db.add(db_beta_config)
        await db.commit()
    beta_settings = {
        'is_active': db_beta_config.is_active,
        'required_for_signup': db_beta_config.required_for_signup,
        'award_points_on_use': db_beta_config.award_points_on_use
    }

    # Plan counts
    total_users = (await db.execute(select(func.count()).select_from(User))).scalar() or 0
    plan_counts = {'starter': 0, 'pro': 0, 'elite': 0}
    if total_users > 0:
        result_plans = await db.execute(select(User.plan, func.count()).group_by(User.plan))
        for plan, count in result_plans.all():
            if plan:
                plan_counts[plan] = count

    # Initial TP Config
    result_initial_tp = await db.execute(select(InitialTpConfig).where(InitialTpConfig.id == 1))
    db_initial_tp = result_initial_tp.scalar_one_or_none()
    if not db_initial_tp:
        db_initial_tp = InitialTpConfig(id=1, amount=3)
        db.add(db_initial_tp)
        await db.commit()
    initial_tp = {'amount': db_initial_tp.amount}

    # Upgrade TP Config
    result_pro = await db.execute(select(UpgradeTpConfig).where(UpgradeTpConfig.id == 1))
    db_pro = result_pro.scalar_one_or_none()
    if not db_pro:
        db_pro = UpgradeTpConfig(id=1, plan='pro', amount=10)
        db.add(db_pro)

    result_elite = await db.execute(select(UpgradeTpConfig).where(UpgradeTpConfig.id == 2))
    db_elite = result_elite.scalar_one_or_none()
    if not db_elite:
        db_elite = UpgradeTpConfig(id=2, plan='elite', amount=20)
        db.add(db_elite)

    upgrade_tp = {'pro': db_pro.amount, 'elite': db_elite.amount}

    # Pricing
    pricing = {'pro_monthly': 9.99, 'pro_yearly': 99.0, 'elite_monthly': 19.99, 'elite_yearly': 199.0}
    result_pricing = await db.execute(select(Pricing).where(
        Pricing.plan.in_(['pro', 'elite']),
        Pricing.interval.in_(['monthly', 'yearly'])
    ))
    for p in result_pricing.scalars().all():
        key = f"{p.plan}_{p.interval}"
        pricing[key] = p.amount

    # Platform Discount (id=1)
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

    # Marketplace Discount (id=2)
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

    # Eligibility Config
    result_config = await db.execute(select(EligibilityConfig).where(EligibilityConfig.id == 1))
    db_config = result_config.scalar_one_or_none()
    if not db_config:
        db_config = EligibilityConfig(id=1, min_trades=50, min_win_rate=80.0, max_marketplace_price=99.99, trader_share_percent=70.0)
        db.add(db_config)
        await db.commit()

    eligibility = {
        'min_trades': db_config.min_trades,
        'min_win_rate': db_config.min_win_rate,
        'max_marketplace_price': db_config.max_marketplace_price,
        'trader_share_percent': db_config.trader_share_percent or 70.0,
    }

    # Upload Limits
    upload_limits = {}
    plans = ['starter', 'pro', 'elite']
    for plan in plans:
        result_limit = await db.execute(select(UploadLimits).where(UploadLimits.plan == plan))
        db_limit = result_limit.scalar_one_or_none()
        if not db_limit:
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

    # Insights Limits
    insights_limits = {}
    for plan in plans:
        result_limit = await db.execute(select(InsightsLimits).where(InsightsLimits.plan == plan))
        db_limit = result_limit.scalar_one_or_none()
        if not db_limit:
            default_limit = 3 if plan == 'starter' else 999
            db_limit = InsightsLimits(plan=plan, monthly_limit=default_limit)
            db.add(db_limit)
            await db.commit()
        insights_limits[plan] = db_limit.monthly_limit

    # AI Chat Limits
    ai_chat_limits = {}
    for plan in plans:
        result_limit = await db.execute(select(AiChatLimits).where(AiChatLimits.plan == plan))
        db_limit = result_limit.scalar_one_or_none()
        if not db_limit:
            if plan == 'starter':
                monthly_default = 5
                tp_default = 1
            elif plan == 'pro':
                monthly_default = 25
                tp_default = 0
            else:  # elite
                monthly_default = 50
                tp_default = 0
            db_limit = AiChatLimits(
                plan=plan,
                monthly_limit=monthly_default,
                tp_cost=tp_default
            )
            db.add(db_limit)
        ai_chat_limits[plan] = {
            'monthly_limit': db_limit.monthly_limit,
            'tp_cost': db_limit.tp_cost
        }

    # Beta Referral TP Config
    result_beta_referral_tp = await db.execute(select(BetaReferralTpConfig).where(BetaReferralTpConfig.id == 1))
    db_beta_referral_tp = result_beta_referral_tp.scalar_one_or_none()
    if not db_beta_referral_tp:
        db_beta_referral_tp = BetaReferralTpConfig(id=1, starter_tp=5, pro_tp=20, elite_tp=45)
        db.add(db_beta_referral_tp)
        await db.commit()
    beta_referral_tp = {
        'starter_tp': db_beta_referral_tp.starter_tp,
        'pro_tp': db_beta_referral_tp.pro_tp,
        'elite_tp': db_beta_referral_tp.elite_tp
    }

    await db.commit()  # Commit any pending initializations

    return {
        "beta_settings": beta_settings,
        "plan_counts": plan_counts,
        "initial_tp": initial_tp,
        "upgrade_tp": upgrade_tp,
        "pricing": pricing,
        "discount": discount,
        "marketplace_discount": marketplace_discount,
        "eligibility": eligibility,
        "upload_limits": upload_limits,
        "insights_limits": insights_limits,
        "ai_chat_limits": ai_chat_limits,
        "beta_referral_tp": beta_referral_tp,
    }


async def get_revenue_metrics(db: AsyncSession) -> Dict[str, Any]:
    """Compute revenue and churn metrics."""
    now = datetime.utcnow()
    one_month_ago = now - timedelta(days=30)

    active_subscribers = (await db.execute(select(func.count()).select_from(Subscription).where(Subscription.status == 'active'))).scalar() or 0
    active_subs = (await db.execute(select(Subscription).where(Subscription.status == 'active'))).scalars().all()
    mrr = sum(sub.amount_usd / (12 if sub.interval_days > 30 else 1) for sub in active_subs)

    monthly_revenue = (await db.execute(select(func.sum(Payment.amount_usd)).select_from(Payment).where(
        func.lower(Payment.status) == 'finished',
        Payment.paid_at >= one_month_ago
    ))).scalar() or 0.0
    total_users = (await db.execute(select(func.count()).select_from(User))).scalar() or 0
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

    return {
        "mrr": round(mrr, 2),
        "monthly_revenue": round(monthly_revenue, 2),
        "arpu": round(arpu, 2),
        "active_subscribers": active_subscribers,
        "user_churn_rate": round(user_churn_rate, 1),
        "revenue_churn_rate": round(revenue_churn_rate, 1),
    }


async def get_recent_users_with_stats(db: AsyncSession, search: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch recent users with detailed stats."""
    user_query = select(User).order_by(desc(User.created_at))
    if search:
        search_term = f"%{search}%"
        user_query = user_query.where(or_(User.email.like(search_term), User.full_name.like(search_term)))
    user_query = user_query.limit(limit)
    recent_users = (await db.execute(user_query)).scalars().all()
    recent_users_with_stats = []
    for user in recent_users:
        trade_query = select(Trade).where(Trade.owner_id == user.id)
        total_res = await db.execute(select(func.count()).select_from(trade_query.subquery()))
        trade_count = total_res.scalar() or 0
        wins_res = await db.execute(select(func.count()).select_from(trade_query.where(Trade.pnl > 0).subquery()))
        wins_count = wins_res.scalar() or 0
        win_rate = round((wins_count / trade_count * 100), 1) if trade_count > 1 else 0.0
        joined_formatted = user.created_at.strftime('%b %d, %Y') if user.created_at else 'N/A'
        raw_plan = getattr(user, 'plan', 'starter')
        if '_' in raw_plan and ('marketplace' in raw_plan.lower() or 'test' in raw_plan.lower()):
            plan_display = 'Starter (Marketplace Add-on)'
        else:
            plan_display = raw_plan.replace('_', ' ').title() if '_' in raw_plan else raw_plan.title()

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

        referrals_count = (await db.execute(
            select(func.count()).select_from(Referral).where(Referral.referrer_id == user.id)
        )).scalar() or 0
        total_earnings = (await db.execute(
            select(func.sum(Referral.commission_earned)).where(Referral.referrer_id == user.id)
        )).scalar() or 0.0

        points_balance = getattr(user, 'trade_points', 0)
        total_points_issued = (await db.execute(
            select(func.sum(PointTransaction.amount)).where(
                PointTransaction.user_id == user.id,
                PointTransaction.amount > 0
            )
        )).scalar() or 0

        total_invites = (await db.execute(
            select(func.count()).select_from(BetaInvite).where(BetaInvite.owner_id == user.id)
        )).scalar() or 0
        available_invites = (await db.execute(
            select(func.count()).select_from(BetaInvite).where(
                BetaInvite.owner_id == user.id, BetaInvite.used_by_id.is_(None)
            )
        )).scalar() or 0

        recent_users_with_stats.append({
            'id': user.id,
            'email': user.email,
            'full_name': user.full_name or 'N/A',
            'joined': joined_formatted,
            'trades': trade_count,
            'plan': plan_display,
            'wallet_address': user.wallet_address,
            'monthly_earnings': round(user.monthly_earnings or 0.0, 2),
            'is_trader': getattr(user, 'is_trader', False),
            'is_trader_pending': getattr(user, 'is_trader_pending', False),
            'win_rate': win_rate,
            'sub_count': sub_count,
            'sub_list': sub_list,
            'referrals_count': referrals_count,
            'total_earnings': round(total_earnings, 2),
            'points_balance': points_balance,
            'total_points_issued': total_points_issued,
            'total_invites': total_invites,
            'available_invites': available_invites,
        })
    return recent_users_with_stats


async def get_marketplace_traders_with_stats(db: AsyncSession, limit: int = 20) -> List[Dict[str, Any]]:
    """Fetch marketplace traders with stats."""
    marketplace_traders = (await db.execute(
        select(User).where(User.is_trader == True).order_by(desc(User.created_at)).limit(limit)
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
            'price': trader.marketplace_price or 19.99,
            'wallet_address': trader.wallet_address,
            'monthly_earnings': round(trader.monthly_earnings or 0.0, 2),
            'referrals_count': (await db.execute(
                select(func.count()).select_from(Referral).where(Referral.referrer_id == trader.id)
            )).scalar() or 0,
            'total_earnings_from_refs': (await db.execute(
                select(func.sum(Referral.commission_earned)).where(Referral.referrer_id == trader.id)
            )).scalar() or 0.0,
        })
    return marketplace_traders_with_stats


async def get_pending_traders_with_stats(db: AsyncSession, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch pending trader applications with stats."""
    pending_traders = (await db.execute(
        select(User).where(User.is_trader_pending == True).order_by(desc(User.created_at)).limit(limit)
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
            'wallet_address': trader.wallet_address,
            'monthly_earnings': round(trader.monthly_earnings or 0.0, 2),
            'applied_at': trader.updated_at.strftime('%b %d, %Y %H:%M') if trader.updated_at else 'N/A',
            'referrals_count': (await db.execute(
                select(func.count()).select_from(Referral).where(Referral.referrer_id == trader.id)
            )).scalar() or 0,
            'total_earnings_from_refs': (await db.execute(
                select(func.sum(Referral.commission_earned)).where(Referral.referrer_id == trader.id)
            )).scalar() or 0.0,
        })
    return pending_traders_with_stats


async def get_recent_trades_list(db: AsyncSession, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch recent trades."""
    recent_trades = (await db.execute(select(Trade).order_by(desc(Trade.created_at)).limit(limit))).scalars().all()
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
    return recent_trades_list


async def get_recent_initiated_platform_list(db: AsyncSession, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch recent initiated platform subscriptions."""
    recent_initiated_platform = (await db.execute(
        select(Subscription).where(
            Subscription.status.in_(['pending', 'pending_renewal']),
            Subscription.trader_id.is_(None)
        ).order_by(desc(Subscription.start_date)).limit(limit)
    )).scalars().all()
    recent_initiated_platform_list = []
    for sub in recent_initiated_platform:
        user = await db.get(User, sub.user_id)
        trader_name = sub.plan_type.replace('_', ' ').title()
        started_formatted = sub.start_date.strftime('%b %d, %Y %H:%M') if sub.start_date else 'N/A'
        recent_initiated_platform_list.append({
            'id': sub.id,
            'user_email': user.email if user else 'Unknown',
            'user_id': sub.user_id,
            'trader_name': trader_name,
            'is_marketplace': False,
            'plan_type': sub.plan_type,
            'amount_usd': sub.amount_usd,
            'status': sub.status,
            'started': started_formatted,
        })
    return recent_initiated_platform_list


async def get_recent_initiated_marketplace_list(db: AsyncSession, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch recent initiated marketplace subscriptions."""
    recent_initiated_marketplace = (await db.execute(
        select(Subscription).where(
            Subscription.status.in_(['pending', 'pending_renewal']),
            Subscription.trader_id.is_not(None)
        ).order_by(desc(Subscription.start_date)).limit(limit)
    )).scalars().all()
    recent_initiated_marketplace_list = []
    for sub in recent_initiated_marketplace:
        user = await db.get(User, sub.user_id)
        trader = await db.get(User, sub.trader_id)
        trader_name = trader.full_name or f'Trader {trader.id}' if trader else 'Unknown'
        started_formatted = sub.start_date.strftime('%b %d, %Y %H:%M') if sub.start_date else 'N/A'
        recent_initiated_marketplace_list.append({
            'id': sub.id,
            'user_email': user.email if user else 'Unknown',
            'user_id': sub.user_id,
            'trader_name': trader_name,
            'is_marketplace': True,
            'plan_type': sub.plan_type,
            'amount_usd': sub.amount_usd,
            'status': sub.status,
            'started': started_formatted,
        })
    return recent_initiated_marketplace_list


async def get_recent_partial_platform_list(db: AsyncSession, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch recent partial platform payments."""
    partial_payments_platform = (await db.execute(
        select(Payment).join(Subscription, Payment.subscription_id == Subscription.id).where(
            func.lower(Payment.status) == 'partially_paid',
            Subscription.trader_id.is_(None)
        ).order_by(desc(Payment.created_at)).limit(limit)
    )).scalars().all()
    recent_partial_platform = []
    for p in partial_payments_platform:
        sub = await db.get(Subscription, p.subscription_id)
        if not sub:
            continue
        user = await db.get(User, sub.user_id)
        trader_name = None
        created_formatted = p.created_at.strftime('%b %d, %Y %H:%M') if p.created_at else 'N/A'
        recent_partial_platform.append({
            'id': p.id,
            'nowpayments_payment_id': p.nowpayments_payment_id,
            'user_email': user.email if user else 'Unknown',
            'user_id': sub.user_id,
            'trader_name': trader_name,
            'is_marketplace': False,
            'amount_usd': p.amount_usd,
            'amount_paid_crypto': p.amount_paid_crypto,
            'crypto_currency': p.crypto_currency,
            'created': created_formatted,
        })
    return recent_partial_platform


async def get_recent_partial_marketplace_list(db: AsyncSession, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch recent partial marketplace payments."""
    partial_payments_marketplace = (await db.execute(
        select(Payment).join(Subscription, Payment.subscription_id == Subscription.id).where(
            func.lower(Payment.status) == 'partially_paid',
            Subscription.trader_id.is_not(None)
        ).order_by(desc(Payment.created_at)).limit(limit)
    )).scalars().all()
    recent_partial_marketplace = []
    for p in partial_payments_marketplace:
        sub = await db.get(Subscription, p.subscription_id)
        if not sub:
            continue
        user = await db.get(User, sub.user_id)
        trader = await db.get(User, sub.trader_id)
        trader_name = trader.full_name or trader.username if trader else None
        created_formatted = p.created_at.strftime('%b %d, %Y %H:%M') if p.created_at else 'N/A'
        recent_partial_marketplace.append({
            'id': p.id,
            'nowpayments_payment_id': p.nowpayments_payment_id,
            'user_email': user.email if user else 'Unknown',
            'user_id': sub.user_id,
            'trader_name': trader_name,
            'is_marketplace': True,
            'amount_usd': p.amount_usd,
            'amount_paid_crypto': p.amount_paid_crypto,
            'crypto_currency': p.crypto_currency,
            'created': created_formatted,
        })
    return recent_partial_marketplace


async def get_recent_referrals_list(db: AsyncSession, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch recent referrals."""
    recent_referrals = (await db.execute(
        select(Referral).order_by(desc(Referral.created_at)).limit(limit)
    )).scalars().all()
    recent_referrals_list = []
    for ref in recent_referrals:
        referrer = await db.get(User, ref.referrer_id)
        referee = await db.get(User, ref.referee_id)
        created_formatted = ref.created_at.strftime('%b %d, %Y %H:%M') if ref.created_at else 'N/A'
        recent_referrals_list.append({
            'id': ref.id,
            'referrer_email': referrer.email if referrer else 'Unknown',
            'referrer_id': ref.referrer_id,
            'referee_email': referee.email if referee else 'Unknown',
            'referee_id': ref.referee_id,
            'status': ref.status,
            'commission_earnings': round(ref.commission_earned, 2),
            'points_earned': ref.points_earned,
            'created': created_formatted,
        })
    return recent_referrals_list


async def get_recent_points_list(db: AsyncSession, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch recent point transactions."""
    recent_points = (await db.execute(
        select(PointTransaction).order_by(desc(PointTransaction.created_at)).limit(limit)
    )).scalars().all()
    recent_points_list = []
    for pt in recent_points:
        user = await db.get(User, pt.user_id)
        created_formatted = pt.created_at.strftime('%b %d, %Y %H:%M') if pt.created_at else 'N/A'
        recent_points_list.append({
            'id': pt.id,
            'user_email': user.email if user else 'Unknown',
            'user_id': pt.user_id,
            'type': pt.type,
            'amount': pt.amount,
            'description': pt.description,
            'created': created_formatted,
        })
    return recent_points_list


async def get_user_details_data(db: AsyncSession, user_id: int) -> Dict[str, Any]:
    """Fetch detailed user data for admin."""
    user = await db.get(User, user_id)
    if not user:
        raise ValueError("User not found")

    referrals_query = select(Referral).where(Referral.referrer_id == user.id).order_by(desc(Referral.created_at))
    referrals_result = await db.execute(referrals_query)
    referrals = referrals_result.scalars().all()

    points_query = select(PointTransaction).where(PointTransaction.user_id == user.id).order_by(desc(PointTransaction.created_at)).limit(50)
    points_result = await db.execute(points_query)
    point_transactions = points_result.scalars().all()

    invites_query = select(BetaInvite).where(BetaInvite.owner_id == user.id).order_by(desc(BetaInvite.created_at))
    invites_result = await db.execute(invites_query)
    invites = invites_result.scalars().all()

    referrals_with_stats = []
    for ref in referrals:
        referrer = await db.get(User, ref.referrer_id)
        referee = await db.get(User, ref.referee_id)
        created_formatted = ref.created_at.strftime('%b %d, %Y %H:%M') if ref.created_at else 'N/A'
        referrals_with_stats.append({
            'id': ref.id,
            'referee_email': referee.email if referee else 'Unknown',
            'referee_id': ref.referee_id,
            'status': ref.status,
            'commission_earned': round(ref.commission_earned, 2),
            'points_earned': ref.points_earned,
            'created': created_formatted,
        })

    point_transactions_list = []
    for pt in point_transactions:
        created_formatted = pt.created_at.strftime('%b %d, %Y %H:%M') if pt.created_at else 'N/A'
        point_transactions_list.append({
            'id': pt.id,
            'type': pt.type,
            'amount': pt.amount,
            'description': pt.description,
            'created': created_formatted,
        })

    beta_invites_list = []
    for invite in invites:
        used_by_user = await db.get(User, invite.used_by_id) if invite.used_by_id else None
        beta_invites_list.append({
            'id': invite.id,
            'code': invite.code,
            'used_by_email': used_by_user.email if used_by_user else None,
            'used_by_id': invite.used_by_id,
            'created_at': invite.created_at.strftime('%Y-%m-%d %H:%M') if invite.created_at else None,
            'used_at': invite.used_at.strftime('%Y-%m-%d %H:%M') if invite.used_at else None,
        })

    total_earnings = (await db.execute(
        select(func.sum(Referral.commission_earned)).where(Referral.referrer_id == user.id)
    )).scalar() or 0.0
    total_points_earned = sum(r.points_earned for r in referrals)

    return {
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name or 'N/A',
            "username": user.username,
            "referral_code": user.referral_code,
            "plan": getattr(user, 'plan', 'starter'),
            "wallet_address": user.wallet_address,
            "monthly_earnings": round(user.monthly_earnings or 0.0, 2),
            "is_trader": getattr(user, 'is_trader', False),
            "is_trader_pending": getattr(user, 'is_trader_pending', False),
            "trade_points": getattr(user, 'trade_points', 0),
            "referral_tier": getattr(user, 'referral_tier', 'rookie'),
            "created_at": user.created_at.strftime('%Y-%m-%d %H:%M') if user.created_at else None,
            "referral_count": len(referrals),
            "total_earnings": round(total_earnings, 2),
            "total_points_earned": total_points_earned,
        },
        "referrals": referrals_with_stats,
        "point_transactions": point_transactions_list,
        "beta_invites": beta_invites_list,
        "subscriptions": (await db.execute(
            select(Subscription).where(Subscription.user_id == user.id)
        )).scalars().all(),
        "payments": (await db.execute(
            select(Payment).where(Payment.user_id == user.id)
        )).scalars().all(),
    }


async def get_users_list(db: AsyncSession, search: Optional[str] = None, plan: Optional[str] = None, is_trader: Optional[bool] = None, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    """Fetch paginated list of users with filters and stats."""
    query = select(User).order_by(desc(User.created_at))
    
    if search:
        search_term = f"%{search}%"
        query = query.where(or_(User.email.like(search_term), User.full_name.like(search_term)))
    if plan:
        query = query.where(User.plan == plan)
    if is_trader is not None:
        query = query.where(User.is_trader == is_trader)
    
    total = (await db.execute(select(func.count()).select_from(query.subquery()))).scalar()
    
    query = query.offset(offset).limit(limit)
    users_result = await db.execute(query)
    users = users_result.scalars().all()
    
    users_with_stats = []
    for u in users:
        trade_count = (await db.execute(
            select(func.count(Trade.id)).where(Trade.owner_id == u.id)
        )).scalar() or 0
        
        referrals_count = (await db.execute(
            select(func.count(Referral.id)).where(Referral.referrer_id == u.id)
        )).scalar() or 0
        
        points_balance = getattr(u, 'trade_points', 0)
        
        total_invites = (await db.execute(
            select(func.count()).select_from(BetaInvite).where(BetaInvite.owner_id == u.id)
        )).scalar() or 0
        available_invites = (await db.execute(
            select(func.count()).select_from(BetaInvite).where(
                BetaInvite.owner_id == u.id, BetaInvite.used_by_id.is_(None)
            )
        )).scalar() or 0
        
        users_with_stats.append({
            'id': u.id,
            'email': u.email,
            'full_name': u.full_name or 'N/A',
            'plan': u.plan,
            'trade_count': trade_count,
            'referrals_count': referrals_count,
            'points_balance': points_balance,
            'is_trader': u.is_trader,
            'total_invites': total_invites,
            'available_invites': available_invites,
            'created_at': u.created_at.strftime('%Y-%m-%d') if u.created_at else None,
        })
    
    return {
        "users": users_with_stats,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


async def get_referrals_list(db: AsyncSession, search: Optional[str] = None, status: Optional[str] = None, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    """Fetch paginated list of referrals with filters."""
    query = select(Referral).order_by(desc(Referral.created_at))
    
    if search:
        search_term = f"%{search}%"
        query = query.join(User, Referral.referee_id == User.id).where(
            or_(User.email.like(search_term), User.full_name.like(search_term))
        )
    if status:
        query = query.where(Referral.status == status)
    
    total = (await db.execute(select(func.count()).select_from(query.subquery()))).scalar()
    
    query = query.offset(offset).limit(limit)
    referrals_result = await db.execute(query)
    referrals = referrals_result.scalars().all()
    
    referrals_with_stats = []
    for r in referrals:
        referrer = await db.get(User, r.referrer_id)
        referee = await db.get(User, r.referee_id)
        referrals_with_stats.append({
            'id': r.id,
            'referrer_email': referrer.email if referrer else 'Unknown',
            'referrer_id': r.referrer_id,
            'referee_email': referee.email if referee else 'Unknown',
            'referee_id': r.referee_id,
            'status': r.status,
            'commission_earned': round(r.commission_earned, 2),
            'points_earned': r.points_earned,
            'tier_bonus': r.tier_bonus,
            'created_at': r.created_at.strftime('%Y-%m-%d %H:%M') if r.created_at else None,
        })
    
    return {
        "referrals": referrals_with_stats,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


async def get_beta_invites_list(db: AsyncSession, search: Optional[str] = None, used: Optional[bool] = None, owner_id: Optional[int] = None, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    """Fetch paginated list of beta invites with filters."""
    query = select(BetaInvite).order_by(desc(BetaInvite.created_at))
    
    if search:
        search_term = f"%{search}%"
        query = query.join(User, BetaInvite.used_by_id == User.id).outerjoin(User, BetaInvite.owner_id == User.id).where(
            or_(
                User.email.like(search_term),
                BetaInvite.code.like(search_term),
                User.full_name.like(search_term)
            )
        )
    if used is not None:
        if used:
            query = query.where(BetaInvite.used_by_id.is_not(None))
        else:
            query = query.where(BetaInvite.used_by_id.is_(None))
    if owner_id:
        query = query.where(BetaInvite.owner_id == owner_id)
    
    total = (await db.execute(select(func.count()).select_from(query.subquery()))).scalar()
    
    query = query.offset(offset).limit(limit)
    invites_result = await db.execute(query)
    invites = invites_result.scalars().all()
    
    invites_with_stats = []
    for i in invites:
        owner = await db.get(User, i.owner_id)
        used_by = await db.get(User, i.used_by_id) if i.used_by_id else None
        invites_with_stats.append({
            'id': i.id,
            'code': i.code,
            'owner_email': owner.email if owner else 'Unknown',
            'owner_id': i.owner_id,
            'used_by_email': used_by.email if used_by else None,
            'used_by_id': i.used_by_id,
            'created_at': i.created_at.strftime('%Y-%m-%d %H:%M') if i.created_at else None,
            'used_at': i.used_at.strftime('%Y-%m-%d %H:%M') if i.used_at else None,
        })
    
    return {
        "beta_invites": invites_with_stats,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


async def get_points_list(db: AsyncSession, search: Optional[str] = None, type_: Optional[str] = None, user_id: Optional[int] = None, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    """Fetch paginated list of point transactions with filters."""
    query = select(PointTransaction).order_by(desc(PointTransaction.created_at))
    
    if search:
        search_term = f"%{search}%"
        query = query.join(User, PointTransaction.user_id == User.id).where(
            or_(User.email.like(search_term), User.full_name.like(search_term))
        )
    if type_:
        query = query.where(PointTransaction.type == type_)
    if user_id:
        query = query.where(PointTransaction.user_id == user_id)
    
    total = (await db.execute(select(func.count()).select_from(query.subquery()))).scalar()
    
    query = query.offset(offset).limit(limit)
    points_result = await db.execute(query)
    points = points_result.scalars().all()
    
    points_with_stats = []
    for pt in points:
        user = await db.get(User, pt.user_id)
        points_with_stats.append({
            'id': pt.id,
            'user_email': user.email if user else 'Unknown',
            'user_id': pt.user_id,
            'type': pt.type,
            'amount': pt.amount,
            'description': pt.description,
            'created_at': pt.created_at.strftime('%Y-%m-%d %H:%M') if pt.created_at else None,
        })
    
    return {
        "point_transactions": points_with_stats,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


async def get_payments_list(db: AsyncSession, status: Optional[str] = None, user_id: Optional[int] = None, crypto_currency: Optional[str] = None, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    """Fetch paginated list of payments with filters."""
    query = select(Payment).order_by(desc(Payment.created_at))
    
    if status:
        query = query.where(func.lower(Payment.status) == status.lower())
    if user_id:
        query = query.where(Payment.user_id == user_id)
    if crypto_currency:
        query = query.where(Payment.crypto_currency == crypto_currency)
    
    total = (await db.execute(select(func.count()).select_from(query.subquery()))).scalar()
    
    query = query.offset(offset).limit(limit)
    payments_result = await db.execute(query)
    payments = payments_result.scalars().all()
    
    payments_with_stats = []
    for p in payments:
        user = await db.get(User, p.user_id)
        sub = await db.get(Subscription, p.subscription_id) if p.subscription_id else None
        sub_type = "Platform" if sub and not sub.trader_id else "Marketplace" if sub else "Standalone"
        payments_with_stats.append({
            'id': p.id,
            'nowpayments_payment_id': p.nowpayments_payment_id,
            'user_email': user.email if user else 'Unknown',
            'user_id': p.user_id,
            'sub_type': sub_type,
            'amount_usd': p.amount_usd,
            'amount_paid_usd': p.amount_paid_usd,
            'crypto_currency': p.crypto_currency,
            'status': p.status,
            'created_at': p.created_at.strftime('%Y-%m-%d %H:%M') if p.created_at else None,
        })
    
    return {
        "payments": payments_with_stats,
        "total": total,
        "limit": limit,
        "offset": offset,
    }
# server/router/dashboard.py
import logging
import json
from fastapi import APIRouter, Request, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from sqlalchemy.orm import aliased
from typing import Optional
from collections import defaultdict
from datetime import datetime, timedelta, date

from auth import get_current_user_optional
from database import get_session
from models.models import (
    User, Trade, Subscription, EligibilityConfig, Referral, BetaInvite, BetaConfig
)
from templates_config import templates

# Redis
from redis.asyncio import Redis
from redis_client import redis_dependency, get_cache, set_cache

# Discount helper
from app_utils.discount import get_discount

logger = logging.getLogger("iTrade")
router = APIRouter()


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_session),
    redis: Redis = Depends(redis_dependency)
):
    if not current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    # Ensure fresh user data (e.g., after TP balance updates)
    await db.refresh(current_user)

    needs_onboarding = not current_user.trading_style
    if needs_onboarding:
        return RedirectResponse(url="/onboard", status_code=status.HTTP_303_SEE_OTHER)

    # === Per-user cache (5 min TTL) ===
    user_cache_key = f"dashboard:user:{current_user.id}"
    cached_data = await get_cache(redis, user_cache_key)

    if cached_data:
        context = json.loads(cached_data)
        logger.info(f"Dashboard cache HIT for user {current_user.id}")
    else:
        # === Fresh computation ===
        logger.info(f"Dashboard cache MISS for user {current_user.id} – computing fresh")

        # --- User initials & greeting ---
        initials = "U"
        first_name = "Trader"
        if current_user.full_name:
            names = current_user.full_name.split()
            first_name = names[0]
            if len(names) >= 2:
                initials = names[0][0].upper() + names[-1][0].upper()
            else:
                initials = names[0][0].upper() * 2
        greeting = f"Hey {first_name},"

        now = datetime.utcnow()
        now_date = now.date()

        # Fixed capital for % calculations
        fixed_capital = current_user.initial_deposit or 10000.0

        # --- Fetch trades once ---
        result = await db.execute(
            select(Trade)
            .where(Trade.owner_id == current_user.id)
            .order_by(Trade.created_at.desc())
        )
        trades = result.scalars().all()

        # --- PNL Summary ---
        def pnl_for_period(start_dt):
            period_trades = [t for t in trades if t.created_at >= start_dt]
            return sum(t.pnl or 0 for t in period_trades)

        daily_pnl = pnl_for_period(now - timedelta(days=1))
        weekly_pnl = pnl_for_period(now - timedelta(weeks=1))
        monthly_pnl = pnl_for_period(now - timedelta(days=30))

        daily_percent = round((daily_pnl / fixed_capital * 100), 1) if fixed_capital else 0
        weekly_percent = round((weekly_pnl / fixed_capital * 100), 1) if fixed_capital else 0
        monthly_percent = round((monthly_pnl / fixed_capital * 100), 1) if fixed_capital else 0

        pnl_summary = {
            'daily': {'value': round(daily_pnl, 0), 'percent': daily_percent},
            'weekly': {'value': round(weekly_pnl, 0), 'percent': weekly_percent},
            'monthly': {'value': round(monthly_pnl, 0), 'percent': monthly_percent},
        }

        # --- PNL History (last 5 periods) ---
        def get_history(period: str):
            if period == 'daily':
                deltas = [timedelta(days=i) for i in range(5)]
                delta_step = timedelta(days=1)
            elif period == 'weekly':
                deltas = [timedelta(weeks=i) for i in range(5)]
                delta_step = timedelta(weeks=1)
            else:
                deltas = [timedelta(days=30 * i) for i in range(5)]
                delta_step = timedelta(days=30)

            history = []
            for delta in reversed(deltas):
                start = now - delta - delta_step
                end = now - delta
                period_pnl = sum(t.pnl or 0 for t in trades if start <= t.created_at < end)
                history.append(round(period_pnl))
            return history

        pnl_history = {
            'daily': get_history('daily'),
            'weekly': get_history('weekly'),
            'monthly': get_history('monthly')
        }

        # --- Session Analysis ---
        session_config = {
            'sydney': {'display': 'Sydney', 'icon': 'bi-sun'},
            'tokyo': {'display': 'Tokyo', 'icon': 'bi-moon'},
            'london': {'display': 'London', 'icon': 'bi-cloud'},
            'ny': {'display': 'New York', 'icon': 'bi-building'},
        }
        sessions_dict = defaultdict(lambda: {'trades': [], 'pnl': 0.0})
        for t in trades:
            if not t.session:
                continue
            norm = t.session.lower()
            if 'new york' in norm or 'ny' in norm:
                norm = 'ny'
            key = norm if norm in session_config else 'other'
            if key == 'other' and 'other' not in session_config:
                session_config['other'] = {'display': 'Other', 'icon': 'bi-globe2'}
            sessions_dict[key]['trades'].append(t)
            sessions_dict[key]['pnl'] += t.pnl or 0

        sessions = []
        tips = {
            'sydney': 'Early bird gets the worm—focus on AUD pairs.',
            'tokyo': 'Zen mode: patience pays in low vol.',
            'london': 'High energy—watch for breakouts.',
            'ny': 'Power hour: momentum rules.',
            'other': 'Keep grinding!'
        }
        for key, data in sessions_dict.items():
            trades_list = data['trades']
            if not trades_list:
                continue
            total_pnl = data['pnl']
            wins = sum(1 for t in trades_list if t.pnl and t.pnl > 0)
            win_rate = round((wins / len(trades_list) * 100))
            avg_pnl = total_pnl / len(trades_list)
            avg_pnl_str = f"${avg_pnl:+.2f}"

            # Cumulative trend (last 5 trades)
            recent = sorted(trades_list, key=lambda t: t.created_at, reverse=True)[:5]
            trend = [0] * (5 - len(recent))
            cum = 0
            for t in reversed(recent):
                cum += t.pnl or 0
                trend.append(round(cum))

            last_trade = max(trades_list, key=lambda t: t.created_at)
            last_trade_str = f"{last_trade.symbol or 'N/A'} {last_trade.pnl:+.2f}" if last_trade.pnl is not None else "N/A"

            sessions.append({
                'name': session_config[key]['display'],
                'icon': session_config[key]['icon'],
                'trades': len(trades_list),
                'winRate': win_rate,
                'avgPnL': avg_pnl_str,
                'trend': trend,
                'lastTrade': last_trade_str,
                'tip': tips.get(key, 'Keep it up!')
            })

        sample_session = sessions[0] if sessions else {'name': 'your sessions', 'winRate': 0}

        # --- Referrals (TP only) ---
        referrals_count = (await db.execute(
            select(func.count()).select_from(User).where(User.referred_by == current_user.id)
        )).scalar() or 0

        refs_result = await db.execute(select(Referral).where(Referral.referrer_id == current_user.id))
        referrals = refs_result.scalars().all()
        total_tp_earned = sum(r.points_earned for r in referrals)
        referral_tier = current_user.referral_tier or "Bronze"

        # --- Beta Invites ---
        invites_result = await db.execute(
            select(BetaInvite)
            .where(BetaInvite.owner_id == current_user.id)
            .order_by(BetaInvite.created_at.desc())
        )
        beta_invites = invites_result.scalars().all()
        available_beta_codes = [i.code for i in beta_invites if i.used_by_id is None]
        used_beta_count = len(beta_invites) - len(available_beta_codes)

        beta_config = (await db.execute(select(BetaConfig).where(BetaConfig.id == 1))).scalar_one_or_none()
        beta_active = beta_config.is_active if beta_config else False

        points_balance = current_user.trade_points if current_user.trade_points is not None else 0

        # --- Base context (cacheable) ---
        context = {
            "initials": initials,
            "greeting": greeting,
            "todays_edge": daily_percent,
            "pnl_summary": pnl_summary,
            "pnl_history": pnl_history,
            "sessions": sessions,
            "session": sample_session,
            "referral_code": current_user.referral_code,
            "referrals_count": referrals_count,
            "earned_analyses": referrals_count,
            "points_balance": points_balance,
            "referral_tier": referral_tier,
            "total_tp_earned": total_tp_earned,
            "available_beta_codes": available_beta_codes,
            "used_beta_count": used_beta_count,
            "total_beta_invites": len(beta_invites),
            "beta_active": beta_active,
            # Will be overridden later
            "active_subs": [],
            "pending_subs": [],
            "traders": [],
            "recommendations": [],
            "platform_discount": 0,
            "marketplace_discount": 0,
        }

        # Cache for 5 minutes
        await set_cache(redis, user_cache_key, json.dumps(context), ttl=300)
        logger.info(f"Dashboard data cached for user {current_user.id}")

    # ============================================================
    # ALWAYS compute fresh dynamic parts (subscriptions, discounts, traders)
    # ============================================================

    now_date = date.today()

    # --- Discounts ---
    platform_config = await get_discount(db, 1)
    marketplace_config = await get_discount(db, 2)

    def is_active(cfg):
        if not cfg or not cfg.get('enabled', False):
            return False
        exp = cfg.get('expiry')
        return exp is None or exp > now_date

    context["platform_discount"] = platform_config['percentage'] if is_active(platform_config) else 0
    context["marketplace_discount"] = marketplace_config['percentage'] if is_active(marketplace_config) else 0

    # --- Active Subscriptions ---
    active_result = await db.execute(
        select(Subscription)
        .where(Subscription.user_id == current_user.id, Subscription.status == 'active')
        .order_by(Subscription.start_date.desc())
    )
    active_subs = []
    for sub in active_result.scalars().all():
        next_bill = sub.next_billing_date.strftime('%b %d') if sub.next_billing_date else 'N/A'
        if sub.trader_id:
            trader = await db.get(User, sub.trader_id)
            name = (trader.full_name or trader.username) if trader else "Unknown"
            win_rate = round(getattr(trader, 'win_rate', 0), 1) if trader else 0
            active_subs.append({
                'trader_id': sub.trader_id,
                'trader_name': name,
                'win_rate': win_rate,
                'next_bill': next_bill
            })
        else:
            active_subs.append({
                'trader_id': None,
                'plan_type': sub.plan_type,
                'next_bill': next_bill
            })

    # Fallback for legacy plan on user object
    if current_user.plan != 'starter' and not any(s['trader_id'] is None for s in active_subs):
        active_subs.append({
            'trader_id': None,
            'plan_type': current_user.plan,
            'next_bill': (datetime.utcnow() + timedelta(days=30)).strftime('%b %d')
        })
    context["active_subs"] = active_subs[:2]

    # --- Pending Marketplace Subs ---
    pending_result = await db.execute(
        select(Subscription)
        .where(
            Subscription.user_id == current_user.id,
            Subscription.status == 'pending',
            Subscription.trader_id.is_not(None)
        )
    )
    context["pending_subs"] = pending_result.scalars().all()

    # --- Eligible Traders (global cache with version to avoid writes) ---
    version = await redis.get("eligible_traders:version") or "1"
    if isinstance(version, bytes):
        version = version.decode()
    global_cache_key = f"eligible_traders:v{version}"

    traders_data = await get_cache(redis, global_cache_key)
    if traders_data is None:
        # Rebuild and cache (this is the only place we write to Redis for traders)
        # If you're on read-only Redis, this will still work as long as your Redis URL points to the primary
        traders = await build_eligible_traders(db, redis)  # Helper below
        await set_cache(redis, global_cache_key, json.dumps(traders), ttl=300)
    else:
        traders = json.loads(traders_data)

    # Attach subscription status
    sub_status = {s.trader_id: s.status for s in context["pending_subs"] + 
                  [s for s in active_result.scalars().all() if s.trader_id]}
    for t in traders:
        t['status'] = sub_status.get(t['id'])

    context["traders"] = traders
    context["recommendations"] = traders[:3]

    # Final fresh values
    await db.refresh(current_user)
    context["points_balance"] = current_user.trade_points if current_user.trade_points is not None else 0
    context["current_user"] = current_user
    context["request"] = request
    context["now"] = datetime.utcnow()

    return templates.TemplateResponse("dashboard.html", context)


# Helper: build eligible traders (called only when cache miss)
async def build_eligible_traders(db: AsyncSession, redis: Redis):
    config = (await db.execute(select(EligibilityConfig).where(EligibilityConfig.id == 1))).scalar_one_or_none()
    min_trades = config.min_trades if config else 50
    min_win_rate = config.min_win_rate if config else 80.0

    trader_users = (await db.execute(
        select(User.id, User.full_name, User.strategy, User.marketplace_price)
        .where(User.is_trader == True)
    )).all()

    eligible = []
    for trader_id, full_name, strategy, price in trader_users:
        if not full_name:
            continue

        # Stats
        trades_query = select(Trade).where(Trade.owner_id == trader_id)
        total = (await db.execute(select(func.count()).select_from(trades_query.subquery()))).scalar() or 0
        if total < min_trades:
            continue

        wins = (await db.execute(select(func.count()).select_from(trades_query.where(Trade.pnl > 0).subquery()))).scalar() or 0
        win_rate = round(wins / total * 100, 1)
        if win_rate < min_win_rate:
            continue

        total_pnl = (await db.execute(select(func.sum(Trade.pnl)).select_from(trades_query.subquery()))).scalar() or 0.0

        # Trend (last 5 cumulative)
        recent = (await db.execute(
            select(Trade.pnl)
            .where(Trade.owner_id == trader_id)
            .order_by(Trade.created_at.desc())
            .limit(5)
        )).scalars().all()
        trend = [0] * (5 - len(recent))
        cum = 0
        for p in reversed(recent or []):
            cum += p or 0
            trend.append(round(cum))

        # Journal tease
        note = (await db.execute(
            select(Trade.notes)
            .where(Trade.owner_id == trader_id)
            .order_by(Trade.created_at.desc())
            .limit(1)
        )).scalar_one_or_none()
        tease = (note[:100] + '...' if note else 'No journal yet.')

        names = full_name.split()
        initials = names[0][0].upper() + (names[-1][0].upper() if len(names) > 1 else names[0][0].upper())

        eligible.append({
            "id": trader_id,
            "name": full_name,
            "initials": initials,
            "strategy": strategy or "Proven System",
            "win_rate": win_rate,
            "trades": total,
            "pnl": round(total_pnl),
            "trend": trend,
            "journal_tease": tease,
            "monthly_price": price or 19.99,
        })

    eligible.sort(key=lambda x: (-x['win_rate'], -x['trades']))
    return eligible[:20]



import logging
from fastapi import APIRouter, Request, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case
from sqlalchemy.orm import aliased
from typing import Optional
from collections import defaultdict
from datetime import datetime, timedelta, date  # ← ADD: date for expiry comparison

from auth import get_current_user_optional
from database import get_session
from models.models import (
    User, Trade, Subscription, EligibilityConfig
)
from templates_config import templates

# --- NEW: Import discount helper ---
from app_utils.discount import get_discount  

logger = logging.getLogger("iTrade")

router = APIRouter()


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_session)
):
    if not current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    needs_onboarding = not current_user.trading_style
    if needs_onboarding:
        return RedirectResponse(url="/onboard", status_code=status.HTTP_303_SEE_OTHER)

    # --- User Initials ---
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
    now_date = now.date()  # ← ADD: For expiry date comparison

    # --- Fetch User's Trades ---
    result = await db.execute(
        select(Trade)
        .where(Trade.owner_id == current_user.id)
        .order_by(Trade.created_at.desc())
    )
    trades = result.scalars().all()

    first_name = current_user.full_name.split()[0] if current_user.full_name else 'Trader'
    greeting = f"Hey {first_name},"

    assumed_capital = 10000

    # --- PNL Calculations ---
    day_ago = now - timedelta(days=1)
    daily_trades = [t for t in trades if t.created_at >= day_ago]
    daily_pnl = sum(t.pnl or 0 for t in daily_trades)
    daily_percent = round((daily_pnl / assumed_capital * 100), 1) if assumed_capital else 0
    todays_edge = daily_percent

    week_ago = now - timedelta(weeks=1)
    weekly_trades = [t for t in trades if t.created_at >= week_ago]
    weekly_pnl = sum(t.pnl or 0 for t in weekly_trades)
    weekly_percent = round((weekly_pnl / assumed_capital * 100), 1) if assumed_capital else 0

    month_ago = now - timedelta(days=30)
    monthly_trades = [t for t in trades if t.created_at >= month_ago]
    monthly_pnl = sum(t.pnl or 0 for t in monthly_trades)
    monthly_percent = round((monthly_pnl / assumed_capital * 100), 1) if assumed_capital else 0

    # --- PNL Summary ---
    pnl_summary = {
        'daily': {'value': daily_pnl, 'percent': daily_percent},
        'weekly': {'value': weekly_pnl, 'percent': weekly_percent},
        'monthly': {'value': monthly_pnl, 'percent': monthly_percent},
    }

    # --- PNL History (Last 5 periods) ---
    def get_history(period):
        if period == 'daily':
            deltas = [timedelta(days=i) for i in range(5)]
            period_delta = timedelta(days=1)
        elif period == 'weekly':
            deltas = [timedelta(weeks=i) for i in range(5)]
            period_delta = timedelta(weeks=1)
        else:
            deltas = [timedelta(days=30 * i) for i in range(5)]
            period_delta = timedelta(days=30)
        history = []
        for delta in reversed(deltas):
            start = now - delta - period_delta
            end = now - delta
            period_trades = [t for t in trades if start <= t.created_at < end]
            history.append(sum(t.pnl or 0 for t in period_trades))
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
    sessions_dict = defaultdict(lambda: {'trades': [], 'pnl': 0})
    for t in trades:
        if not t.session:
            continue
        norm = t.session.lower()
        if 'new york' in norm:
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
    assumed_position = 1000
    for key, data in sessions_dict.items():
        trades_list = data['trades']
        if not trades_list:
            continue
        total_pnl = data['pnl']
        wins = sum(1 for t in trades_list if t.pnl and t.pnl > 0)
        win_rate = round((wins / len(trades_list) * 100))
        avg_pnl = total_pnl / len(trades_list)
        avg_pnl_str = f"${avg_pnl:+.2f}"

        recent_trades = sorted(trades_list, key=lambda t: t.created_at)[-5:]
        trend = [0] * (5 - len(recent_trades))
        current = 0
        for t in recent_trades:
            change = ((t.pnl or 0) / assumed_position * 100)
            current += change
            trend.append(round(current))

        last_trade = max(trades_list, key=lambda t: t.created_at)
        pnl_str = f"{last_trade.pnl:+.2f}" if last_trade.pnl is not None else "N/A"
        last_trade_str = f"{last_trade.symbol or 'N/A'} ${pnl_str}"
        tip = tips.get(key, 'Keep it up!')

        sessions.append({
            'name': session_config[key]['display'],
            'icon': session_config[key]['icon'],
            'trades': len(trades_list),
            'winRate': win_rate,
            'avgPnL': avg_pnl_str,
            'trend': trend,
            'lastTrade': last_trade_str,
            'tip': tip
        })

    sample_session = sessions[0] if sessions else {'name': 'your sessions', 'winRate': 0}

    # --- Referrals ---
    referrals_count = (await db.execute(
        select(func.count()).select_from(User).where(User.referred_by == current_user.id)
    )).scalar() or 0
    earned_analyses = referrals_count

    # --- Active Subscriptions (Platform + Marketplace) ---
    active_subs_result = await db.execute(
        select(Subscription)
        .where(
            Subscription.user_id == current_user.id,
            Subscription.status == 'active'
        )
        .order_by(Subscription.start_date.desc())
    )
    all_active_subs = active_subs_result.scalars().all()

    active_subs = []
    for sub in all_active_subs:
        next_bill = sub.next_billing_date.strftime('%b %d') if sub.next_billing_date else 'N/A'
        if sub.trader_id:
            # Marketplace sub
            trader = await db.get(User, sub.trader_id)
            trader_name = trader.full_name or trader.username if trader else "Unknown Trader"
            win_rate = getattr(trader, 'win_rate', 0) if trader else 0
            active_subs.append({
                'trader_id': sub.trader_id,
                'trader_name': trader_name,
                'win_rate': round(win_rate, 1),
                'next_bill': next_bill
            })
        else:
            # Platform sub
            active_subs.append({
                'trader_id': None,
                'plan_type': sub.plan_type,
                'next_bill': next_bill
            })

    # Fallback for platform plans set directly on user
    if current_user.plan != 'starter' and not any(sub.trader_id is None for sub in all_active_subs):
        next_bill = (now + timedelta(days=30)).strftime('%b %d')
        active_subs.append({
            'trader_id': None,
            'plan_type': current_user.plan,
            'next_bill': next_bill
        })

    # Limit to 2 most recent subscriptions for display
    active_subs = active_subs[:2]

    # --- NEW: Fetch Pending Marketplace Subs for Button Status ---
    pending_subs_result = await db.execute(
        select(Subscription)
        .where(
            Subscription.user_id == current_user.id,
            Subscription.status == 'pending',
            Subscription.trader_id.is_not(None)
        )
    )
    pending_marketplace_subs = pending_subs_result.scalars().all()

    # --- Trader Status Dict (active + pending) ---
    trader_status = {}
    for sub in all_active_subs + pending_marketplace_subs:
        if sub.trader_id:
            trader_status[sub.trader_id] = sub.status

    # --- Eligibility Config ---
    config_result = await db.execute(select(EligibilityConfig).where(EligibilityConfig.id == 1))
    config = config_result.scalar_one_or_none()
    min_trades = config.min_trades if config else 50
    min_win_rate = config.min_win_rate if config else 80.0
    logger.info(f"[DASHBOARD DEBUG] Using eligibility: {min_trades} trades, {min_win_rate}% win")

    # --- Computed Win Rate Expression ---
    win_rate_computed = func.round(
        (func.sum(case((Trade.pnl > 0, 1), else_=0)) / func.nullif(func.count(Trade.id), 0) * 100), 1
    ).label('computed_win_rate')

    # --- Eligible Traders Query ---
    eligible_traders_query = select(
        User.id,
        User.full_name,
        User.strategy,
        win_rate_computed,
        func.count(Trade.id).label('total_trades'),
        func.sum(Trade.pnl).label('total_pnl'),
        func.sum(case((Trade.pnl < 0, 1), else_=0)).label('losses'),
        User.marketplace_price
    ).outerjoin(
        Trade, User.id == Trade.owner_id
    ).where(
        User.is_trader == True,
    ).group_by(
        User.id, User.full_name, User.strategy, User.marketplace_price
    ).having(
        func.count(Trade.id) >= min_trades,
        win_rate_computed >= min_win_rate
    ).order_by(
        win_rate_computed.desc(),
        func.count(Trade.id).desc()
    ).limit(20)

    eligible_traders_result = await db.execute(eligible_traders_query)
    trader_rows = eligible_traders_result.all()

    # --- Build Trader Cards ---
    traders = []
    for row in trader_rows:
        trader_id, full_name, strategy, computed_win_rate, total_trades, total_pnl, losses, marketplace_price = row
        if not full_name:
            continue

        names = full_name.split()
        trader_initials = (names[0][0].upper() +
                           (names[-1][0].upper() if len(names) > 1 else names[0][0].upper()))

        # --- Trend: Last 5 PnL % ---
        trend_result = await db.execute(
            select(func.coalesce(Trade.pnl, 0))
            .where(Trade.owner_id == trader_id)
            .order_by(Trade.trade_date.desc() if hasattr(Trade, 'trade_date') else Trade.created_at.desc())
            .limit(5)
        )
        recent_pnls = trend_result.scalars().all()
        trend = [0] * (5 - len(recent_pnls))
        current = 0
        for pnl in reversed(recent_pnls):
            current += (pnl / assumed_capital * 100)
            trend.append(round(current))

        # --- Journal Tease ---
        note_result = await db.execute(
            select(Trade.notes)
            .where(Trade.owner_id == trader_id)
            .order_by(Trade.created_at.desc())
            .limit(1)
        )
        last_note = note_result.scalar_one_or_none()
        journal_tease = (last_note[:100] + '...' if last_note else 'No journal yet.')

        status = trader_status.get(trader_id)

        traders.append({
            'id': trader_id,
            'name': full_name,
            'initials': trader_initials,
            'strategy': strategy or 'Proven System',
            'win_rate': round(computed_win_rate or 0, 1),
            'trades': total_trades or 0,
            'losses': losses or 0,
            'pnl': total_pnl or 0,
            'trend': trend,
            'journal_tease': journal_tease,
            'monthly_price': marketplace_price or 19.99,
            'status': status  # NEW: 'active', 'pending', or None
        })

    recommendations = traders[:3]

    # --- FIXED: Fetch & Apply Real Discounts (No Hardcoding) ---
    platform_config = await get_discount(db, 1)  # Pro/Elite plans
    marketplace_config = await get_discount(db, 2)  # Marketplace traders

    # Apply logic: enabled + not expired → use percentage; else 0
    def is_active_discount(config, current_date):
        if not config or not config.get('enabled', False):
            return False
        expiry = config.get('expiry')
        return expiry is None or expiry > current_date

    platform_discount = platform_config['percentage'] if is_active_discount(platform_config, now_date) else 0
    marketplace_discount = marketplace_config['percentage'] if is_active_discount(marketplace_config, now_date) else 0

    logger.info(f"[DASHBOARD DEBUG] Discounts - Platform: {platform_discount}%, Marketplace: {marketplace_discount}%")  # ← ADD: For debugging

    # --- Render Template ---
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "current_user": current_user,
            "initials": initials,
            "greeting": greeting,
            "todays_edge": todays_edge,
            "pnl_summary": pnl_summary,
            "pnl_history": pnl_history,
            "sessions": sessions,
            "session": sample_session,
            "referral_code": current_user.referral_code,
            "referrals_count": referrals_count,
            "earned_analyses": earned_analyses,
            "active_subs": active_subs,
            "recommendations": recommendations,
            "traders": traders,
            "now": datetime.utcnow(),
            "platform_discount": platform_discount,
            "marketplace_discount": marketplace_discount,
        }
    )
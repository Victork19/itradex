# server/router/dashboard.py
import logging
import json  # NEW: For JSON serialization in caching
from fastapi import APIRouter, Request, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case, desc, and_  # FIXED: Add 'and_' import for HAVING clause
from sqlalchemy.orm import aliased
from typing import Optional
from collections import defaultdict
from datetime import datetime, timedelta, date  # ← ADD: date for expiry comparison

from auth import get_current_user_optional
from database import get_session
from models.models import (
    User, Trade, Subscription, EligibilityConfig, Referral, BetaInvite, BetaConfig  # ← ADD: BetaConfig
)
from templates_config import templates

# NEW: Import Redis
from redis.asyncio import Redis
from redis_client import redis_dependency, get_cache, set_cache  # FIXED: Correct import path

# --- NEW: Import discount helper ---
from app_utils.discount import get_discount  

logger = logging.getLogger("iTrade")

router = APIRouter()


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_session),
    redis: Redis = Depends(redis_dependency)  # NEW: Add Redis dependency
):
    if not current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    # FIXED: Ensure fresh user data by refreshing from DB (resolves stale TP balance after uploads)
    await db.refresh(current_user)

    needs_onboarding = not current_user.trading_style
    if needs_onboarding:
        return RedirectResponse(url="/onboard", status_code=status.HTTP_303_SEE_OTHER)

    # NEW: Cache user-specific dashboard data (TTL 5 min, invalidate on trade upload/etc.)
    cache_key = f"dashboard:{current_user.id}"
    cached_data = await get_cache(redis, cache_key)
    if cached_data:
        # Unpack cached data (assume we cache the full context minus user-specific non-cacheable parts)
        context = json.loads(cached_data)
        logger.info(f"Using cached dashboard for user {current_user.id}")
    else:
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

        # FIXED: Use FIXED capital for % calcs (avoids distortion from running balance)
        # If initial_deposit is None, fallback to 10000
        fixed_capital = current_user.initial_deposit or 10000.0
        # Keep running balance for display elsewhere if needed
        assumed_capital = current_user.account_balance or fixed_capital  # For any absolute displays

        # --- Fetch User's Trades ---
        result = await db.execute(
            select(Trade)
            .where(Trade.owner_id == current_user.id)
            .order_by(Trade.created_at.desc())
        )
        trades = result.scalars().all()

        first_name = current_user.full_name.split()[0] if current_user.full_name else 'Trader'
        greeting = f"Hey {first_name},"

        # --- PNL Calculations (FIXED: pnl is $; sum $ for value, compute % from $/capital) ---
        day_ago = now - timedelta(days=1)
        daily_trades = [t for t in trades if t.created_at >= day_ago]
        daily_pnl_dollars = sum(t.pnl or 0 for t in daily_trades)  # Sum $
        daily_percent = round((daily_pnl_dollars / fixed_capital * 100), 1) if fixed_capital else 0
        todays_edge = daily_percent

        week_ago = now - timedelta(weeks=1)
        weekly_trades = [t for t in trades if t.created_at >= week_ago]
        weekly_pnl_dollars = sum(t.pnl or 0 for t in weekly_trades)
        weekly_percent = round((weekly_pnl_dollars / fixed_capital * 100), 1) if fixed_capital else 0

        month_ago = now - timedelta(days=30)
        monthly_trades = [t for t in trades if t.created_at >= month_ago]
        monthly_pnl_dollars = sum(t.pnl or 0 for t in monthly_trades)
        monthly_percent = round((monthly_pnl_dollars / fixed_capital * 100), 1) if fixed_capital else 0

        # --- PNL Summary (value: $; percent: %) ---
        pnl_summary = {
            'daily': {'value': round(daily_pnl_dollars, 0), 'percent': daily_percent},
            'weekly': {'value': round(weekly_pnl_dollars, 0), 'percent': weekly_percent},
            'monthly': {'value': round(monthly_pnl_dollars, 0), 'percent': monthly_percent},
        }

        # --- PNL History (Last 5 periods: $ values for sparkline) ---
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
                period_pnl_dollars = sum(t.pnl or 0 for t in period_trades)  # FIXED: Sum $
                history.append(round(period_pnl_dollars, 0))
            return history

        pnl_history = {
            'daily': get_history('daily'),
            'weekly': get_history('weekly'),
            'monthly': get_history('monthly')
        }

        # --- Session Analysis (FIXED: avg_pnl as $ string, not %; trend cumulative $) ---
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
        for key, data in sessions_dict.items():
            trades_list = data['trades']
            if not trades_list:
                continue
            total_pnl_dollars = data['pnl']  # FIXED: sum $ (was misnamed 'percent')
            wins = sum(1 for t in trades_list if t.pnl and t.pnl > 0)
            win_rate = round((wins / len(trades_list) * 100))
            avg_pnl_dollars = total_pnl_dollars / len(trades_list)  # FIXED: avg $
            avg_pnl_str = f"${avg_pnl_dollars:+.2f}"  # FIXED: Display as $

            # FIXED: Trend cumulative $ (matches spark scale)
            recent_trades = sorted(trades_list, key=lambda t: t.created_at)[-5:]
            trend = [0] * (5 - len(recent_trades))
            current = 0
            for t in recent_trades:
                change = t.pnl or 0  # $
                current += change
                trend.append(round(current))

            last_trade = max(trades_list, key=lambda t: t.created_at)
            pnl_str = f"{last_trade.pnl:+.2f}" if last_trade.pnl is not None else "N/A"  # FIXED: $ not %
            last_trade_str = f"{last_trade.symbol or 'N/A'} {pnl_str}"
            tip = tips.get(key, 'Keep it up!')

            sessions.append({
                'name': session_config[key]['display'],
                'icon': session_config[key]['icon'],
                'trades': len(trades_list),
                'winRate': win_rate,
                'avgPnL': avg_pnl_str,  # Now "$+10.00"
                'trend': trend,
                'lastTrade': last_trade_str,
                'tip': tip
            })

        sample_session = sessions[0] if sessions else {'name': 'your sessions', 'winRate': 0}

        # --- Referrals - UPDATED: Focus on TP Only (No Commissions) ---
        referrals_count = (await db.execute(
            select(func.count()).select_from(User).where(User.referred_by == current_user.id)
        )).scalar() or 0

        # Query Referral table for TP earned
        referrals_result = await db.execute(
            select(Referral).where(Referral.referrer_id == current_user.id)
        )
        referrals = referrals_result.scalars().all()
        total_tp_earned = sum(r.points_earned for r in referrals)
        referral_tier = current_user.referral_tier

        earned_analyses = referrals_count  # Legacy; can repurpose for TP if needed

        # --- NEW: Beta Invites ---
        invites_query = select(BetaInvite).where(BetaInvite.owner_id == current_user.id).order_by(desc(BetaInvite.created_at))
        invites_result = await db.execute(invites_query)
        beta_invites = invites_result.scalars().all()
        available_beta_codes = [i.code for i in beta_invites if i.used_by_id is None]
        used_beta_count = len(beta_invites) - len(available_beta_codes)
        total_beta_invites = len(beta_invites)

        # --- NEW: Fetch Beta Config ---
        beta_config_result = await db.execute(select(BetaConfig).where(BetaConfig.id == 1))
        beta_config = beta_config_result.scalar_one_or_none()
        beta_active = beta_config.is_active if beta_config else False

        logger.info(f"[DASHBOARD DEBUG] Beta active: {beta_active}")

        # --- FIXED: Coerce TP Balance to 0 if None (handles legacy NULLs) ---
        points_balance = current_user.trade_points if current_user.trade_points is not None else 0

        # --- Build context dict ---
        context = {
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
            "active_subs": [],  # Placeholder - set later
            "pending_subs": [],  # Placeholder - set later
            "recommendations": [],  # Placeholder - set later
            "traders": [],  # Placeholder - set later
            "platform_discount": 0,  # Placeholder - set later
            "marketplace_discount": 0,  # Placeholder - set later
            # UPDATED: TP Data Only (No Commissions/Payouts in Referral Context)
            "points_balance": points_balance,  # FIXED: Now coerced to 0
            "referral_tier": referral_tier,
            "total_tp_earned": total_tp_earned,
            # NEW: Beta Invites Data
            "available_beta_codes": available_beta_codes,
            "used_beta_count": used_beta_count,
            "total_beta_invites": total_beta_invites,
            # UPDATED: Beta Active Flag from Config
            "beta_active": beta_active,
        }

        # Cache the serializable parts (exclude request, current_user objects)
        await set_cache(redis, cache_key, json.dumps(context), ttl=300)  # 5 min TTL
        logger.info(f"Cached dashboard data for user {current_user.id}")

    # NEW: ALWAYS COMPUTE FRESH SUBS & DISCOUNTS (quick, user-specific; overrides cache if hit)
    now_date = datetime.utcnow().date()

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

    logger.info(f"[DASHBOARD DEBUG] Discounts - Platform: {platform_discount}%, Marketplace: {marketplace_discount}%")

    context["platform_discount"] = platform_discount
    context["marketplace_discount"] = marketplace_discount

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
    if current_user.plan != 'starter' and not any(sub['trader_id'] is None for sub in active_subs):
        next_bill = (datetime.utcnow() + timedelta(days=30)).strftime('%b %d')
        active_subs.append({
            'trader_id': None,
            'plan_type': current_user.plan,
            'next_bill': next_bill
        })

    # Limit to 2 most recent subscriptions for display
    active_subs = active_subs[:2]
    context["active_subs"] = active_subs

    # --- NEW: Fetch Pending Marketplace Subs for Button Status ---
    pending_subs_result = await db.execute(
        select(Subscription)
        .where(
            Subscription.user_id == current_user.id,
            Subscription.status == 'pending',
            Subscription.trader_id.is_not(None)
        )
    )
    pending_subs = pending_subs_result.scalars().all()
    context["pending_subs"] = pending_subs

    # NEW: ALWAYS RELOAD TRADERS HERE (after cache check/miss) - ensures freshness even on per-user cache hit
    # Load from global cache (which respects invalidations from toggle_trader)
    global_cache_key = "eligible_traders"

    # TEMP HACK: Force invalidation every load for debugging
    await redis.delete(global_cache_key)
    logger.info("[DASHBOARD DEBUG] Forced invalidation of eligible_traders cache")

    traders_data = await get_cache(redis, global_cache_key)
    if traders_data is None:
        # --- Eligibility Config ---
        config_result = await db.execute(select(EligibilityConfig).where(EligibilityConfig.id == 1))
        config = config_result.scalar_one_or_none()
        min_trades = config.min_trades if config else 50
        min_win_rate = config.min_win_rate if config else 80.0
        logger.info(f"[DASHBOARD DEBUG] Using eligibility: {min_trades} trades, {min_win_rate}% win")

        # NEW: Debug log - Count users with is_trader=True (before join/filtering)
        trader_users_count_result = await db.execute(select(func.count(User.id)).where(User.is_trader == True))
        trader_users_count = trader_users_count_result.scalar() or 0
        logger.info(f"[DASHBOARD DEBUG] Users with is_trader=True: {trader_users_count}")

        # FIXED: Fetch all trader users first, then compute stats per user (mirrors journal eligibility logic for accuracy)
        trader_users_result = await db.execute(
            select(User.id, User.full_name, User.strategy, User.marketplace_price)
            .where(User.is_trader == True)
        )
        trader_users = trader_users_result.all()
        logger.info(f"[DASHBOARD DEBUG] Fetched {len(trader_users)} trader users for stat computation")

        # --- Build Eligible Traders List ---
        eligible_rows = []
        for row in trader_users:
            trader_id, full_name, strategy, marketplace_price = row
            if not full_name:
                continue

            # Compute stats separately (like journal eligibility)
            trade_query = select(Trade).where(Trade.owner_id == trader_id)
            total_trades_result = await db.execute(select(func.count()).select_from(trade_query.subquery()))
            total_trades = total_trades_result.scalar() or 0

            if total_trades < min_trades:
                logger.debug(f"[DASHBOARD DEBUG] Trader {trader_id} skipped: {total_trades} < {min_trades} trades")
                continue

            wins_result = await db.execute(select(func.count()).select_from(trade_query.where(Trade.pnl > 0).subquery()))
            wins_count = wins_result.scalar() or 0
            computed_win_rate = round((wins_count / total_trades * 100), 1)

            if computed_win_rate < min_win_rate:
                logger.debug(f"[DASHBOARD DEBUG] Trader {trader_id} skipped: {computed_win_rate}% < {min_win_rate}% win rate")
                continue

            # Compute additional aggregates
            total_pnl_result = await db.execute(select(func.sum(Trade.pnl)).select_from(trade_query.subquery()))
            total_pnl_dollars = total_pnl_result.scalar() or 0.0

            losses_result = await db.execute(select(func.count()).select_from(trade_query.where(Trade.pnl < 0).subquery()))
            losses = losses_result.scalar() or 0

            eligible_rows.append({
                'id': trader_id,
                'full_name': full_name,
                'strategy': strategy,
                'computed_win_rate': computed_win_rate,
                'total_trades': total_trades,
                'total_pnl': total_pnl_dollars,
                'losses': losses,
                'marketplace_price': marketplace_price
            })

            logger.debug(f"[DASHBOARD DEBUG] Trader {trader_id} eligible: {total_trades} trades, {computed_win_rate}% win")

        # Sort and limit
        eligible_rows.sort(key=lambda r: (-r['computed_win_rate'], -r['total_trades']))
        trader_rows = eligible_rows[:20]

        # UPDATED: Add debug log for raw query results
        logger.info(f"[DASHBOARD DEBUG] Eligible traders after filtering: {len(trader_rows)} rows")

        # TEMP DEBUG: Log raw row details
        logger.info(f"[DASHBOARD DEBUG] Trader rows raw: {[(r['id'], r['full_name'], r['total_trades'], r['computed_win_rate']) for r in trader_rows]}")

        # --- Build Trader Cards (FIXED: total_pnl as sum $, trend as cumulative $) ---
        traders = []
        for row in trader_rows:
            trader_id = row['id']
            full_name = row['full_name']
            strategy = row['strategy']
            computed_win_rate = row['computed_win_rate']
            total_trades = row['total_trades']
            total_pnl_dollars = row['total_pnl']
            losses = row['losses']
            marketplace_price = row['marketplace_price']

            names = full_name.split()
            trader_initials = (names[0][0].upper() +
                               (names[-1][0].upper() if len(names) > 1 else names[0][0].upper()))

            # --- Trend: Last 5 PnL $ (FIXED: Cumulative $ return) ---
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
                current += pnl  # Already $
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

            traders.append({
                'id': trader_id,
                'name': full_name,
                'initials': trader_initials,
                'strategy': strategy or 'Proven System',
                'win_rate': round(computed_win_rate or 0, 1),
                'trades': total_trades or 0,
                'losses': losses or 0,
                'pnl': round(total_pnl_dollars, 0),  # Pass as total $; round to whole $ for display
                'trend': trend,
                'journal_tease': journal_tease,
                'monthly_price': marketplace_price or 19.99,
                # Note: status added later per user
            })

        # Cache the base traders list
        await set_cache(redis, global_cache_key, json.dumps(traders), ttl=300)  # FIXED: Reduced TTL to 5 min for faster refresh
        logger.info("Cached eligible traders")
        logger.info(f"[DASHBOARD DEBUG] Fresh traders loaded: {len(traders)} total")
    else:
        traders = json.loads(traders_data)
        logger.info(f"[DASHBOARD DEBUG] Loaded cached traders: {len(traders)} total")

    # NEW: Overwrite traders in context (fresh even if per-user cache hit)
    context["traders"] = traders

    # --- Trader Status Dict (active + pending) ---
    trader_status = {}
    for sub in all_active_subs + pending_subs:
        if sub.trader_id:
            trader_status[sub.trader_id] = sub.status

    # Merge user-specific status into traders
    for trader in traders:
        trader['status'] = trader_status.get(trader['id'])

    recommendations = traders[:3]
    context["recommendations"] = recommendations  # Set after traders loaded

    # FIXED: Always override with fresh TP balance (and refresh user if needed) to handle admin updates
    await db.refresh(current_user)
    points_balance = current_user.trade_points if current_user.trade_points is not None else 0
    context["points_balance"] = points_balance

    # For non-cached or unpacked, ensure current_user is fresh
    context["current_user"] = current_user
    context["request"] = request
    context["now"] = datetime.utcnow()

    # --- Render Template ---
    return templates.TemplateResponse("dashboard.html", context)
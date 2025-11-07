# router/journal.py
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, nulls_last
from sqlalchemy.orm import aliased
from typing import List, Optional
import logging
import re
from pydantic import BaseModel, Field
from datetime import datetime, date, timedelta, timezone

from database import get_session
from models import models
import auth
from models.schemas import TradeResponse, TradeUpdate, PaginatedTrades, PriceUpdate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/journal", tags=["Journal"])

# High [Model Update]: Fallback for AssetType enum (uses str() for safety)
try:
    from models.models import AssetType  # Import enum if available
except ImportError:
    AssetType = None  # Fallback: Use string comparison

class TradeCreate(TradeUpdate):  # Reuse fields for creation
    pass  # Extend as needed (e.g., required fields)


@router.post("/trades", response_model=TradeResponse, status_code=201)
async def create_trade(
    create_data: TradeCreate = Body(...),
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Create a new trade for the current user."""
    correlation_id = f"journal_create_{datetime.utcnow().timestamp()}"
    logger.info("Creating trade: user=%s, data=%s", current_user.id, create_data.model_dump(), extra={"corr_id": correlation_id})

    # Validate/apply similar to update (symbol uppercase, etc.)
    trade_dict = create_data.model_dump(exclude_none=True)
    
    # Create new trade instance
    trade = models.Trade(owner_id=current_user.id, **trade_dict)
    
    # Default strategy
    if not trade.strategy:
        trade.strategy = 'Manual'
    
    # High [Model Update]: Always recalc P/L if fields present
    if (trade.entry_price is not None and trade.exit_price is not None and
        trade.position_size is not None):
        direction_str = str(trade.direction) if trade.direction else None  # Enum safety
        delta = (trade.exit_price - trade.entry_price) if direction_str in (None, 'LONG') else (trade.entry_price - trade.exit_price)
        leverage = trade.leverage or 1.0
        base_pnl = delta * trade.position_size * leverage
        # FIXED: Check against uppercase enum value
        asset_type_str = str(trade.asset_type) if trade.asset_type else None  # Enum safety
        if asset_type_str == 'FOREX':
            pip_factor = 100 if 'JPY' in (trade.symbol or '') else 10000
            base_pnl *= (abs(delta) * pip_factor / abs(delta)) * 10  # Approx pip value
        fees_val = trade.fees or 0.0
        trade.pnl = round(base_pnl - fees_val, 2)
    else:
        trade.pnl = 0.0
    
    # FIXED: Uppercase asset_type
    if trade.asset_type:
        trade.asset_type = trade.asset_type.upper()
    # Critical: TZ for trade_date
    if trade.trade_date:
        trade.trade_date = trade.trade_date.replace(tzinfo=timezone.utc)
    
    try:
        db.add(trade)
        await db.flush()  # Assign ID without commit
        
        # Update user balance incrementally
        pnl_delta = trade.pnl
        if pnl_delta != 0:
            current_user.account_balance += pnl_delta
            logger.info(f"Created trade {trade.id} for user {current_user.id}, added {pnl_delta:.2f} to balance (new: {current_user.account_balance:.2f})", extra={"corr_id": correlation_id})
        
        await db.commit()
        await db.refresh(trade)
        response = TradeResponse.model_validate(trade, from_attributes=True)
        logger.info(f"Created & serialized trade {trade.id}: {dict(response)}", extra={"corr_id": correlation_id})
    except Exception as e:
        logger.error("Create failed: %s", str(e), extra={"corr_id": correlation_id})
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create trade")
    
    return response

@router.get("/trades", response_model=PaginatedTrades)
async def get_trades(
    source: Optional[str] = Query("personal"),  # NEW: 'personal' or 'trader'
    trader_id: Optional[int] = Query(None),
    search: Optional[str] = Query(None),
    filter_type: Optional[str] = Query("all"),
    session: Optional[str] = Query(None),
    strategy: Optional[str] = Query(None),
    asset_type: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    sort: Optional[str] = Query("dateDesc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Fetch paginated trades for the current user with filtering and sorting. NEW: Support trader_id for subbed access (read-only)."""
    correlation_id = f"journal_{datetime.utcnow().timestamp()}"
    logger.info("Fetching trades: user=%s, source=%s, trader_id=%s, params=%s", current_user.id, source, trader_id, {
        'search': search, 'filter_type': filter_type, 'session': session, 'strategy': strategy,
        'asset_type': asset_type, 'date_from': date_from, 'date_to': date_to, 'sort': sort, 'page': page, 'page_size': page_size
    }, extra={"corr_id": correlation_id})

    if source == "trader" and trader_id:
        # NEW: Check active sub to trader
        sub_result = await db.execute(
            select(models.Subscription).where(
                models.Subscription.user_id == current_user.id,
                models.Subscription.trader_id == trader_id,
                models.Subscription.status == 'active'
            )
        )
        if not sub_result.scalar():
            raise HTTPException(status_code=403, detail="No active subscription to this trader")
        # Fetch trader's trades (owner_id = trader_id)
        query = select(models.Trade).where(models.Trade.owner_id == trader_id)
        trader_name = (await db.get(models.User, trader_id)).full_name or "Trader"
    else:
        # Personal trades
        query = select(models.Trade).where(models.Trade.owner_id == current_user.id)
        trader_name = None

    if search:
        search_term = f"%{search.lower()}%"
        query = query.filter(
            func.lower(models.Trade.symbol).like(search_term) |
            func.lower(models.Trade.notes).like(search_term)
        )

    if filter_type == "win":
        query = query.filter(models.Trade.pnl > 0)
    elif filter_type == "loss":
        query = query.filter(models.Trade.pnl < 0)

    if session:
        query = query.filter(models.Trade.session == session)

    if strategy:
        query = query.filter(models.Trade.strategy == strategy)

    if asset_type:
        # FIXED: Uppercase for enum match
        query = query.filter(models.Trade.asset_type == asset_type.upper())

    if date_from:
        try:
            df = datetime.strptime(date_from, '%Y-%m-%d')
            query = query.filter(models.Trade.trade_date >= df)
        except ValueError:
            pass
    if date_to:
        try:
            dt = datetime.strptime(date_to, '%Y-%m-%d')
            next_day = dt + timedelta(days=1)
            query = query.filter(models.Trade.trade_date < next_day)
        except ValueError:
            pass

    base_query = query

    count_query = select(func.count()).select_from(base_query.subquery())
    total_count = (await db.execute(count_query)).scalar() or 0

    wins_query = select(func.count()).select_from(base_query.where(models.Trade.pnl > 0).subquery())
    wins_count = (await db.execute(wins_query)).scalar() or 0

    avg_query = select(func.avg(models.Trade.pnl)).select_from(base_query.subquery())
    avg_pl_result = (await db.execute(avg_query)).scalar()
    avg_pl = avg_pl_result if avg_pl_result is not None else 0.0

    win_rate = (wins_count / total_count * 100) if total_count > 0 else 0.0

    if sort == "plHigh":
        query = query.order_by(models.Trade.pnl.desc())
    elif sort == "plLow":
        query = query.order_by(models.Trade.pnl.asc())
    elif sort == "symbol":
        query = query.order_by(models.Trade.symbol.asc())
    else:
        query = query.order_by(nulls_last(models.Trade.trade_date.desc()))

    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)

    try:
        # FIXED: Wrapped execution for enum/DB mismatch handling
        result = await db.execute(query)
        trades = result.scalars().all()
    except LookupError as le:
        if "'forex'" in str(le) or "'crypto'" in str(le):
            logger.error("Enum case mismatch detected. Run DB migration to uppercase asset_type values.", extra={"corr_id": correlation_id})
            raise HTTPException(status_code=500, detail="Data inconsistency; contact support for DB fix.")
        raise

    total_pages = (total_count + page_size - 1) // page_size

    if total_count == 0:
        logger.warning("No trades for user %s", current_user.id, extra={"corr_id": correlation_id})

    logger.info("Fetched %d trades, total=%d, page=%d", len(trades), total_count, page, extra={"corr_id": correlation_id})

    return PaginatedTrades(
        trades=[TradeResponse.model_validate(t, from_attributes=True) for t in trades],
        total=total_count,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        win_rate=win_rate,
        avg_pl=avg_pl,
        source=source,
        trader_name=trader_name  # NEW: For frontend badge
    )

@router.get("/trades/{trade_id}", response_model=TradeResponse)
async def get_trade(
    trade_id: int,
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Fetch a single trade by ID."""
    trade = await db.get(models.Trade, trade_id)
    if not trade or trade.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Trade not found")
    response = TradeResponse.model_validate(trade, from_attributes=True)
    logger.info(f"Serialized trade {trade_id}: {dict(response)}")  # Debug: Log full response fields
    return response

# High [Model Update]: Enhanced P/L calc with asset_type, fees; always recalc; null-skip + logging
@router.put("/trades/{trade_id}", response_model=TradeResponse)
async def update_trade(
    trade_id: int,
    update_data: TradeUpdate = Body(...),
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user)
):
    trade = await db.get(models.Trade, trade_id)
    if not trade or trade.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Trade not found")
    
    # Capture pre-update P&L for delta
    old_pnl = trade.pnl
    
    # Apply updates, skipping None and preventing nullification of non-null core fields
    update_dict = update_data.model_dump(exclude_none=True)
    core_fields = ['entry_price', 'exit_price', 'position_size']  # High: Protect historical
    skipped_fields = []  # For logging
    for key, value in update_dict.items():
        if not hasattr(trade, key):
            skipped_fields.append(key)
            logger.warning(f"Model missing field '{key}' for trade {trade_id}; skipping update")
            continue
        if key in core_fields and value is None and getattr(trade, key) is not None:
            logger.warning(f"Skipping nullification of {key} for trade {trade_id}")
            continue
        # FIXED: Uppercase asset_type for enum
        if key == 'asset_type' and value:
            value = value.upper()
        # Critical: TZ for trade_date
        if key == 'trade_date' and value:
            value = value.replace(tzinfo=timezone.utc)
        setattr(trade, key, value)
    
    if skipped_fields:
        logger.error(f"Update for trade {trade_id} skipped fields: {skipped_fields} - Add to model/DB!")
    
    # Ensure strategy defaults
    if not trade.strategy:
        trade.strategy = 'Manual'
    
    # High [Model Update]: Always recalc P/L if fields present
    if (trade.entry_price is not None and trade.exit_price is not None and
        trade.position_size is not None):
        direction_str = str(trade.direction) if trade.direction else None  # Enum safety
        delta = (trade.exit_price - trade.entry_price) if direction_str in (None, 'LONG') else (trade.entry_price - trade.exit_price)
        leverage = trade.leverage or 1.0
        base_pnl = delta * trade.position_size * leverage
        # FIXED: Check against uppercase enum value
        asset_type_str = str(trade.asset_type) if trade.asset_type else None  # Enum safety
        if asset_type_str == 'FOREX':
            pip_factor = 100 if 'JPY' in (trade.symbol or '') else 10000
            base_pnl *= (abs(delta) * pip_factor / abs(delta)) * 10  # Approx pip value
        fees_val = trade.fees or 0.0
        trade.pnl = round(base_pnl - fees_val, 2)
    else:
        trade.pnl = 0.0
    
    # Calculate delta for balance update
    pnl_delta = trade.pnl - old_pnl if old_pnl is not None else trade.pnl
    
    try:
        # Update user balance incrementally
        user = await db.get(models.User, current_user.id)  # Refresh user in session
        if user and pnl_delta != 0:
            user.account_balance += pnl_delta
            logger.info(f"Updated user {current_user.id} balance by {pnl_delta:.2f} (new: {user.account_balance:.2f})")
        
        await db.commit()
        await db.refresh(trade)
        await db.refresh(user)  # Refresh for response if needed
        response = TradeResponse.model_validate(trade, from_attributes=True)
        logger.info(f"Updated & serialized trade {trade_id}: {dict(response)}")  # Debug: Log post-save fields
    except Exception as e:
        logger.error("Update failed for trade %d: %s", trade_id, str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Update failed")
    
    return response

@router.delete("/trades/{trade_id}")
async def delete_trade(
    trade_id: int,
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Delete a trade and reverse its P/L impact on the user's balance."""
    trade = await db.get(models.Trade, trade_id)
    if not trade or trade.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Trade not found")
    
    pnl_to_reverse = trade.pnl or 0.0
    
    try:
        # Reverse impact on balance
        if pnl_to_reverse != 0:
            current_user.account_balance -= pnl_to_reverse
            logger.info(f"Deleted trade {trade_id} for user {current_user.id}, subtracted {pnl_to_reverse:.2f} from balance (new: {current_user.account_balance:.2f})")
        
        await db.delete(trade)
        await db.commit()
        logger.info(f"Deleted trade {trade_id}")
    except Exception as e:
        logger.error("Delete failed for trade %d: %s", trade_id, str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Delete failed")
    
    return {"message": "Trade deleted successfully"}

# Add this route to your journal router or a new one
@router.get("/subscriptions/active")
async def get_active_subscriptions(
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Return list of active trader subscriptions for frontend dropdown."""
    result = await db.execute(
        select(models.Subscription.trader_id, models.User.full_name.label("trader_name"))
        .join(models.User, models.User.id == models.Subscription.trader_id)
        .where(
            models.Subscription.user_id == current_user.id,
            models.Subscription.status == 'active'
        )
    )
    subs = result.all()
    return [{"trader_id": r.trader_id, "trader_name": r.trader_name} for r in subs]

# NEW: Marketplace Eligibility Check
@router.get("/eligibility")
async def check_eligibility(
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Check if the current user is eligible to become a marketplace trader."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Fetch config
    result_config = await db.execute(select(models.EligibilityConfig).where(models.EligibilityConfig.id == 1))
    config = result_config.scalar_one_or_none()
    if not config:
        config = models.EligibilityConfig(id=1, min_trades=50, min_win_rate=80.0, max_marketplace_price=99.99, trader_share_percent=70.0)  # Default with split
        db.add(config)
        await db.commit()

    min_trades = config.min_trades
    min_win_rate = config.min_win_rate
    trader_share_percent = config.trader_share_percent or 70.0

    # Calculate user's stats
    trade_query = select(models.Trade).where(models.Trade.owner_id == current_user.id)
    total_trades = (await db.execute(select(func.count()).select_from(trade_query.subquery()))).scalar() or 0
    wins_count = (await db.execute(select(func.count()).select_from(trade_query.where(models.Trade.pnl > 0).subquery()))).scalar() or 0
    win_rate = round((wins_count / total_trades * 100), 1) if total_trades > 0 else 0.0

    eligible = total_trades >= min_trades and win_rate >= min_win_rate

    return {
        "eligible": eligible,
        "current_trades": total_trades,
        "current_win_rate": win_rate,
        "required_trades": min_trades,
        "required_win_rate": min_win_rate,
        "is_trader": current_user.is_trader,
        "is_trader_pending": current_user.is_trader_pending,  # NEW
        "marketplace_price": current_user.marketplace_price or 19.99,  # Default display price
        "trader_share_percent": trader_share_percent  # NEW: Revenue split
    }

# NEW: Apply to Become Trader (Pending Review)
@router.post("/apply-trader")
async def apply_to_become_trader(
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Apply to become a marketplace trader if eligible. Sets pending for admin review."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if current_user.is_trader or current_user.is_trader_pending:
        raise HTTPException(status_code=400, detail="Application already submitted or approved.")

    # Re-check eligibility (same as above)
    result_config = await db.execute(select(models.EligibilityConfig).where(models.EligibilityConfig.id == 1))
    config = result_config.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=500, detail="Eligibility config not found.")

    min_trades = config.min_trades
    min_win_rate = config.min_win_rate

    trade_query = select(models.Trade).where(models.Trade.owner_id == current_user.id)
    total_trades = (await db.execute(select(func.count()).select_from(trade_query.subquery()))).scalar() or 0
    wins_count = (await db.execute(select(func.count()).select_from(trade_query.where(models.Trade.pnl > 0).subquery()))).scalar() or 0
    win_rate = round((wins_count / total_trades * 100), 1) if total_trades > 0 else 0.0

    if total_trades < min_trades or win_rate < min_win_rate:
        raise HTTPException(
            status_code=400,
            detail=f"Ineligible: Need {min_trades - total_trades} more trades and/or {min_win_rate - win_rate:.1f}% higher win rate."
        )

    # Set pending + notify admin
    current_user.is_trader_pending = True
    if not current_user.marketplace_price:
        current_user.marketplace_price = 19.99  # Default price
    current_user.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(current_user)

    # TODO: Email admin notification here (e.g., via your email service)
    logger.info(f"New trader application from user {current_user.id} ({current_user.email}) - pending review")

    return {
        "success": True,
        "message": "Application submitted! We'll review within 24-48 hours.",
        "win_rate": win_rate,
        "total_trades": total_trades,
        "status": "pending"
    }

@router.put("/marketplace-price")
async def update_marketplace_price(
    update_data: PriceUpdate,
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user),
):
    """Update the user's marketplace subscription price if they are a trader."""
    logger.info("Received price update request: %s", update_data.model_dump())
    price = update_data.price

    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not current_user.is_trader:
        raise HTTPException(status_code=400, detail="Must be a marketplace trader to set price.")

    # Fetch config
    result_config = await db.execute(
        select(models.EligibilityConfig).where(models.EligibilityConfig.id == 1)
    )
    config = result_config.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=500, detail="Config not found.")
    max_price = config.max_marketplace_price or 99.99

    min_price = 1.00

    if price < min_price or price > max_price:
        raise HTTPException(
            status_code=400,
            detail=f"Price must be ${min_price:.2f}–${max_price:.2f}."
        )

    current_user.marketplace_price = round(price, 2)
    current_user.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(current_user)

    return {
        "success": True,
        "message": f"Marketplace price updated to ${price:.2f}/month.",
        "new_price": price,
    }

# NEW: Earnings Summary for Traders
@router.get("/earnings")
async def get_earnings(
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Get marketplace earnings summary for the trader."""
    if not current_user.is_trader:
        raise HTTPException(status_code=403, detail="Access denied: Not a marketplace trader")

    total_earnings = current_user.marketplace_earnings or 0.0
    pending_payout = current_user.monthly_earnings or 0.0
    threshold = current_user.payout_threshold or 50.0
    has_wallet = bool(current_user.wallet_address)
    last_payout = current_user.last_payout_date
    today = date.today()

    # Determine if can request payout (monthly, above threshold, wallet set)
    can_request = (
        pending_payout >= threshold
        and has_wallet
        and (not last_payout or (last_payout.year != today.year or last_payout.month != today.month))
    )

    # Next payout date: 1st of next month
    next_month = today.replace(day=1) + timedelta(days=32)
    next_payout_date = next_month.replace(day=1)

    # Fetch config for split (for display, though earnings already reflect it)
    result_config = await db.execute(select(models.EligibilityConfig).where(models.EligibilityConfig.id == 1))
    config = result_config.scalar_one_or_none()
    trader_share_percent = config.trader_share_percent or 70.0 if config else 70.0

    return {
        "total_earnings": total_earnings,
        "pending_payout": pending_payout,
        "payout_threshold": threshold,
        "can_request": can_request,
        "has_wallet": has_wallet,
        "wallet_address": current_user.wallet_address,
        "last_payout_date": last_payout.isoformat() if last_payout else None,
        "next_payout_date": next_payout_date.isoformat(),
        "trader_share_percent": trader_share_percent  # NEW: Include split in earnings response
    }



# NEW: Request Monthly Payout
@router.post("/request-payout")
async def request_payout(
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Request monthly payout if eligible."""
    if not current_user.is_trader:
        raise HTTPException(status_code=403, detail="Access denied: Not a marketplace trader")

    if not current_user.wallet_address:
        raise HTTPException(status_code=400, detail="Wallet address required for payouts")

    pending = current_user.monthly_earnings or 0.0
    threshold = current_user.payout_threshold or 50.0
    today = date.today()
    last_payout = current_user.last_payout_date

    if pending < threshold:
        raise HTTPException(status_code=400, detail=f"Pending earnings (${pending:.2f}) below minimum threshold (${threshold:.2f})")
    if last_payout and last_payout.year == today.year and last_payout.month == today.month:
        raise HTTPException(status_code=400, detail="Payout already requested this month. Next available on the 1st of next month.")

    # Simulate payout processing (in production: integrate with payout service, record tx_hash)
    payout_amount = pending
    current_user.monthly_earnings = 0.0
    current_user.last_payout_date = today
    current_user.updated_at = datetime.utcnow()
    await db.commit()

    logger.info(f"Monthly payout processed for trader {current_user.id}: ${payout_amount:.2f} to {current_user.wallet_address}")

    return {
        "message": f"Monthly payout of ${payout_amount:.2f} requested and processed to your wallet ({current_user.wallet_address[:6]}...{current_user.wallet_address[-4:]}).",
        "payout_amount": payout_amount
    }
# router/journal.py
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, nulls_last
from sqlalchemy.orm import aliased
from typing import List, Optional
import logging
from pydantic import BaseModel, Field
from datetime import datetime, date, timedelta, timezone

from database import get_session
from models import models
import auth
from models.schemas import TradeResponse, TradeUpdate, PaginatedTrades  # Now includes PaginatedTrades

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/journal", tags=["Journal"])

# High [Model Update]: Fallback for AssetType enum (uses str() for safety)
try:
    from models.models import AssetType  # Import enum if available
except ImportError:
    AssetType = None  # Fallback: Use string comparison

@router.get("/trades", response_model=PaginatedTrades)
async def get_trades(
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
    """Fetch paginated trades for the current user with filtering and sorting."""
    correlation_id = f"journal_{datetime.utcnow().timestamp()}"
    logger.info("Fetching trades: user=%s, params=%s", current_user.id, {
        'search': search, 'filter_type': filter_type, 'session': session, 'strategy': strategy,
        'asset_type': asset_type, 'date_from': date_from, 'date_to': date_to, 'sort': sort, 'page': page, 'page_size': page_size
    }, extra={"corr_id": correlation_id})

    # Base query
    query = select(models.Trade).where(models.Trade.owner_id == current_user.id)

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
        avg_pl=avg_pl
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

@router.delete("/trades/{trade_id}")
async def delete_trade(
    trade_id: int,
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user)
):
    trade = await db.get(models.Trade, trade_id)
    if not trade or trade.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Trade not found")
    await db.delete(trade)
    await db.commit()
    return {"message": "Trade deleted successfully"}

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
    
    try:
        await db.commit()
        await db.refresh(trade)
        response = TradeResponse.model_validate(trade, from_attributes=True)
        logger.info(f"Updated & serialized trade {trade_id}: {dict(response)}")  # Debug: Log post-save fields
    except Exception as e:
        logger.error("Update failed for trade %d: %s", trade_id, str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Update failed")
    
    return response
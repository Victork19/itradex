# app_utils/points.py
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import case, update, select
from sqlalchemy.exc import SQLAlchemyError

# Assuming these are accessible; adjust imports based on project structure
from redis.asyncio import Redis
from redis_client import get_cache, set_cache  # Assuming these exist

import models.models as models  # Adjust to your models path
from database import get_session  # Not directly used here, but for context
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# NEW: Maximum Trade Points cap per user
MAX_TP_CAP = 1000  # Configurable cap; adjust as needed (e.g., via env var or DB config)

async def spend_trade_points(
    db: AsyncSession, 
    user: models.User, 
    action: str, 
    amount: int = 1, 
    redis: Optional[Redis] = None
) -> None:
    """Deduct Trade Points for an action (e.g., 'insight')."""
    current_points = user.trade_points or 0  # Coerce None to 0 for check
    if current_points < amount:
        raise HTTPException(status_code=402, detail=f"Insufficient Trade Points ({current_points} remaining). Refer friends to earn more!")
    
    # Direct SQL UPDATE for reliability (bypasses object state issues); handle None as 0
    stmt = (
        update(models.User)
        .where(models.User.id == user.id)
        .values(trade_points=case((models.User.trade_points.is_(None), 0 - amount), else_=models.User.trade_points - amount))
        .execution_options(synchronize_session="fetch")
    )
    await db.execute(stmt)
    
    # Log transaction
    tx = models.PointTransaction(
        user_id=user.id,
        type=action,
        amount=-amount,
        description=f"Spent {amount} TP on {action}"
    )
    db.add(tx)
    await db.flush()  # Flush tx before commit for atomicity
    await db.commit()
    
    # Refresh user object post-update for in-request consistency
    await db.refresh(user)
    
    # Invalidate insights cache (as credits/TP change)
    if redis:
        await redis.delete(f"insights:{user.id}")

    logger.info(f"Spent {amount} TP for user {user.id} on {action}; new balance: {user.trade_points}")

async def grant_trade_points(
    db: AsyncSession, 
    user: models.User, 
    action: str, 
    amount: int = 1, 
    description: Optional[str] = None,
    redis: Optional[Redis] = None
) -> None:
    """Grant Trade Points for an action (e.g., 'plan_upgrade'). Caps total at MAX_TP_CAP."""
    if amount <= 0:
        raise ValueError("Amount must be positive for granting points.")
    
    current_points = user.trade_points or 0  # Coerce None to 0
    potential_total = current_points + amount
    
    # NEW: Apply cap if exceeding limit
    if potential_total > MAX_TP_CAP:
        actual_amount = max(0, MAX_TP_CAP - current_points)
        if actual_amount == 0:
            logger.warning(f"User {user.id} at TP cap ({MAX_TP_CAP}); no points granted for {action}")
            return  # No grant needed; early exit
        else:
            amount = actual_amount  # Adjust amount
            logger.info(f"Capped grant for user {user.id}: {potential_total} -> {MAX_TP_CAP} TP ({amount} granted for {action})")
    else:
        logger.info(f"Full grant for user {user.id}: {amount} TP for {action}")
    
    # Direct SQL UPDATE for reliability (bypasses object state issues); handle None as 0
    stmt = (
        update(models.User)
        .where(models.User.id == user.id)
        .values(trade_points=case((models.User.trade_points.is_(None), amount), else_=models.User.trade_points + amount))
        .execution_options(synchronize_session="fetch")
    )
    await db.execute(stmt)
    
    # Log transaction (with actual granted amount)
    tx = models.PointTransaction(
        user_id=user.id,
        type=action,
        amount=amount,
        description=description or f"Granted {amount} TP for {action}" + (" (capped)" if amount < potential_total - current_points else "")
    )
    db.add(tx)
    await db.flush()  # Flush tx before commit for atomicity
    await db.commit()
    
    # Refresh user object post-update for in-request consistency
    await db.refresh(user)
    
    # NEW: Create notification for point grant (include cap info if applied)
    cap_msg = f" (You've reached the {MAX_TP_CAP} TP cap!)" if user.trade_points == MAX_TP_CAP else ""
    notif = models.Notification(
        user_id=user.id,
        title="Trade Points Granted!",
        message=f"You've received {amount} TP for {action}. New balance: {user.trade_points} TP.{cap_msg}",
        type="points_grant"
    )
    db.add(notif)
    await db.commit()
    
    # Invalidate relevant caches (e.g., insights, as TP change)
    if redis:
        await redis.delete(f"insights:{user.id}")

    logger.info(f"Granted {amount} TP for user {user.id} on {action}; new balance: {user.trade_points}")

async def get_upgrade_tp_amount(base_plan: str, db: AsyncSession) -> int:
    """Fetch upgrade TP amount for the plan from database."""
    config_id = 1 if base_plan == 'pro' else 2
    result = await db.execute(
        select(models.UpgradeTpConfig.amount).where(
            models.UpgradeTpConfig.plan == base_plan,
            models.UpgradeTpConfig.id == config_id
        )
    )
    amount = result.scalar()
    return amount if amount is not None else 0
# utils/discount.py  (create this file)
from datetime import date
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models.models import Discount

async def get_discount(db: AsyncSession, discount_id: int = 1) -> dict:
    """
    Returns the discount dict for the given id.
    If the discount is disabled or expired → returns a disabled discount.
    """
    result = await db.execute(select(Discount).where(Discount.id == discount_id))
    db_discount = result.scalar_one_or_none()

    if not db_discount:
        return {"enabled": False, "percentage": 0.0, "expiry": None}

    # auto-disable if expired
    enabled = db_discount.enabled and (
        db_discount.expiry is None or db_discount.expiry >= date.today()
    )
    return {
        "enabled": enabled,
        "percentage": db_discount.percentage if enabled else 0.0,
        "expiry": db_discount.expiry,
    }
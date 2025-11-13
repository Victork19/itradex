# router/notifications.py
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, desc, func
from sqlalchemy.orm import joinedload
from typing import List

from models.models import User, Notification
from models.schemas import NotificationsResponse  # NEW: Import the response model
from database import get_session
import auth

logger = logging.getLogger("iTrade")

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


@router.get("", response_model=NotificationsResponse)  # FIXED: Use the new model instead of List[dict]
async def get_notifications(
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session),
    unread_only: bool = False
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    query = select(Notification).where(Notification.user_id == current_user.id)
    if unread_only:
        query = query.where(Notification.is_read == False)
    query = query.order_by(desc(Notification.created_at)).limit(50)

    result = await db.execute(query)
    notifications = result.scalars().all()

    # Return as dicts for JSON (now validated by NotificationsResponse)
    notif_list = []
    for n in notifications:
        notif_list.append({
            "id": n.id,
            "title": n.title,
            "message": n.message,
            "type": n.type,
            "is_read": n.is_read,
            "created_at": n.created_at.strftime("%Y-%m-%d %H:%M:%S") if n.created_at else None
        })

    # Count unread
    unread_count_query = select(func.count()).select_from(Notification).where(
        Notification.user_id == current_user.id, Notification.is_read == False
    )
    unread_count = (await db.execute(unread_count_query)).scalar() or 0

    return {
        "notifications": notif_list,
        "unread_count": unread_count
    }


@router.post("/{notif_id}/read")
async def mark_as_read(
    notif_id: int,
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    result = await db.execute(
        select(Notification).where(Notification.id == notif_id, Notification.user_id == current_user.id)
    )
    notification = result.scalar_one_or_none()
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")

    if notification.is_read:
        return {"success": True, "message": "Already read"}

    notification.is_read = True
    # Note: No 'updated_at' set—add column/model if needed
    await db.commit()
    await db.refresh(notification)

    return {"success": True, "message": "Marked as read"}


@router.post("/mark_all_read")
async def mark_all_read(
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    await db.execute(
        update(Notification)
        .where(Notification.user_id == current_user.id, Notification.is_read == False)
        .values(is_read=True)  # No 'updated_at'—add if schema updated
    )
    await db.commit()

    return {"success": True, "message": "All notifications marked as read"}
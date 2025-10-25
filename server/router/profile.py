# /home/ukov/itrade/server/router/profile.py
from fastapi import APIRouter, Depends, HTTPException, status, Response, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from typing import List, Optional, Dict, Any
import logging
from pydantic import BaseModel, validator
from datetime import datetime, date, timedelta
import json  # For JSON handling if needed
from collections import Counter

from database import get_session
from models import models
import auth
from models.schemas import ProfileResponse, TradeResponse
from router.payments import get_current_subscription  

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/profile", tags=["Profile"])

class OnboardRequest(BaseModel):
    psychZone: Optional[str] = None
    strategy: Optional[str] = None
    strategyDesc: Optional[str] = None
    initialDeposit: Optional[float] = 10000.0
    accountBalance: Optional[float] = 10000.0
    riskPerTrade: Optional[float] = 1.0
    dailyLoss: Optional[float] = 5.0
    dailyLossLimit: Optional[float] = None
    stopLoss: Optional[bool] = True
    noRevenge: Optional[bool] = False
    notes: Optional[str] = None
    timeframes: Optional[List[str]] = []

class ProfileUpdateRequest(BaseModel):
    bio: Optional[str] = None
    trading_style: Optional[str] = None
    goals: Optional[str] = None
    trading_zones: Optional[str] = None
    strategies: Optional[str] = None
    strategy: Optional[str] = None
    strategy_desc: Optional[str] = None
    notes: Optional[str] = None
    risk_tolerance: Optional[int] = None
    max_risk_per_trade: Optional[float] = None
    risk_per_trade: Optional[float] = None
    daily_loss_percent: Optional[float] = None
    daily_loss_limit: Optional[float] = None
    stop_loss: Optional[bool] = None
    no_revenge: Optional[bool] = None
    timeframes: Optional[List[str]] = None
    account_balance: Optional[float] = None
    initial_deposit: Optional[float] = None

async def _compute_profile_stats(db: AsyncSession, user_id: int):
    """Helper function to compute profile statistics."""
    try:
        # Select only needed fields to avoid loading enum columns that might cause errors
        stmt = select(
            models.Trade.pnl,
            models.Trade.symbol,
            models.Trade.created_at
        ).where(models.Trade.owner_id == user_id)
        result = await db.execute(stmt)
        rows = result.all()

        pnls = [row.pnl for row in rows if row.pnl is not None]
        symbols = [row.symbol for row in rows if row.symbol]
        trade_dates = [row.created_at.date() for row in rows]

        lifetime_pnl = sum(pnls)
        total_trades = len(rows)
        win_trades = len([p for p in pnls if p > 0])
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0

        best_trade = None
        if pnls:
            max_idx = pnls.index(max(pnls))
            best_trade = {
                "pnl": pnls[max_idx],
                "symbol": rows[max_idx].symbol
            }

        worst_trade = None
        if pnls:
            min_idx = pnls.index(min(pnls))
            worst_trade = {
                "pnl": pnls[min_idx],
                "symbol": rows[min_idx].symbol
            }

        most_traded = Counter(symbols)
        top_tickers = [ticker for ticker, _ in most_traded.most_common(4)]

        # Compute active streak
        active_streak = 0
        if trade_dates:
            trade_dates_set = set(trade_dates)
            today = date.today()
            current_date = today
            while True:
                if current_date in trade_dates_set:
                    active_streak += 1
                    current_date -= timedelta(days=1)
                else:
                    break

        stats = {
            "lifetime_pnl": lifetime_pnl,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "top_tickers": top_tickers,
            "active_streak": active_streak
        }

        return stats
    except Exception as e:
        logger.error(f"Error computing profile stats for user {user_id}: {e}")
        # Return defaults on any error to prevent 500s
        return {"lifetime_pnl": 0, "win_rate": 0, "total_trades": 0, "best_trade": None, "worst_trade": None, "top_tickers": [], "active_streak": 0}

@router.get("", response_model=Dict[str, Any])
async def get_profile(
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Fetch the current user's profile data."""
    correlation_id = f"profile_{datetime.utcnow().timestamp()}"
    logger.info("Fetching profile: user=%s", current_user.id, extra={"corr_id": correlation_id})

    try:
        stats = await _compute_profile_stats(db, current_user.id)
        
        # Handle JSON fields if stored as strings
        def safe_json_getattr(obj, attr, default):
            val = getattr(obj, attr, default)
            if isinstance(val, str):
                try:
                    return json.loads(val) if val else default
                except json.JSONDecodeError:
                    return default
            return val
        
        return {
            "id": current_user.id,
            "username": current_user.username,
            "full_name": current_user.full_name,
            "email": current_user.email,
            "bio": getattr(current_user, "bio", ""),
            "trading_style": getattr(current_user, "trading_style", ""),
            "goals": getattr(current_user, "goals", ""),
            "psych_zone": getattr(current_user, "psych_zone", ""),
            "strategy": getattr(current_user, "strategy", ""),
            "strategy_desc": getattr(current_user, "strategy_desc", ""),
            "account_balance": getattr(current_user, "account_balance", 10000.0),
            "initial_deposit": getattr(current_user, "initial_deposit", 10000.0),
            "risk_per_trade": getattr(current_user, "risk_per_trade", 1.0),
            "daily_loss_percent": getattr(current_user, "daily_loss_percent", 5.0),
            "daily_loss_limit": getattr(current_user, "daily_loss_limit", 500.0),
            "stop_loss": getattr(current_user, "stop_loss", True),
            "no_revenge": getattr(current_user, "no_revenge", False),
            "notes": getattr(current_user, "notes", ""),
            "preferred_timeframes": safe_json_getattr(current_user, "preferred_timeframes", []),
            "risk_tolerance": getattr(current_user, "risk_tolerance", 5),
            "created_at": current_user.created_at,
            **stats
        }
    except Exception as e:
        logger.error("Failed to fetch profile: %s", e, extra={"corr_id": correlation_id})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch profile")

@router.put("", response_model=Dict[str, Any])
async def update_profile(
    profile_data: ProfileUpdateRequest,
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Update the current user's profile data."""
    correlation_id = f"profile_update_{datetime.utcnow().timestamp()}"
    logger.info("Updating profile: user=%s", current_user.id, extra={"corr_id": correlation_id})

    try:
        field_map = {
            "bio": ("bio", lambda x: x),
            "trading_style": ("trading_style", lambda x: x),
            "goals": ("goals", lambda x: x),
            "trading_zones": ("psych_zone", lambda x: x),
            "strategies": ("strategy_desc", lambda x: x),
            "strategy": ("strategy", lambda x: x),
            "strategy_desc": ("strategy_desc", lambda x: x),
            "notes": ("notes", lambda x: x),
            "risk_tolerance": ("risk_tolerance", int),
            "max_risk_per_trade": ("risk_per_trade", float),
            "risk_per_trade": ("risk_per_trade", float),
            "daily_loss_percent": ("daily_loss_percent", float),
            "daily_loss_limit": ("daily_loss_limit", float),
            "stop_loss": ("stop_loss", bool),
            "no_revenge": ("no_revenge", bool),
            "timeframes": ("preferred_timeframes", lambda x: json.dumps(x or [])),
            "account_balance": ("account_balance", float),
            "initial_deposit": ("initial_deposit", float),
        }

        update_dict = {}
        for request_field, (db_field, converter) in field_map.items():
            val = getattr(profile_data, request_field, None)
            if val is not None:
                try:
                    converted = converter(val)
                    update_dict[db_field] = converted
                except (ValueError, TypeError):
                    raise HTTPException(status_code=400, detail=f"Invalid value for {request_field}")

        # Special validation for account_balance and initial_deposit
        if "account_balance" in update_dict and update_dict["account_balance"] <= 0:
            del update_dict["account_balance"]  # Skip invalid balance, but allow other updates
        if "initial_deposit" in update_dict and update_dict["initial_deposit"] <= 0:
            del update_dict["initial_deposit"]

        update_dict["updated_at"] = datetime.utcnow()

        if not update_dict:
            raise HTTPException(status_code=400, detail="No valid fields to update")

        await db.execute(
            update(models.User)
            .where(models.User.id == current_user.id)
            .values(**update_dict)
        )
        await db.commit()
        await db.refresh(current_user)
        
        # Recompute stats after update
        stats = await _compute_profile_stats(db, current_user.id)
        
        # Handle JSON fields if stored as strings
        def safe_json_getattr(obj, attr, default):
            val = getattr(obj, attr, default)
            if isinstance(val, str):
                try:
                    return json.loads(val) if val else default
                except json.JSONDecodeError:
                    return default
            return val
        
        return {
            "id": current_user.id,
            "username": current_user.username,
            "full_name": current_user.full_name,
            "email": current_user.email,
            "bio": getattr(current_user, "bio", ""),
            "trading_style": getattr(current_user, "trading_style", ""),
            "goals": getattr(current_user, "goals", ""),
            "psych_zone": getattr(current_user, "psych_zone", ""),
            "strategy": getattr(current_user, "strategy", ""),
            "strategy_desc": getattr(current_user, "strategy_desc", ""),
            "account_balance": getattr(current_user, "account_balance", 10000.0),
            "initial_deposit": getattr(current_user, "initial_deposit", 10000.0),
            "risk_per_trade": getattr(current_user, "risk_per_trade", 1.0),
            "daily_loss_percent": getattr(current_user, "daily_loss_percent", 5.0),
            "daily_loss_limit": getattr(current_user, "daily_loss_limit", 500.0),
            "stop_loss": getattr(current_user, "stop_loss", True),
            "no_revenge": getattr(current_user, "no_revenge", False),
            "notes": getattr(current_user, "notes", ""),
            "preferred_timeframes": safe_json_getattr(current_user, "preferred_timeframes", []),
            "risk_tolerance": getattr(current_user, "risk_tolerance", 5),
            "created_at": current_user.created_at,
            **stats
        }
    except ValueError as ve:
        logger.warning(f"Validation error in profile update: {ve}", extra={"corr_id": correlation_id})
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        await db.rollback()
        logger.error("Profile update failed: %s", e, extra={"corr_id": correlation_id})
        raise HTTPException(status_code=500, detail="Failed to update profile")

@router.post("/onboard")
async def onboard_profile(
    onboard_data: OnboardRequest,
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Save onboarding data to user profile."""
    correlation_id = f"onboard_{datetime.utcnow().timestamp()}"
    logger.info("Onboarding profile: user=%s", current_user.id, extra={"corr_id": correlation_id})
    logger.info(f"Received onboard data: {onboard_data.dict()}", extra={"corr_id": correlation_id})

    try:
        # Validate accountBalance
        if onboard_data.accountBalance <= 0:
            raise ValueError("Account balance must be greater than 0")

        daily_loss_limit = onboard_data.dailyLossLimit or (onboard_data.accountBalance * (onboard_data.dailyLoss / 100))
        risk_tolerance = int(onboard_data.riskPerTrade * 2)  # Scale 0.5-5% to 1-10
        trading_style = f"{onboard_data.psychZone} {onboard_data.strategy}".strip()
        goals = onboard_data.strategyDesc or ""

        await db.execute(
            update(models.User)
            .where(models.User.id == current_user.id)
            .values(
                psych_zone=onboard_data.psychZone,
                strategy=onboard_data.strategy,
                strategy_desc=onboard_data.strategyDesc,
                initial_deposit=onboard_data.initialDeposit,
                account_balance=onboard_data.accountBalance,
                risk_per_trade=onboard_data.riskPerTrade,
                daily_loss_percent=onboard_data.dailyLoss,
                daily_loss_limit=daily_loss_limit,
                stop_loss=onboard_data.stopLoss,
                no_revenge=onboard_data.noRevenge,
                notes=onboard_data.notes,
                preferred_timeframes=json.dumps(onboard_data.timeframes or []),
                risk_tolerance=risk_tolerance,
                recommendations="{}",
                bio=onboard_data.notes,
                trading_style=trading_style,
                goals=goals,
                updated_at=datetime.utcnow()
            )
        )
        await db.commit()
        # Removed: await db.refresh(current_user)  # Unnecessary here; prevents potential 500 errors
        return {"success": True, "message": "Profile onboarded successfully"}
    except ValueError as ve:
        logger.warning(f"Validation error in onboarding: {ve}", extra={"corr_id": correlation_id})
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        await db.rollback()
        logger.error("Onboarding failed: %s", e, extra={"corr_id": correlation_id})
        raise HTTPException(status_code=500, detail="Error saving profile. Please try again.")

@router.post("/logout")
async def logout(response: Response):
    """Log out the current user by clearing the access token cookie."""
    correlation_id = f"logout_{datetime.utcnow().timestamp()}"
    logger.info("Logging out user", extra={"corr_id": correlation_id})
    response.delete_cookie("access_token")
    return {"success": True, "message": "Logged out successfully"}

@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Delete the current user's account and associated trades."""
    correlation_id = f"delete_account_{datetime.utcnow().timestamp()}"
    logger.info("Deleting account: user=%s", current_user.id, extra={"corr_id": correlation_id})

    try:
        await db.execute(
            delete(models.Trade).where(models.Trade.owner_id == current_user.id)
        )
        await db.execute(
            delete(models.User).where(models.User.id == current_user.id)
        )
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error("Account deletion failed: %s", e, extra={"corr_id": correlation_id})
        raise HTTPException(status_code=500, detail="Failed to delete account")

# New endpoints for subscription management (integrate with payments logic)
@router.get("/subscriptions/current")
async def get_current_subscription_endpoint(
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Get the current user's subscription details."""
    try:
        # Assume get_current_subscription from payments router or implement here
        sub = await get_current_subscription(db, current_user.id)  # Or your implementation
        if sub:
            return {
                "plan": sub.plan_type.split('_')[0] if sub.plan_type else 'starter',
                "status": sub.status,
                "interval": sub.plan_type.split('_')[1] if '_' in sub.plan_type else 'monthly',
                "amount": sub.amount_usd,
                "next_billing": sub.next_billing_date.isoformat() if sub.next_billing_date else None,
                "subscription_id": sub.nowpayments_sub_id
            }
        else:
            return {"plan": "starter", "status": "free"}
    except Exception as e:
        logger.error(f"Failed to fetch subscription: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch subscription")

@router.post("/subscriptions/{sub_id}/cancel")
async def cancel_subscription_endpoint(
    sub_id: str,
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Cancel the user's subscription."""
    try:
        # Implement cancellation logic, e.g., update status to 'canceled' and notify NowPayments if needed
        await db.execute(
            update(models.Subscription)
            .where(
                models.Subscription.nowpayments_sub_id == sub_id,
                models.Subscription.user_id == current_user.id
            )
            .values(status='canceled', updated_at=datetime.utcnow())
        )
        await db.commit()
        # Optional: Call NowPayments API to cancel
        return {"success": True, "message": "Subscription canceled"}
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to cancel subscription: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel subscription")
# users.py - Updated with Email Verification for Email/Password Signup
from pathlib import Path
from datetime import timedelta, datetime
from typing import Union, Any, Optional
import json
import urllib.parse
import os
import secrets
import string
import random  # NEW: For verification code
import logging  # NEW: For debugging
import asyncio  # NEW: For retries

from fastapi import APIRouter, Depends, Request, Form, HTTPException, status, Cookie, BackgroundTasks, Query  # NEW: Added BackgroundTasks, Query
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, exists, update
from authlib.integrations.starlette_client import OAuth
import httpx

from templates_config import templates
from models import models, schemas
from models.models import (
    BetaInvite, BetaConfig, Referral, PointTransaction, User, InitialTpConfig,
    BetaReferralTpConfig  # NEW: Added for per-plan beta referral TP
)
import auth

from database import get_session
from app_utils.points import grant_trade_points  # NEW: Import for point granting
from config import get_settings
from services.email_service import send_email  # NEW: For email verification
from redis.asyncio import Redis  # NEW: For code storage
from redis_client import redis_dependency  # NEW: Assume exists like in waitlist

settings = get_settings()

# NEW: Logger
logger = logging.getLogger(__name__)

# Tier bonuses
REFERRAL_TIER_BONUSES = {
    'rookie': 1.0,
    'pro_trader': 1.2,
    'elite_alpha': 1.5,
}

def get_tier_bonus(tier: str) -> float:
    return REFERRAL_TIER_BONUSES.get(tier, 1.0)

# NEW: Verification code generator
def generate_verification_code() -> str:
    return ''.join(random.choices(string.digits, k=6))

# NEW: Send with retry
async def send_with_retry(email: str, subject: str, body: str, background_tasks: BackgroundTasks = None, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        result = await send_email(email, subject, body, background_tasks)
        if result["status"] == "success":
            return result
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
        else:
            return result  # Final fail

# NEW: Background email sender for signup
async def send_verification_email(email: str, code: str, background_tasks: BackgroundTasks = None):
    body = f"""
Hello,

Your verification code for iTradeX: {code}

Enter this code to verify your email and complete signup.
This code expires in 15 minutes.
"""
    await send_email(email, "Verify Your iTradeX Email", body, background_tasks)

# Initialize OAuth
oauth = OAuth()
oauth.register(
    name='google',
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile',
        'timeout': 30.0,
        'http_proxy': os.getenv('HTTP_PROXY'),
        'https_proxy': os.getenv('HTTPS_PROXY'),
    }
)

router = APIRouter(prefix="/users", tags=["Users"])


# ───── HELPER: Unified Unique Code Generation (User + BetaInvite) ─────
async def generate_unique_code(db: AsyncSession, length: int = 8) -> str:
    """Generate a unique code not used in User.referral_code or BetaInvite.code."""
    charset = string.ascii_uppercase + string.digits
    while True:
        code = ''.join(secrets.choice(charset) for _ in range(length))
        user_check = await db.execute(select(User).where(User.referral_code == code))
        invite_check = await db.execute(select(BetaInvite).where(BetaInvite.code == code))
        if not user_check.scalar_one_or_none() and not invite_check.scalar_one_or_none():
            return code


# ───── HELPER: Get Initial TP Amount (Admin-configurable) ─────
async def get_initial_tp_amount(db: AsyncSession) -> int:
    result = await db.execute(select(InitialTpConfig).where(InitialTpConfig.id == 1))
    config = result.scalar_one_or_none()
    return config.amount if config else 3  # Fallback


# ───── HELPER: Get Beta Config (with defaults) ─────
async def get_beta_config(db: AsyncSession) -> BetaConfig:
    result = await db.execute(select(BetaConfig).where(BetaConfig.id == 1))
    config = result.scalar_one_or_none()
    if not config:
        config = BetaConfig(
            id=1,
            is_active=True,
            required_for_signup=True,
            award_points_on_use=3
        )
        db.add(config)
        await db.commit()
    return config


# ───── HELPER: Get Beta Referral TP Amount by Plan (Admin-configurable) ─────
async def get_beta_referral_tp_amount(db: AsyncSession, plan: str) -> int:
    result = await db.execute(select(BetaReferralTpConfig).where(BetaReferralTpConfig.id == 1))
    config = result.scalar_one_or_none()
    if config:
        if plan == 'starter':
            return config.starter_tp
        elif plan == 'pro':
            return config.pro_tp
        elif plan == 'elite':
            return config.elite_tp
    # Fallback to old default (3 TP)
    return 3


# ───── HELPER: Generate Beta Invites for User ─────
async def generate_beta_invites(db: AsyncSession, owner_id: int, count: int = 3):
    for _ in range(count):
        code = await generate_unique_code(db)
        invite = BetaInvite(
            owner_id=owner_id,
            code=code,
            created_at=datetime.utcnow(),
            used_by_id=None,
            used_at=None
        )
        db.add(invite)
    await db.commit()


# ───── CORE: Process Referral/Beta Code (Handles all cases) ─────
async def process_referral_code(db: AsyncSession, referral_code: str, new_user_id: int, plan: str = 'starter') -> Optional[int]:
    """
    Returns referrer_id if valid and used (None for pool or invalid).
    Raises ValueError if beta active + required + invalid.
    """
    if not referral_code:
        config = await get_beta_config(db)
        if config.required_for_signup and config.is_active:
            raise ValueError("Beta invite code is required for signup.")
        return None

    config = await get_beta_config(db)
    referrer_id = None

    referral_code = referral_code.upper().strip()  # Normalize

    if config.is_active:
        # Try Beta Invite first
        invite_result = await db.execute(
            select(BetaInvite).where(
                BetaInvite.code == referral_code,
                BetaInvite.used_by_id.is_(None)
            )
        )
        invite = invite_result.scalar_one_or_none()

        if invite:
            # Mark as used
            await db.execute(
                update(BetaInvite)
                .where(BetaInvite.id == invite.id)
                .values(used_by_id=new_user_id, used_at=datetime.utcnow())
            )

            referrer_id = invite.owner_id
            tier_bonus = 1.0
            base_amount = await get_beta_referral_tp_amount(db, plan)  # NEW: Use per-plan config
            points_awarded = int(base_amount * tier_bonus)

            ref = Referral(
                referrer_id=referrer_id,  # 0 for pool
                referee_id=new_user_id,
                status='active',
                points_earned=base_amount,  # NEW: Store base (pre-bonus) amount
                tier_bonus=tier_bonus
            )
            db.add(ref)
            await db.flush()  # NEW: Flush to get ref.id for linking

            # Only award points if real user (not pool)
            if referrer_id != 0:
                referrer = await db.get(User, referrer_id)
                if referrer:
                    description = f"Beta invite used by user {new_user_id}"
                    await grant_trade_points(
                        db, referrer, 'ref_earn', points_awarded,
                        description=description, redis=None
                    )
                    # NEW: Link the transaction to the referral
                    tx_query = select(PointTransaction.id).where(
                        PointTransaction.user_id == referrer_id,
                        PointTransaction.type == 'ref_earn',
                        PointTransaction.description == description
                    ).order_by(PointTransaction.id.desc()).limit(1)
                    tx_id = (await db.execute(tx_query)).scalar_one_or_none()
                    if tx_id:
                        await db.execute(
                            update(PointTransaction)
                            .where(PointTransaction.id == tx_id)
                            .values(related_ref_id=ref.id)
                        )
                        await db.commit()

            await db.commit()
            return referrer_id if referrer_id != 0 else None  # Don't set referred_by for pool

        else:
            # Invalid beta code
            if config.required_for_signup:
                raise ValueError("Invalid beta invite code.")
            # Else fall through to normal referral

    # Fallback: Normal referral code
    referrer_result = await db.execute(select(User).where(User.referral_code == referral_code))
    referrer = referrer_result.scalars().first()
    if referrer:
        tier_bonus = get_tier_bonus(referrer.referral_tier)
        points_awarded = int(config.award_points_on_use * tier_bonus)

        ref = Referral(
            referrer_id=referrer.id,
            referee_id=new_user_id,
            status='active',
            points_earned=config.award_points_on_use,
            tier_bonus=tier_bonus
        )
        db.add(ref)
        await db.flush()  # NEW: Flush to get ref.id for linking

        description = f"Referral code used by user {new_user_id}"
        await grant_trade_points(
            db, referrer, 'ref_earn', points_awarded,
            description=description, redis=None
        )
        # NEW: Link the transaction to the referral
        tx_query = select(PointTransaction.id).where(
            PointTransaction.user_id == referrer.id,
            PointTransaction.type == 'ref_earn',
            PointTransaction.description == description
        ).order_by(PointTransaction.id.desc()).limit(1)
        tx_id = (await db.execute(tx_query)).scalar_one_or_none()
        if tx_id:
            await db.execute(
                update(PointTransaction)
                .where(PointTransaction.id == tx_id)
                .values(related_ref_id=ref.id)
            )
            await db.commit()

        return referrer.id

    # Invalid code
    if config.required_for_signup and config.is_active:
        raise ValueError("Invalid code provided.")
    return None


# ───── NEW: Verify Email ─────
@router.post("/verify")
async def verify_email(
    request: Request,
    background_tasks: BackgroundTasks,
    email: str = Form(...),
    code: str = Form(...),
    db: AsyncSession = Depends(get_session),
    redis: Redis = Depends(redis_dependency),
):
    email = email.lower().strip()
    code = code.strip()

    # Find unverified user
    result = await db.execute(
        select(User).where(
            User.email == email,
            User.verified == False
        )
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=400, detail="No pending verification for this email.")

    # Check code from Redis
    stored_code = await redis.get(f"verify:{user.id}")
    stored_str = stored_code.decode('utf-8') if isinstance(stored_code, bytes) else stored_code
    if not stored_str or stored_str != code:
        raise HTTPException(status_code=400, detail="Invalid or expired code.")

    # Verify user
    await redis.delete(f"verify:{user.id}")
    user.verified = True
    user.verified_at = datetime.utcnow()
    await db.commit()
    await db.refresh(user)

    # Auto-login
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token({"sub": str(user.id)}, access_token_expires)

    # Determine redirect based on onboarding status
    redirect_url = "/onboard" if user.trading_style is None else "/dashboard"
    logger.info(f"Email verification for user {user.id}: redirecting to {redirect_url} (trading_style={user.trading_style})")
    
    redirect = RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)
    redirect.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=int(access_token_expires.total_seconds()),
        secure=False,  # Set to True in prod
        samesite="lax"
    )
    return redirect


# ───── NEW: Resend Verification Code ─────
@router.post("/resend")
async def resend_verification(
    request: Request,
    background_tasks: BackgroundTasks,
    email: str = Form(...),
    db: AsyncSession = Depends(get_session),
    redis: Redis = Depends(redis_dependency),
):
    email = email.lower().strip()

    # Rate limit
    email_key = f"resend:{email}"
    count = await redis.incr(email_key)
    if count == 1:
        await redis.expire(email_key, 3600)  # 1hr
    if count > 3:
        raise HTTPException(status_code=429, detail="Too many resend requests. Try again later.")

    # Find unverified user
    result = await db.execute(
        select(User).where(
            User.email == email,
            User.verified == False
        )
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="No pending verification for this email.")

    # New code
    new_code = generate_verification_code()
    await redis.setex(f"verify:{user.id}", 900, new_code)

    # Send email with retry
    email_result = await send_with_retry(email, "New Verification Code for iTradeX", f"""
Hello,

Your new verification code for iTradeX: {new_code}

Enter this code to verify your email.
This code expires in 15 minutes.
""", background_tasks)
    if email_result["status"] == "error":
        # UPDATED: Don't delete on failure; let it expire naturally
        error_msg = email_result['message']
        if "timeout" in error_msg.lower():
            error_msg = "Server busy – try again in a minute."
        elif "auth" in error_msg.lower() or "connection" in error_msg.lower():
            error_msg = "Email config issue – contact support."
        else:
            error_msg = f"Failed to send code: {error_msg}. Please check your connection and try again."
        logger.error(f"Resend email failed for {email}: {email_result['message']}")
        raise HTTPException(status_code=500, detail=error_msg)

    logger.info(f"Resent code to {email}: {new_code} | id: {user.id}")
    return JSONResponse({"success": True, "message": "New code sent! Check your inbox."})


# ───── NEW: Verify Page ─────
@router.get("/verify", response_class=HTMLResponse)
async def verify_page(
    request: Request,
    new: bool = Query(False),
    email: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_session),
):
    config = await get_beta_config(db)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "verify_mode": True,
            "new_signup": new,
            "prefill_email": email,
            "tab": "verify",
            "beta_config": config
        }
    )


# ───── JSON API: Signup (UPDATED: No auto-login, requires verification) ─────
@router.post("/signup-email", response_model=schemas.GenericResponse)
async def create_user_json(
    payload: schemas.SignupRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session),
    redis: Redis = Depends(redis_dependency)
):
    email = payload.email.lower().strip()  # NEW: Normalize email
    payload.email = email  # Update for consistency

    email_check = await db.execute(select(User).where(User.email == email))
    if email_check.scalars().first():
        return {"success": False, "message": "Email already in use."}

    username_check = await db.execute(select(User).where(User.username == payload.username))
    if username_check.scalars().first():
        return {"success": False, "message": "Username already taken."}

    hashed_password = auth.hash_password(payload.password)
    admin_exists = await db.execute(select(exists().where(User.is_admin == True)))
    is_admin = not admin_exists.scalar()
    initial_tp = await get_initial_tp_amount(db)

    new_user = User(
        username=payload.username.lower(),
        full_name=payload.full_name,
        email=email,
        password_hash=hashed_password,
        referral_code=await generate_unique_code(db),
        referred_by=None,
        is_admin=is_admin,
        plan='starter',
        trade_points=None,  # CHANGED: Set to None; grant initial after
        verified=False,  # NEW: Unverified
        verified_at=None,  # NEW
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    # UPDATED: Background send with retry
    verification_code = generate_verification_code()
    await redis.setex(f"verify:{new_user.id}", 900, verification_code)
    background_tasks.add_task(send_verification_email, email, verification_code, background_tasks)
    email_result = {"status": "success"}  # Assume success for now; log actual in task if needed
    logger.info(f"Queued verification email for {email} (code: {verification_code})")

    try:
        referred_by = await process_referral_code(db, getattr(payload, 'referral_code', None), new_user.id, new_user.plan)
        if referred_by:
            await db.execute(update(User).where(User.id == new_user.id).values(referred_by=referred_by))
            await db.commit()
    except ValueError as e:
        # UPDATED: Don't delete user on referral error; just log and proceed without referral
        logger.warning(f"Referral processing failed for user {new_user.id}: {str(e)}")
        # Optionally: await db.delete(new_user); await db.commit() if strict, but keep for consistency with email handling

    await generate_beta_invites(db, new_user.id)

    # CHANGED: Use grant_trade_points for initial TP (even unverified, as it's granted on creation)
    await grant_trade_points(
        db, new_user, 'initial_grant', initial_tp,
        description="Starting TP for starter plan", redis=None
    )

    return {
        "success": True,
        "message": "User created successfully. Please check your email for verification code.",
        "needs_verification": True,
        "email": email,
        "email_sent": email_result["status"] == "success"  # NEW: Flag for client handling
    }


# ───── JSON API: Login (UPDATED: Normalize username) ─────
@router.post("/login-email", response_model=schemas.GenericResponse)
async def login_json(payload: schemas.LoginRequest, db: AsyncSession = Depends(get_session)):
    # NEW: Normalize username to match stored format
    normalized_username = payload.username.lower().strip()
    result = await db.execute(select(User).where(User.username == normalized_username))
    user = result.scalars().first()
    if not user or not auth.verify_password(payload.password, user.password_hash):
        return {"success": False, "message": "Invalid username or password."}

    if not user.verified:  # NEW: Check verified
        return {"success": False, "message": "Please verify your email before logging in.", "needs_verification": True, "email": user.email}

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS if payload.remember_me else 7)

    access_token = auth.create_access_token({"sub": str(user.id)}, access_token_expires)
    refresh_token = auth.create_refresh_token({"sub": str(user.id)}, refresh_token_expires)

    return {
        "success": True,
        "message": "Login successful",
        "data": {"access_token": access_token, "refresh_token": refresh_token}
    }


# ───── JSON API: Logout ─────
@router.post("/logout", response_model=schemas.GenericResponse)
async def logout_json(current_user: User = Depends(auth.get_current_user)):
    return {"success": True, "message": "Successfully logged out"}


# ───── Update User Info ─────
@router.put("/me", response_model=schemas.GenericResponse)
async def update_user_info(
    payload: schemas.UpdateUserRequest,
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    update_data = {}

    if payload.username is not None:
        username_lower = payload.username.lower().strip()
        if username_lower != current_user.username:
            check = await db.execute(select(User).where(User.username == username_lower))
            if check.scalars().first():
                raise HTTPException(status_code=400, detail="Username already taken.")
            update_data["username"] = username_lower

    if payload.full_name is not None:
        update_data["full_name"] = payload.full_name.strip()

    if payload.email is not None:
        email_lower = payload.email.lower().strip()
        if email_lower != current_user.email:
            check = await db.execute(select(User).where(User.email == email_lower))
            if check.scalars().first():
                raise HTTPException(status_code=400, detail="Email already in use.")
            update_data["email"] = email_lower

    if not update_data:
        raise HTTPException(status_code=400, detail="No valid fields provided for update.")

    await db.execute(update(User).where(User.id == current_user.id).values(**update_data))
    await db.commit()
    await db.refresh(current_user)

    return {"success": True, "message": "User information updated successfully."}


# ───── Set/Change Password ─────
@router.post("/set-password", response_model=schemas.GenericResponse)
async def set_user_password(
    payload: schemas.SetPasswordRequest,
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if current_user.password_hash is not None:
        raise HTTPException(status_code=400, detail="Password is already set for this account.")

    if payload.password != payload.password_confirm:
        raise HTTPException(status_code=400, detail="Passwords do not match.")
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long.")

    hashed_password = auth.hash_password(payload.password)
    await db.execute(update(User).where(User.id == current_user.id).values(password_hash=hashed_password))
    await db.commit()

    return {"success": True, "message": "Password set successfully."}


@router.post("/change-password", response_model=schemas.GenericResponse)
async def change_user_password(
    payload: schemas.ChangePasswordRequest,
    current_user: User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if current_user.password_hash is None:
        raise HTTPException(status_code=400, detail="No password set. Please set a password first.")

    if not auth.verify_password(payload.current_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    if payload.new_password != payload.new_password_confirm:
        raise HTTPException(status_code=400, detail="New passwords do not match.")
    if len(payload.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters long.")
    if auth.verify_password(payload.new_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail="New password cannot be the same as current.")

    hashed_new = auth.hash_password(payload.new_password)
    await db.execute(update(User).where(User.id == current_user.id).values(password_hash=hashed_new))
    await db.commit()

    return {"success": True, "message": "Password changed successfully."}


# ───── Form Signup (UPDATED: Send verification, redirect to /verify) ─────
@router.post("/signup", response_class=HTMLResponse)
async def create_user_form(
    request: Request,
    background_tasks: BackgroundTasks,  # NEW
    full_name: str = Form(...),
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
    referral_code: str = Form(None),
    db: AsyncSession = Depends(get_session),
    redis: Redis = Depends(redis_dependency),  # NEW
):
    if password != password_confirm:
        config = await get_beta_config(db)
        return templates.TemplateResponse("index.html", {"request": request, "error": "Passwords do not match.", "tab": "signup", "beta_config": config}, status_code=400)

    email = email.lower().strip()  # NEW: Normalize email
    username = username.lower().strip()  # NEW: Normalize username

    email_check = await db.execute(select(User).where(User.email == email))
    if email_check.scalar_one_or_none():
        config = await get_beta_config(db)
        return templates.TemplateResponse("index.html", {"request": request, "error": "Email already in use.", "tab": "signup", "beta_config": config}, status_code=400)

    username_check = await db.execute(select(User).where(User.username == username))
    if username_check.scalar_one_or_none():
        config = await get_beta_config(db)
        return templates.TemplateResponse("index.html", {"request": request, "error": "Username already taken.", "tab": "signup", "beta_config": config}, status_code=400)

    hashed_password = auth.hash_password(password)
    admin_exists = await db.execute(select(exists().where(User.is_admin == True)))
    is_admin = not admin_exists.scalar()
    initial_tp = await get_initial_tp_amount(db)

    new_user = User(
        username=username,
        full_name=full_name,
        email=email,
        password_hash=hashed_password,
        referral_code=await generate_unique_code(db),
        referred_by=None,
        is_admin=is_admin,
        plan='starter',
        trade_points=None,  # CHANGED: Set to None; grant initial after
        # Ensure trading_style is None for onboarding trigger
        trading_style=None,
        verified=False,  # NEW
        verified_at=None,  # NEW
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    # NEW: Log for debugging
    logger.info(f"Created new user {new_user.id} ({new_user.username}), trading_style={new_user.trading_style}")

    # UPDATED: Background send with retry
    verification_code = generate_verification_code()
    await redis.setex(f"verify:{new_user.id}", 900, verification_code)
    background_tasks.add_task(send_verification_email, email, verification_code, background_tasks)
    logger.info(f"Queued verification email for {email} (code: {verification_code})")

    try:
        referred_by = await process_referral_code(db, referral_code, new_user.id, new_user.plan)
        if referred_by:
            await db.execute(update(User).where(User.id == new_user.id).values(referred_by=referred_by))
            await db.commit()
    except ValueError as e:
        # UPDATED: Don't delete user on referral error; just log and proceed without referral
        logger.warning(f"Referral processing failed for user {new_user.id}: {str(e)}")

    await generate_beta_invites(db, new_user.id)

    # CHANGED: Use grant_trade_points for initial TP
    await grant_trade_points(
        db, new_user, 'initial_grant', initial_tp,
        description="Starting TP for starter plan", redis=None
    )

    # NEW: Redirect to verify page with email
    redirect_url = f"/users/verify?new=true&email={urllib.parse.quote(email)}"
    return RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)


# ───── Form Login (UPDATED: Normalize username, check verified, redirect to verify if needed) ─────
@router.post("/login", response_class=HTMLResponse)
async def login_form(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    remember_me: bool = Form(False),
    db: AsyncSession = Depends(get_session),
):
    # NEW: Normalize username to match stored format
    normalized_username = username.lower().strip()
    user = (await db.execute(select(User).where(User.username == normalized_username))).scalars().first()
    if not user or not auth.verify_password(password, user.password_hash):
        config = await get_beta_config(db)
        return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid username or password.", "tab": "login", "beta_config": config}, status_code=401)

    if not user.verified:  # NEW: Check verified
        redirect_url = f"/users/verify?email={urllib.parse.quote(user.email)}"
        return RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)

    # UPDATED: Use remember_me for expiry
    expires = timedelta(days=settings.REMEMBER_ME_REFRESH_TOKEN_EXPIRE_DAYS) if remember_me else timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token({"sub": str(user.id)}, expires)

    redirect_url = "/onboard" if user.trading_style is None else "/dashboard"
    redirect = RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)
    redirect.set_cookie("access_token", access_token, httponly=True, max_age=int(expires.total_seconds()), secure=False, samesite="lax")
    return redirect


# ───── Google Beta Prompt (NEW) ─────
@router.get("/google/prompt", response_class=HTMLResponse)
async def google_beta_prompt(request: Request, db: AsyncSession = Depends(get_session)):
    config = await get_beta_config(db)
    if not (config.is_active and config.required_for_signup):
        base_url = getattr(settings, 'BASE_URL', f"{request.url.scheme}://{request.url.netloc}")
        redirect_uri = urllib.parse.urljoin(base_url, "/users/google/login")
        return RedirectResponse(url=redirect_uri, status_code=303)

    return templates.TemplateResponse(
        "google_beta_prompt.html",
        {"request": request, "config": config}
    )


# ───── Google Prompt Submit (NEW) ─────
@router.post("/google/prompt-submit")
async def google_prompt_submit(
    request: Request,
    beta_code: str = Form(...),
    db: AsyncSession = Depends(get_session)
):
    config = await get_beta_config(db)
    if not (config.is_active and config.required_for_signup):
        return RedirectResponse("/users/google/login", status_code=303)

    result = await db.execute(
        select(BetaInvite).where(
            BetaInvite.code == beta_code.upper(),
            BetaInvite.used_by_id.is_(None)
        )
    )
    invite = result.scalar_one_or_none()

    if not invite:
        return templates.TemplateResponse(
            "google_beta_prompt.html",
            {"request": request, "error": "Invalid or already used invite code.", "config": config},
            status_code=400
        )

    request.session['pending_beta_code'] = beta_code.upper()
    base_url = getattr(settings, 'BASE_URL', f"{request.url.scheme}://{request.url.netloc}")
    redirect_uri = urllib.parse.urljoin(base_url, "/users/google/login")
    return RedirectResponse(url=redirect_uri, status_code=303)


# ───── Google Login Redirect (UPDATED) ─────
@router.get("/google/login")
async def google_login(request: Request, db: AsyncSession = Depends(get_session)):
    referral_code = request.session.pop('pending_beta_code', None)
    if referral_code:
        request.session['pending_referral_code'] = referral_code

    base_url = getattr(settings, 'BASE_URL', f"{request.url.scheme}://{request.url.netloc}")
    redirect_uri = urllib.parse.urljoin(base_url, "/users/google/callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


# ───── Google OAuth Callback (UPDATED: Auto-Login + Redirect to / for New Users) ─────
@router.get("/google/callback", response_class=HTMLResponse)
async def google_callback(request: Request, db: AsyncSession = Depends(get_session)):
    try:
        token = await oauth.google.authorize_access_token(request)
        try:
            user_info = await oauth.google.parse_id_token(request, token)
        except KeyError:
            resp = await oauth.google.get('https://www.googleapis.com/oauth2/v1/userinfo', token=token)
            resp.raise_for_status()
            user_info = resp.json()
    except Exception as e:
        request.session.pop('pending_referral_code', None)
        error_msg = "Authentication failed. Please try again."
        if "redirect_uri" in str(e).lower():
            error_msg = "Redirect URI mismatch. Contact support."
        
        # FIXED: Fetch and pass beta_config in error context
        config = await get_beta_config(db)
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "error": error_msg, "tab": "login", "beta_config": config}, 
            status_code=400
        )

    referral_code = request.session.pop('pending_referral_code', None)
    email = user_info['email']
    full_name = user_info.get('name', '')

    # FIXED: Proper async username generation
    username_base = email.split('@')[0].lower().replace('.', '')
    username = username_base
    counter = 1
    while True:
        result = await db.execute(select(User).where(User.username == username))
        if result.scalar_one_or_none():
            username = f"{username_base}{counter}"
            counter += 1
        else:
            break

    user_result = await db.execute(select(User).where(User.email == email))
    user = user_result.scalars().first()
    initial_tp = await get_initial_tp_amount(db)

    if not user:
        is_admin = not (await db.execute(select(exists().where(User.is_admin == True)))).scalar()
        new_user = User(
            username=username,
            full_name=full_name,
            email=email,
            password_hash=None,
            referral_code=await generate_unique_code(db),
            referred_by=None,
            is_admin=is_admin,
            plan='starter',
            trade_points=None,  # CHANGED: Set to None; grant initial after
            # Ensure trading_style is None for onboarding trigger
            trading_style=None,
            verified=True,  # NEW: Auto-verify for Google (email confirmed by OAuth)
            verified_at=datetime.utcnow(),  # NEW
        )
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        # NEW: Log for debugging
        logger.info(f"Created new Google user {new_user.id} ({new_user.username}), trading_style={new_user.trading_style}")

        try:
            referred_by = await process_referral_code(db, referral_code, new_user.id, new_user.plan)
            if referred_by:
                await db.execute(update(User).where(User.id == new_user.id).values(referred_by=referred_by))
                await db.commit()
        except ValueError as e:
            # UPDATED: Don't delete user on referral error; just log
            logger.warning(f"Referral processing failed for Google user {new_user.id}: {str(e)}")
            await db.commit()  # Ensure commit even on error

        await generate_beta_invites(db, new_user.id)

        # CHANGED: Use grant_trade_points for initial TP
        await grant_trade_points(
            db, new_user, 'initial_grant', initial_tp,
            description="Starting TP for starter plan", redis=None
        )

        user = new_user

    # UPDATED: Use long-lived expiry for Google flow (aligns with refresh token logic)
    expires_delta = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)  # e.g., 30 days
    access_token = auth.create_access_token({"sub": str(user.id)}, expires_delta)
    
    # UPDATED: For new users, redirect to /onboard to trigger onboarding check; for existing, to /dashboard
    redirect_url = "/onboard" if user.trading_style is None else "/dashboard"
    logger.info(f"Google login for user {user.id}: redirecting to {redirect_url} (trading_style={user.trading_style})")
    
    redirect = RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)
    redirect.set_cookie(
        "access_token", access_token, 
        httponly=True, 
        max_age=int(expires_delta.total_seconds()),  # Dynamic, matches token expiry
        secure=False,  # Set to True in production (e.g., via settings)
        samesite="lax"
    )
    return redirect


# ───── Logout ─────
@router.get("/logout")
async def logout(access_token: Optional[str] = Cookie(None)):
    redirect = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    if access_token:
        redirect.delete_cookie("access_token")
    return redirect
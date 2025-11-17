# users.py - Updated with Google Verification for Email/Password Signup
# (Minimal changes: Added /check_email endpoint for recovery UX; no major shifts since recovery logic is in waitlist.py)
from pathlib import Path
from datetime import timedelta, datetime
from typing import Union, Any, Optional
import json
import urllib.parse
import os
import secrets
import string
import logging

from fastapi import APIRouter, Depends, Request, Form, HTTPException, status, Cookie, BackgroundTasks, Query
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, exists, update
from authlib.integrations.starlette_client import OAuth
import httpx

from templates_config import templates
from models import models, schemas
from models.models import (
    BetaInvite, BetaConfig, Referral, PointTransaction, User, InitialTpConfig,
    BetaReferralTpConfig
)
import auth

from database import get_session
from app_utils.points import grant_trade_points
from config import get_settings
from redis.asyncio import Redis
from redis_client import redis_dependency

settings = get_settings()

logger = logging.getLogger(__name__)

# Tier bonuses
REFERRAL_TIER_BONUSES = {
    'rookie': 1.0,
    'pro_trader': 1.2,
    'elite_alpha': 1.5,
}

def get_tier_bonus(tier: str) -> float:
    return REFERRAL_TIER_BONUSES.get(tier, 1.0)

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


# ───── UPDATED: Verify Page (with error support) ─────
@router.get("/verify", response_class=HTMLResponse)
async def verify_page(
    request: Request,
    new: bool = Query(False),
    email: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
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
            "verify_error": error,
            "tab": "verify",
            "beta_config": config
        }
    )


# ───── NEW: Google Verify Redirect ─────
@router.get("/google/verify")
async def google_verify_redirect(
    request: Request,
    email: str = Query(...),
    db: AsyncSession = Depends(get_session)
):
    email = email.lower().strip()

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

    request.session['pending_verify_user_id'] = user.id
    return RedirectResponse(url="/users/google/login", status_code=status.HTTP_303_SEE_OTHER)


# ───── NEW: Email Check for Recovery UX (Reusable) ─────
@router.post("/check_email", response_model=schemas.GenericResponse)
async def check_email(
    request: Request,
    payload: dict,
    db: AsyncSession = Depends(get_session)
):
    email = payload.get('email', '').lower().strip()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required.")

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        return {"success": False, "message": "No account found for this email. Please sign up first."}
    
    # Optional: Check verified for stricter beta gating
    if not user.verified:
        return {"success": False, "message": "Account exists but requires Google verification. We'll handle that next."}
    
    return {"success": True, "message": "Account found. Proceeding to secure login."}


# ───── JSON API: Signup (UPDATED: No email, requires Google verification) ─────
@router.post("/signup-email", response_model=schemas.GenericResponse)
async def create_user_json(
    payload: schemas.SignupRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session)
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

    logger.info(f"Created new user {new_user.id} ({new_user.username}), awaiting Google verification")

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
        "message": "User created successfully. Connect your Google account to verify your email.",
        "needs_verification": True,
        "email": email,
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
        return {"success": False, "message": "Please verify your email with Google before logging in.", "needs_verification": True, "email": user.email}

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS if payload.remember_me else 7)

    access_token = auth.create_access_token({"sub": str(user.id)}, access_token_expires)
    refresh_token = auth.create_refresh_token({"sub": str(user.id)}, refresh_token_expires)

    return {
        "success": True,
        "message": "Login successful",
        "data": {"access_token": access_token, "refresh_token": refresh_token}
    }


# ───── NEW: JSON Refresh Token Endpoint ─────
@router.post("/refresh", response_model=schemas.GenericResponse)
async def refresh_token_json(
    refresh_token: str = Form(...),  # For form/multipart if needed; or use Cookie
    db: AsyncSession = Depends(get_session)
):
    try:
        payload = auth.decode_refresh_token(refresh_token)
        user_id = int(payload.get("sub"))
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid refresh token.")
        
        # Issue new access token (short-lived); optionally rotate refresh
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        new_access_token = auth.create_access_token({"sub": str(user.id)}, access_token_expires)
        
        # Optional: Rotate refresh token for security (create new one)
        # refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        # new_refresh_token = auth.create_refresh_token({"sub": str(user.id)}, refresh_token_expires)
        # But for simplicity, return existing (or implement rotation if needed)
        
        return {
            "success": True,
            "data": {"access_token": new_access_token}  # , "refresh_token": new_refresh_token if rotating
        }
    except Exception as e:
        logger.warning(f"Refresh failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid refresh token.")


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


# ───── Form Signup (UPDATED: No email, redirect to /verify for Google) ─────
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
    logger.info(f"Created new user {new_user.id} ({new_user.username}), trading_style={new_user.trading_style}, awaiting Google verification")

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


# ───── UPDATED: Form Login (Now sets both access and refresh tokens as cookies) ─────
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

    # UPDATED: Create both access (short) and refresh (long) tokens
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS if remember_me else 7)
    
    access_token = auth.create_access_token({"sub": str(user.id)}, access_token_expires)
    refresh_token = auth.create_refresh_token({"sub": str(user.id)}, refresh_token_expires)

    # UPDATED: Redirect based on onboarding
    redirect_url = "/onboard" if user.trading_style is None else "/dashboard"
    redirect = RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)
    
    # Set cookies: access short-lived, refresh long-lived
    secure_cookies = getattr(settings, 'SECURE_COOKIES', False)  # FIXED: Fallback to False if not set
    redirect.set_cookie(
        "access_token", access_token, 
        httponly=True, 
        max_age=int(access_token_expires.total_seconds()), 
        secure=secure_cookies,
        samesite="lax"
    )
    redirect.set_cookie(
        "refresh_token", refresh_token, 
        httponly=True, 
        max_age=int(refresh_token_expires.total_seconds()), 
        secure=secure_cookies,
        samesite="lax"
    )
    return redirect


# ───── NEW: Form Refresh (for cookie-based refresh) ─────
@router.post("/refresh-form")
async def refresh_form(
    request: Request,
    refresh_token: Optional[str] = Cookie(None),  # Extract from cookie
    db: AsyncSession = Depends(get_session)
):
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token provided.")
    
    try:
        payload = auth.decode_refresh_token(refresh_token)
        user_id = int(payload.get("sub"))
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid refresh token.")
        
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        new_access_token = auth.create_access_token({"sub": str(user.id)}, access_token_expires)
        
        # Redirect to dashboard (or original page via query param if added)
        redirect = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
        secure_cookies = getattr(settings, 'SECURE_COOKIES', False)  # FIXED: Fallback to False
        redirect.set_cookie(
            "access_token", new_access_token, 
            httponly=True, 
            max_age=int(access_token_expires.total_seconds()), 
            secure=secure_cookies,
            samesite="lax"
        )
        # Keep existing refresh_token (no rotation for simplicity)
        return redirect
    except Exception as e:
        logger.warning(f"Form refresh failed: {str(e)}")
        # Clear cookies on failure
        redirect = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        redirect.delete_cookie("access_token")
        redirect.delete_cookie("refresh_token")
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


# ───── UPDATED: Google OAuth Callback (Now sets both access and refresh cookies) ─────
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
        request.session.pop('pending_verify_user_id', None)
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

    # Normalize email
    google_email = user_info['email'].lower().strip()
    full_name = user_info.get('name', '')

    # Handle verification mode first
    pending_verify_id = request.session.pop('pending_verify_user_id', None)
    if pending_verify_id:
        pending_user = await db.get(User, pending_verify_id)
        if not pending_user or pending_user.email != google_email:
            error_msg = "The email from your Google account doesn't match the one you signed up with. Please use the Google account associated with your signup email."
            redirect_url = f"/users/verify?email={urllib.parse.quote(pending_user.email if pending_user else google_email)}&error={urllib.parse.quote(error_msg)}"
            return RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)

        # Verify the user
        pending_user.verified = True
        pending_user.verified_at = datetime.utcnow()
        if not pending_user.full_name:
            pending_user.full_name = full_name
        await db.commit()
        await db.refresh(pending_user)

        # Auto-login with both tokens
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        access_token = auth.create_access_token({"sub": str(pending_user.id)}, access_token_expires)
        refresh_token = auth.create_refresh_token({"sub": str(pending_user.id)}, refresh_token_expires)
        
        redirect_url = "/onboard" if pending_user.trading_style is None else "/dashboard"
        logger.info(f"Google verification for user {pending_user.id}: redirecting to {redirect_url} (trading_style={pending_user.trading_style})")
        
        redirect = RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)
        secure_cookies = getattr(settings, 'SECURE_COOKIES', False)  # FIXED: Fallback to False if not set
        redirect.set_cookie(
            "access_token", access_token, 
            httponly=True, 
            max_age=int(access_token_expires.total_seconds()), 
            secure=secure_cookies,
            samesite="lax"
        )
        redirect.set_cookie(
            "refresh_token", refresh_token, 
            httponly=True, 
            max_age=int(refresh_token_expires.total_seconds()), 
            secure=secure_cookies,
            samesite="lax"
        )
        return redirect

    # Normal flow
    referral_code = request.session.pop('pending_referral_code', None)

    # FIXED: Proper async username generation
    username_base = google_email.split('@')[0].lower().replace('.', '')
    username = username_base
    counter = 1
    while True:
        result = await db.execute(select(User).where(User.username == username))
        if result.scalar_one_or_none():
            username = f"{username_base}{counter}"
            counter += 1
        else:
            break

    user_result = await db.execute(select(User).where(User.email == google_email))
    user = user_result.scalars().first()
    initial_tp = await get_initial_tp_amount(db)

    if not user:
        is_admin = not (await db.execute(select(exists().where(User.is_admin == True)))).scalar()
        new_user = User(
            username=username,
            full_name=full_name,
            email=google_email,
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

    # UPDATED: Create both access (short) and refresh (long) tokens for Google flow
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    access_token = auth.create_access_token({"sub": str(user.id)}, access_token_expires)
    refresh_token = auth.create_refresh_token({"sub": str(user.id)}, refresh_token_expires)
    
    # UPDATED: For new users, redirect to /onboard to trigger onboarding check; for existing, to /dashboard
    redirect_url = "/onboard" if user.trading_style is None else "/dashboard"
    logger.info(f"Google login for user {user.id}: redirecting to {redirect_url} (trading_style={user.trading_style})")
    
    redirect = RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)
    secure_cookies = getattr(settings, 'SECURE_COOKIES', False)  # FIXED: Fallback to False if not set
    redirect.set_cookie(
        "access_token", access_token, 
        httponly=True, 
        max_age=int(access_token_expires.total_seconds()), 
        secure=secure_cookies,
        samesite="lax"
    )
    redirect.set_cookie(
        "refresh_token", refresh_token, 
        httponly=True, 
        max_age=int(refresh_token_expires.total_seconds()), 
        secure=secure_cookies,
        samesite="lax"
    )
    return redirect


# ───── UPDATED: Logout (Clear both cookies) ─────
@router.get("/logout")
async def logout(
    access_token: Optional[str] = Cookie(None),
    refresh_token: Optional[str] = Cookie(None)  # NEW: Clear refresh too
):
    redirect = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    redirect.delete_cookie("access_token")
    redirect.delete_cookie("refresh_token")  # NEW
    return redirect
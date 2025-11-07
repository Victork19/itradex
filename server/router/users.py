# users.py - Full Updated File
from pathlib import Path
from datetime import timedelta, datetime
from typing import Union, Any, Optional
import json
import urllib.parse
import os
import secrets
import string

from fastapi import APIRouter, Depends, Request, Form, HTTPException, status, Cookie
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, exists, update
from authlib.integrations.starlette_client import OAuth
import httpx

from templates_config import templates
from models import models, schemas
from models.models import (
    BetaInvite, BetaConfig, Referral, PointTransaction, User, InitialTpConfig
)
import auth
from config import settings
from database import get_session

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
async def process_referral_code(db: AsyncSession, referral_code: str, new_user_id: int) -> Optional[int]:
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
            points_awarded = int(config.award_points_on_use * tier_bonus)

            ref = Referral(
                referrer_id=referrer_id,  # 0 for pool
                referee_id=new_user_id,
                status='active',
                points_earned=config.award_points_on_use,
                tier_bonus=tier_bonus
            )
            db.add(ref)

            # Only award points if real user (not pool)
            if referrer_id != 0:
                referrer = await db.get(User, referrer_id)
                if referrer:
                    referrer.trade_points += points_awarded
                    tx = PointTransaction(
                        user_id=referrer_id,
                        type='ref_earn',
                        amount=points_awarded,
                        description=f"Beta invite used by user {new_user_id}",
                        related_ref_id=ref.id
                    )
                    db.add(tx)

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

        referrer.trade_points += points_awarded
        tx = PointTransaction(
            user_id=referrer.id,
            type='ref_earn',
            amount=points_awarded,
            description=f"Referral code used by user {new_user_id}",
            related_ref_id=ref.id
        )
        db.add(tx)

        await db.commit()
        return referrer.id

    # Invalid code
    if config.required_for_signup and config.is_active:
        raise ValueError("Invalid code provided.")
    return None


# ───── JSON API: Signup ─────
@router.post("/signup-email", response_model=schemas.GenericResponse)
async def create_user_json(payload: schemas.SignupRequest, db: AsyncSession = Depends(get_session)):
    email_check = await db.execute(select(User).where(User.email == payload.email))
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
        email=payload.email,
        password_hash=hashed_password,
        referral_code=await generate_unique_code(db),
        referred_by=None,
        is_admin=is_admin,
        plan='starter',
        trade_points=initial_tp,
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    try:
        referred_by = await process_referral_code(db, getattr(payload, 'referral_code', None), new_user.id)
        if referred_by:
            await db.execute(update(User).where(User.id == new_user.id).values(referred_by=referred_by))
            await db.commit()
    except ValueError as e:
        await db.delete(new_user)
        await db.commit()
        return {"success": False, "message": str(e)}

    await generate_beta_invites(db, new_user.id)

    initial_tx = PointTransaction(
        user_id=new_user.id,
        type='initial_grant',
        amount=initial_tp,
        description="Starting TP for starter plan"
    )
    db.add(initial_tx)
    await db.commit()

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    access_token = auth.create_access_token({"sub": str(new_user.id)}, access_token_expires)
    refresh_token = auth.create_refresh_token({"sub": str(new_user.id)}, refresh_token_expires)

    return {
        "success": True,
        "message": "User created successfully",
        "data": {"access_token": access_token, "refresh_token": refresh_token}
    }


# ───── JSON API: Login ─────
@router.post("/login-email", response_model=schemas.GenericResponse)
async def login_json(payload: schemas.LoginRequest, db: AsyncSession = Depends(get_session)):
    result = await db.execute(select(User).where(User.username == payload.username))
    user = result.scalars().first()
    if not user or not auth.verify_password(payload.password, user.password_hash):
        return {"success": False, "message": "Invalid username or password."}

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


# ───── Form Signup ─────
@router.post("/signup", response_class=HTMLResponse)
async def create_user_form(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
    referral_code: str = Form(None),
    db: AsyncSession = Depends(get_session),
):
    if password != password_confirm:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Passwords do not match.", "tab": "signup"}, status_code=400)

    email_check = await db.execute(select(User).where(User.email == email))
    if email_check.scalar_one_or_none():
        return templates.TemplateResponse("index.html", {"request": request, "error": "Email already in use.", "tab": "signup"}, status_code=400)

    username_check = await db.execute(select(User).where(User.username == username))
    if username_check.scalar_one_or_none():
        return templates.TemplateResponse("index.html", {"request": request, "error": "Username already taken.", "tab": "signup"}, status_code=400)

    hashed_password = auth.hash_password(password)
    admin_exists = await db.execute(select(exists().where(User.is_admin == True)))
    is_admin = not admin_exists.scalar()
    initial_tp = await get_initial_tp_amount(db)

    new_user = User(
        username=username.lower(),
        full_name=full_name,
        email=email,
        password_hash=hashed_password,
        referral_code=await generate_unique_code(db),
        referred_by=None,
        is_admin=is_admin,
        plan='starter',
        trade_points=initial_tp,
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    try:
        referred_by = await process_referral_code(db, referral_code, new_user.id)
        if referred_by:
            await db.execute(update(User).where(User.id == new_user.id).values(referred_by=referred_by))
            await db.commit()
    except ValueError as e:
        await db.delete(new_user)
        await db.commit()
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e), "tab": "signup"}, status_code=400)

    await generate_beta_invites(db, new_user.id)

    initial_tx = PointTransaction(
        user_id=new_user.id,
        type='initial_grant',
        amount=initial_tp,
        description="Starting TP for starter plan"
    )
    db.add(initial_tx)
    await db.commit()

    return RedirectResponse(url="/?success=true&tab=login", status_code=status.HTTP_303_SEE_OTHER)


# ───── Form Login ─────
@router.post("/login", response_class=HTMLResponse)
async def login_form(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    remember_me: bool = Form(False),
    db: AsyncSession = Depends(get_session),
):
    user = (await db.execute(select(User).where(User.username == username))).scalars().first()
    if not user or not auth.verify_password(password, user.password_hash):
        return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid username or password.", "tab": "login"}, status_code=401)

    expires = timedelta(days=settings.REMEMBER_ME_REFRESH_TOKEN_EXPIRE_DAYS) if remember_me else timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token({"sub": str(user.id)}, expires)

    redirect = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
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


# ───── Google OAuth Callback (FIXED) ─────
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
            trade_points=initial_tp,
        )
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        try:
            referred_by = await process_referral_code(db, referral_code, new_user.id)
            if referred_by:
                await db.execute(update(User).where(User.id == new_user.id).values(referred_by=referred_by))
                await db.commit()
        except ValueError as e:
            await db.delete(new_user)
            await db.commit()
            # FIXED: Fetch and pass beta_config in error context
            config = await get_beta_config(db)
            return templates.TemplateResponse(
                "index.html", 
                {"request": request, "error": str(e), "tab": "login", "beta_config": config}, 
                status_code=400
            )

        await generate_beta_invites(db, new_user.id)

        initial_tx = PointTransaction(
            user_id=new_user.id,
            type='initial_grant',
            amount=initial_tp,
            description="Starting TP for starter plan"
        )
        db.add(initial_tx)
        await db.commit()

        user = new_user

    access_token = auth.create_access_token({"sub": str(user.id)}, timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    redirect = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    redirect.set_cookie("access_token", access_token, httponly=True, max_age=3600, secure=False, samesite="lax")
    return redirect


# ───── Logout ─────
@router.get("/logout")
async def logout(access_token: Optional[str] = Cookie(None)):
    redirect = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    if access_token:
        redirect.delete_cookie("access_token")
    return redirect
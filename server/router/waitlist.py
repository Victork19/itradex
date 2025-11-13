# /app/router/waitlist.py
import logging
import json
import string
import random
from secrets import token_hex
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import re
import bcrypt  # New: For hashing tokens
import asyncio  # New: For async email wrapping if needed

from fastapi import APIRouter, Request, Depends, HTTPException, status, BackgroundTasks, Body, Query, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, EmailStr, validator, Field  # Updated: Added Field if needed
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, asc, desc, or_, text, exists, and_  # Added exists explicitly
from sqlalchemy.orm import selectinload, aliased  # Added aliased
from sqlalchemy.exc import IntegrityError  # New: For constraint handling

from database import get_session
from models.models import Waitlist
from templates_config import templates
from redis.asyncio import Redis
from redis_client import redis_dependency, get_cache, set_cache
from services.email_service import send_email  # New: Centralized email service

logger = logging.getLogger("iTrade")
router = APIRouter()

# New: Pydantic models for validation
class UsernameCheckData(BaseModel):
    username: str = Field(..., min_length=1, max_length=15)

    @validator('username')
    def validate_username(cls, v):
        v = v.strip().lstrip("@")
        if not re.match(r'^[a-zA-Z0-9_]{1,15}$', v):
            raise ValueError('Username must be 1-15 alphanumeric chars or underscores')
        return v.lower()

class SignupData(BaseModel):
    username: str  # Twitter handle
    email: EmailStr
    wallet: Optional[str] = None
    referred_by: Optional[str] = None
    retweet: Optional[str] = None  # Unused

    @validator('username')
    def validate_username(cls, v):
        v = v.strip().lstrip("@")
        if not re.match(r'^[a-zA-Z0-9_]{1,15}$', v):
            raise ValueError('Username must be 1-15 alphanumeric chars or underscores')
        return v.lower()

    @validator('wallet')
    def validate_wallet(cls, v):
        if v:
            # Normalize to lowercase for case-insensitivity and validate format
            v = v.strip().lower()
            if not (v.startswith('0x') and len(v) == 42 and re.match(r'^0x[a-f0-9]{40}$', v)):
                raise ValueError('Invalid wallet format (0x + 40 hex chars)')
            return v
        return v

    @validator('referred_by')
    def validate_referred_by(cls, v):
        return v.strip().upper() if v else None

class VerifyData(BaseModel):
    email: EmailStr
    code: str

class ResendData(BaseModel):
    email: EmailStr

class RecoverData(BaseModel):
    email: EmailStr

def generate_referral_code(length: int = 8) -> str:
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def generate_verification_code() -> str:
    return ''.join(random.choices(string.digits, k=6))

def hash_token(token: str) -> str:
    # New: Hash for secure storage
    return bcrypt.hashpw(token.encode(), bcrypt.gensalt()).decode()

def verify_token_hash(token: str, hashed: str) -> bool:
    if not hashed:
        return False
    # Handle legacy plain tokens
    if hashed.startswith('$2'):
        try:
            return bcrypt.checkpw(token.encode(), hashed.encode())
        except ValueError:
            logger.warning(f"Invalid bcrypt hash encountered for token verification: {hashed[:20]}...")
            return False
    else:
        # Legacy: plain text comparison
        return hashed == token

def get_remote_address(request: Request) -> str:
    # Helper to get client IP (implement based on your setup, e.g., proxy headers)
    return request.client.host if request.client else "unknown"

@router.get("/", response_class=HTMLResponse)
async def waitlist_landing(request: Request):
    return templates.TemplateResponse("waitlist_signup.html", {"request": request})

@router.post("/", response_model=Dict[str, Any])
async def waitlist_signup(
    request: Request,
    background_tasks: BackgroundTasks,  # New: For async email
    data: SignupData = Body(..., description="Signup data"),
    db: AsyncSession = Depends(get_session),
    redis: Redis = Depends(redis_dependency),
):
    # Rate limit signups per IP
    ip = get_remote_address(request)
    ip_key = f"signup_ip:{ip}"
    count = await redis.incr(ip_key)
    if count == 1:
        await redis.expire(ip_key, 3600)  # 1hr window
    if count > 5:
        raise HTTPException(429, "Too many signups from this IP. Slow down.")

    # No need for manual SignupData instantiation—it's already validated via Body
    username = data.username  # Already validated/lowercased
    email = data.email.lower()
    wallet = data.wallet  # Already normalized
    referred_by = data.referred_by
    retweet = data.retweet  # Still unused, but available if needed later

    # Enhanced uniqueness check with case-insensitive wallet
    where_clause = or_(
        func.lower(Waitlist.twitter) == username,
        func.lower(Waitlist.email) == email
    )
    if wallet:
        where_clause = or_(where_clause, func.lower(Waitlist.wallet) == wallet)

    # FIXED: Proper EXISTS subquery to avoid SQL syntax issues in asyncpg/Postgres
    inner_query = select(1).where(where_clause)
    exists_query = select(exists(inner_query))
    result = await db.execute(exists_query)
    if result.scalar():
        # UPDATED: Field-specific checks for better debug logs and user messages
        # FIXED: Same proper EXISTS construction
        u_inner = select(1).where(func.lower(Waitlist.twitter) == username)
        u_exists = select(exists(u_inner))
        username_conflict = await db.execute(u_exists)
        
        e_inner = select(1).where(func.lower(Waitlist.email) == email)
        e_exists = select(exists(e_inner))
        email_conflict = await db.execute(e_exists)
        
        w_conflict = False
        if wallet:
            w_inner = select(1).where(func.lower(Waitlist.wallet) == wallet)
            w_exists = select(exists(w_inner))
            w_conflict_result = await db.execute(w_exists)
            w_conflict = w_conflict_result.scalar()
        
        conflict_details = []
        if username_conflict.scalar():
            conflict_details.append("username")
        if email_conflict.scalar():
            conflict_details.append("email")
        if w_conflict:
            conflict_details.append("wallet")
        
        logger.warning(f"Signup conflict for {email}: {', '.join(conflict_details)} | username={username}, email={email}, wallet={wallet} | IP: {ip}")
        raise HTTPException(409, f"One or more of these details ({', '.join(conflict_details)}) are already in use by someone else. Try recovery if it's your account, or use different details.")

    # New: Validate referred_by
    if referred_by:
        ref_result = await db.execute(
            select(Waitlist).where(
                func.upper(Waitlist.referral_code) == referred_by,
                Waitlist.verified == True  # Only verified referrers count
            )
        )
        if not ref_result.scalar_one_or_none():
            raise HTTPException(400, "Invalid referral code.")

    # Unique referral code (with timeout loop)
    max_attempts = 10
    for _ in range(max_attempts):
        code = generate_referral_code(8)
        exists_code = await db.execute(select(Waitlist).where(Waitlist.referral_code == code))
        if not exists_code.scalar_one_or_none():
            break
    else:
        raise HTTPException(500, "Unable to generate unique code.")

    verification_code = generate_verification_code()

    # Create entry
    entry = Waitlist(
        twitter=username,
        email=email,
        wallet=wallet,
        referral_code=code,
        referred_by=referred_by,
        verified=False,
    )
    db.add(entry)
    await db.flush()  # Get ID

    # Store verification in Redis (15 min TTL)
    await redis.setex(f"verify:{entry.id}", 900, verification_code)

    # Generate & hash login token (12-char hex, 30-day expiry)
    login_token = token_hex(6).upper()
    hashed_token = hash_token(login_token)
    entry.login_token = hashed_token
    entry.token_expires = datetime.utcnow() + timedelta(days=30)

    try:
        await db.commit()
        await db.refresh(entry)
    except IntegrityError as e:
        await db.rollback()
        logger.error(f"DB constraint violation on signup: {e}")
        raise HTTPException(409, "Account already exists. Try recovery.")

    logger.info(f"New signup: {email} | code: {code} | token: {login_token} | verify: {verification_code} | id: {entry.id} | ip: {ip}")

    # UPDATED: Await the async send_email call to get result and handle errors properly
    body = f"""
Hello,

Your verification code for iTradeX waitlist: {verification_code}

Enter this code to verify your email and complete signup.
This code expires in 15 minutes.
"""
    email_result = await send_email(email, "Verify Your iTradeX Waitlist Email", body, background_tasks)
    if email_result["status"] == "error":
        logger.error(f"Verification email failed: {email_result['message']}")

    return {
        "success": True,
        "referral_code": entry.referral_code,
        "login_token": login_token,  # Plain for user
        "waitlist_id": entry.id
    }

@router.post("/check_username")
async def check_username(
    data: UsernameCheckData,  # Use the new model
    db: AsyncSession = Depends(get_session),
):
    # FIXED: Proper EXISTS subquery
    inner_query = select(1).where(func.lower(Waitlist.twitter) == data.username)
    exists_query = select(exists(inner_query))
    result = await db.execute(exists_query)
    return {"available": not result.scalar()}

@router.post("/check_referral")
async def check_referral(
    data: Dict[str, Any] = Body(...),  # Keep simple for now
    db: AsyncSession = Depends(get_session),
):
    code = data.get("code", "").strip().upper()
    if not code:
        return {"valid": False}

    # FIXED: Proper where for referral check (and verified)
    inner_query = select(1).where(
        func.upper(Waitlist.referral_code) == code,
        Waitlist.verified == True
    )
    exists_query = select(exists(inner_query))
    result = await db.execute(exists_query)
    return {"valid": result.scalar()}

@router.post("/resend_code")
async def resend_code(
    request: Request,
    background_tasks: BackgroundTasks,
    data: ResendData,
    db: AsyncSession = Depends(get_session),
    redis: Redis = Depends(redis_dependency),
):
    email = data.email.lower()

    # New: Rate limit per email
    email_key = f"resend:{email}"
    count = await redis.incr(email_key)
    if count == 1:
        await redis.expire(email_key, 3600)
    if count > 3:
        raise HTTPException(429, "Too many resend requests. Try again later.")

    entry_result = await db.execute(
        select(Waitlist).where(func.lower(Waitlist.email) == email, Waitlist.verified == False)
    )
    entry = entry_result.scalar_one_or_none()
    if not entry:
        raise HTTPException(404, "Unverified entry not found.")

    new_code = generate_verification_code()
    await redis.setex(f"verify:{entry.id}", 900, new_code)

    body = f"""
Hello,

Your new verification code for iTradeX waitlist: {new_code}

Enter this code to verify your email.
This code expires in 15 minutes.
"""
    # UPDATED: Await the async send_email call to get result and handle errors properly
    email_result = await send_email(email, "New Verification Code for iTrade Waitlist", body, background_tasks)
    if email_result["status"] == "error":
        logger.error(f"Resend email failed: {email_result['message']}")

    logger.info(f"Resent code to {email}: {new_code} | id: {entry.id}")
    return {"success": True}

@router.post("/verify_code")
async def verify_code(
    data: VerifyData,
    db: AsyncSession = Depends(get_session),
    redis: Redis = Depends(redis_dependency),
):
    email = data.email.lower()
    code = data.code.strip()

    entry_result = await db.execute(
        select(Waitlist).where(
            func.lower(Waitlist.email) == email,
            Waitlist.verified == False
        )
    )
    entry = entry_result.scalar_one_or_none()
    if not entry:
        raise HTTPException(400, "Entry not found or already verified.")

    stored_code = await redis.get(f"verify:{entry.id}")
    
    # FIXED: Safe decode—handle str (decode_responses=True) or bytes
    stored_str = stored_code.decode('utf-8') if isinstance(stored_code, bytes) else stored_code
    if not stored_str or stored_str != code:
        raise HTTPException(400, "Invalid or expired code.")

    await redis.delete(f"verify:{entry.id}")

    # NEW: Invalidate referrer's cache if this is a referral
    if entry.referred_by:
        ref_result = await db.execute(
            select(Waitlist.id).where(
                func.upper(Waitlist.referral_code) == entry.referred_by
            )
        )
        ref_id = ref_result.scalar_one_or_none()
        if ref_id:
            await redis.delete(f"waitlist_dash:{ref_id}")
            logger.info(f"Invalidated referrer cache on verification: {ref_id} for verified {entry.id}")

    # Existing: Invalidate own caches
    await redis.delete(f"waitlist_dash:{entry.id}")
    await redis.delete("waitlist_global_stats")

    entry.verified = True
    entry.verified_at = datetime.utcnow()

    await db.commit()
    logger.info(f"Verified: {email} | id: {entry.id}")
    return {"success": True}

# New: Token recovery endpoint
@router.post("/recover_token")
async def recover_token(
    request: Request,
    background_tasks: BackgroundTasks,
    data: RecoverData,
    db: AsyncSession = Depends(get_session),
    redis: Redis = Depends(redis_dependency),
):
    email = data.email.lower()
    entry_result = await db.execute(
        select(Waitlist).where(
            func.lower(Waitlist.email) == email,
            Waitlist.verified == True  # Only verified
        )
    )
    entry = entry_result.scalar_one_or_none()
    if not entry:
        raise HTTPException(404, "Verified entry not found.")

    # Generate new token
    new_token = token_hex(6).upper()
    hashed = hash_token(new_token)
    old_expires = entry.token_expires
    entry.login_token = hashed
    entry.token_expires = datetime.utcnow() + timedelta(days=30)
    await db.commit()

    # New: Per-email rate limit
    email_key = f"recover:{email}"
    count = await redis.incr(email_key)
    if count == 1:
        await redis.expire(email_key, 3600)
    if count > 3:
        raise HTTPException(429, "Too many recovery requests.")

    body = f"""
Hello,

Your new iTradeX waitlist dashboard token: {new_token}

Access your dashboard at: /waitlist/dashboard/{new_token}

This token expires on {entry.token_expires.date()}. If it expires, request a new one.

If you didn't request this, ignore this email.
"""
    # UPDATED: Await the async send_email call to get result and handle errors properly
    email_result = await send_email(email, "Your New iTradeX Dashboard Token", body, background_tasks)
    if email_result["status"] == "error":
        logger.error(f"Recovery email failed: {email_result['message']}")

    logger.info(f"Recovered token for {email} | new: {new_token} | old_expires: {old_expires} | id: {entry.id}")
    return {"success": True, "message": "New token sent to email."}

@router.get("/recover", response_class=HTMLResponse)
async def recover_token_form(request: Request, expired: bool = Query(False)):
    """NEW: Simple form page for token recovery."""
    context = {
        "request": request,
        "expired": expired,  # For showing "Your token has expired" message
        "email_hint": "",  # Could pre-fill from query param if passed
    }
    return templates.TemplateResponse("waitlist_recover.html", context)

@router.get("/success", response_class=HTMLResponse)
async def waitlist_success(
    request: Request,
    code: str = Query(...),
    twitter: str = Query(...),
    token: str = Query(...),
):
    # New: Add login instructions in template context
    return templates.TemplateResponse(
        "waitlist_success.html",
        {
            "request": request,
            "referral_code": code,
            "twitter": twitter,
            "login_token": token,
            "dashboard_url": f"/waitlist/dashboard/{token}",  # For easy link
        },
    )

@router.get("/dashboard/{token}", response_class=HTMLResponse)
async def waitlist_dashboard(
    request: Request,
    token: str,
    db: AsyncSession = Depends(get_session),
    redis: Redis = Depends(redis_dependency),
):
    if len(token) != 12 or not re.match(r'^[A-F0-9]{12}$', token):
        raise HTTPException(400, "Invalid token format")

    # FIXED: Fetch all non-expired entries and check hash in Python (since bcrypt salt prevents direct query)
    # UPDATED: Add verified == True to ensure only verified users access dashboard
    result = await db.execute(
        select(Waitlist)
        .options(selectinload(Waitlist.referrals))
        .where(
            Waitlist.verified == True,
            Waitlist.token_expires > datetime.utcnow(),
        )
    )
    entries = result.scalars().all()
    entry = None
    for e in entries:
        if verify_token_hash(token, e.login_token):
            entry = e
            break

    if not entry:
        # NEW: For expired/invalid, redirect to recovery instead of 401
        # (We can't get email here without full query, so generic redirect; user enters email on form)
        return RedirectResponse(url="/waitlist/recover?expired=true", status_code=302)

    # Existing auto-renew logic (now only runs if fetched successfully, i.e., not expired)
    now = datetime.utcnow()
    # Optional: If close to expiry (e.g., <1 day), still warn but don't renew/redirect
    days_left = (entry.token_expires - now).days
    if days_left <= 0:  # Shouldn't hit here due to where clause, but safety
        return RedirectResponse(url="/waitlist/recover?expired=true", status_code=302)

    # NEW: Force recompute if refresh param present
    refresh_param = request.query_params.get('refresh')
    force_recompute = bool(refresh_param)

    cache_key = f"waitlist_dash:{entry.id}"
    cached = await get_cache(redis, cache_key)
    render_context = None
    if cached and not force_recompute:
        cached_str = cached.decode('utf-8') if isinstance(cached, bytes) else str(cached)
        try:
            context = json.loads(cached_str)
            # Convert back datetime strings for template rendering
            if context.get("access_granted_at"):
                context["access_granted_at"] = datetime.fromisoformat(context["access_granted_at"])
            render_context = context
        except json.JSONDecodeError:
            render_context = None

    if render_context is None:
        # Enhanced position with window function (handles ties)
        pos_query = text("""
            SELECT ROW_NUMBER() OVER (ORDER BY created_at ASC, id ASC) as pos
            FROM waitlist
            WHERE id = :id
        """)
        pos_result = await db.execute(pos_query, {"id": entry.id})
        position = pos_result.scalar_one() or 1

        total = (await db.execute(select(func.count(Waitlist.id)))).scalar_one()
        total_referrals_result = await db.execute(
            select(func.count(Waitlist.id)).where(Waitlist.referred_by.is_not(None))
        )
        total_referrals = total_referrals_result.scalar_one()

        # UPDATED: Use join-based count to match leaderboard logic exactly
        referral_table = aliased(Waitlist)
        referrer_alias = aliased(Waitlist)
        personal_query = (
            select(func.count(referral_table.id))
            .select_from(referrer_alias)
            .outerjoin(
                referral_table,
                and_(
                    referral_table.referred_by == referrer_alias.referral_code,
                    referral_table.verified == True
                )
            )
            .where(referrer_alias.id == entry.id)
        )
        referrals_count = (await db.execute(personal_query)).scalar_one() or 0

        logger.info(f"Computed referrals for {entry.twitter}: {referrals_count} via code {entry.referral_code}")

        # FIXED: Use aliased table for self-referential join to count referrals (via string match on referral_code)
        # Assumes only verified referrals count; filters leaderboard to verified users
        leaderboard_query = (
            select(
                Waitlist.id.label('id'),
                Waitlist.twitter.label('twitter'),
                func.count(referral_table.id).label('ref_count'),
                Waitlist.created_at.label('created_at')
            )
            .select_from(Waitlist)
            .where(Waitlist.verified == True)
            .outerjoin(
                referral_table,
                and_(
                    referral_table.referred_by == Waitlist.referral_code,
                    referral_table.verified == True  # Only count verified referrals
                )
            )
            .group_by(Waitlist.id, Waitlist.twitter, Waitlist.created_at)
            .order_by(
                desc('ref_count'),
                asc('created_at')
            )
            .limit(5)
        )
        top_result = await db.execute(leaderboard_query)
        top_entries = top_result.fetchall()
        leaderboard = [
            {"position": i + 1, "twitter": row.twitter or "Anon", "referrals": row.ref_count}
            for i, row in enumerate(top_entries)
        ]

        # Stats with longer cache (30 min)
        stats_key = "waitlist_global_stats"
        stats_json = await get_cache(redis, stats_key)
        stats_ttl = await redis.ttl(stats_key)
        if stats_json and stats_ttl > 0:  # New: Check TTL
            stats_json_str = stats_json.decode('utf-8') if isinstance(stats_json, bytes) else str(stats_json)
            stats = json.loads(stats_json_str)
        else:
            verified = (await db.execute(select(func.count(Waitlist.id)).where(Waitlist.verified.is_(True)))).scalar_one()
            avg_referrals = round(total_referrals / max(total, 1), 2)
            # NEW: Simple placeholders for template vars (enhance as needed)
            daily_growth = 0  # Could query COUNT(created_at >= now - 1 day)
            stats = {
                "total_entries": total,
                "verified_count": verified,
                "avg_referrals": avg_referrals,
                "daily_growth": daily_growth,
            }
            await set_cache(redis, stats_key, json.dumps(stats), ttl=1800)  # 30 min

        # NEW: Placeholder computations for template (customize based on logic)
        access_progress = min((position / total * 100) if total else 0, 100) if not entry.access_granted_at else 100
        access_eta = "Soon" if access_progress < 100 else "Granted"

        serializable_context = {
            "twitter": entry.twitter,
            "referral_code": entry.referral_code,
            "position": position,
            "total_size": total,
            "referrals_count": referrals_count,
            "total_referrals_tp": total_referrals,  # Renamed for clarity
            "verified": entry.verified,
            "access_granted": entry.access_granted_at is not None,
            "access_granted_at": entry.access_granted_at.isoformat() if entry.access_granted_at else None,
            "access_code": entry.access_code if hasattr(entry, 'access_code') else None,  # Assuming model has this
            "leaderboard": leaderboard,
            "stats": stats,
            "token": token,
            "days_left": days_left,
            "show_expiry_warning": days_left <= 3,
            "access_progress": access_progress,
            "access_eta": access_eta,
        }

        await set_cache(redis, cache_key, json.dumps(serializable_context), ttl=300)  # Reduced to 5 min for snappier updates

        # Prepare render context with datetime object
        render_context = serializable_context.copy()
        if render_context["access_granted_at"]:
            render_context["access_granted_at"] = datetime.fromisoformat(render_context["access_granted_at"])

    # Add request to render context
    render_context["request"] = request

    return templates.TemplateResponse("waitlist_dashboard.html", render_context)
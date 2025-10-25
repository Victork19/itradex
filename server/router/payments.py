import httpx
import hmac
import hashlib
import json
import logging
from typing import Optional
from datetime import datetime, timedelta, date
import time
import asyncio
import base64

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from config import settings
from auth import get_current_user
from database import get_session
from models.models import User, Subscription, Payment, Pricing, Discount

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/payments", tags=["Payments"])

NOWPAYMENTS_BASE_URL = getattr(settings, "NOWPAYMENTS_BASE_URL", "https://api.nowpayments.io/v1")
API_KEY_HEADERS = {
    "x-api-key": settings.NOWPAYMENTS_API_KEY,
    "Content-Type": "application/json",
}

# Token cache (in-memory; use Redis for production)
_token_cache: Optional[dict] = None
_token_expiry: Optional[float] = None

async def get_nowpayments_token() -> str:
    """Obtain JWT token for NowPayments Recurring Payments API with caching and retries."""
    global _token_cache, _token_expiry
    current_time = time.time()

    # Refresh if expired or near expiry (token lasts 5 min / 300s)
    if _token_cache and _token_expiry and current_time < _token_expiry - 30:  # Buffer: 30s early
        logger.debug("Using cached JWT token")
        return _token_cache["token"]

    if not hasattr(settings, 'NOWPAYMENTS_EMAIL') or not hasattr(settings, 'NOWPAYMENTS_PASSWORD'):
        raise HTTPException(status_code=500, detail="NowPayments email/password not configured for subscriptions")

    # Retry logic: up to 3 attempts with exponential backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            timeout = httpx.Timeout(60.0, connect=30.0)  # Total 60s, connect 30s
            async with httpx.AsyncClient(timeout=timeout) as client:
                payload = {
                    "email": settings.NOWPAYMENTS_EMAIL,
                    "password": settings.NOWPAYMENTS_PASSWORD
                }
                resp = await client.post(
                    f"{NOWPAYMENTS_BASE_URL}/auth",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                if resp.status_code != 200:
                    logger.error(f"Failed to obtain JWT token (attempt {attempt+1}): {resp.status_code} - {resp.text}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                        continue
                    raise HTTPException(status_code=500, detail=f"Failed to authenticate with NowPayments: {resp.text}")
                
                token_data = resp.json()
                logger.info(f"Auth response: {token_data}")  # Log full response to verify token structure
                token = token_data.get("token") or token_data.get("result", {}).get("token")
                if not token:
                    raise HTTPException(status_code=500, detail=f"No token in auth response. Response: {token_data}")
                
                # Dynamically parse JWT expiry if possible
                try:
                    payload_b64 = token.split('.')[1]
                    payload_b64 += '=' * (4 - len(payload_b64) % 4)
                    payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode('utf-8'))
                    token_exp_unix = payload.get('exp')
                    if token_exp_unix:
                        _token_expiry = token_exp_unix  # Use actual expiry timestamp
                        logger.info(f"Token expires at Unix {token_exp_unix} ({datetime.fromtimestamp(token_exp_unix)})")
                    else:
                        _token_expiry = current_time + 240  # Fallback
                except Exception as e:
                    logger.warning(f"Failed to parse JWT exp: {e}, using fallback 4 min")
                    _token_expiry = current_time + 240
                
                _token_cache = {"token": token}
                logger.info(f"Successfully obtained new NowPayments JWT token (expires in ~5 min)")
                return token

        except httpx.ConnectTimeout:
            logger.error(f"Connect timeout to NowPayments API (attempt {attempt+1})")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise HTTPException(status_code=503, detail="Payment service temporarily unavailable - please try again in a moment.")
        except httpx.TimeoutException:
            logger.error(f"Request timeout to NowPayments API (attempt {attempt+1})")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise HTTPException(status_code=503, detail="Payment service temporarily unavailable - please try again in a moment.")
        except Exception as e:
            logger.error(f"Unexpected error obtaining token (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise HTTPException(status_code=500, detail=f"Failed to connect to payment service: {str(e)}")

    raise HTTPException(status_code=500, detail="Max retries exceeded for NowPayments authentication")

def get_subscription_headers(token: str):
    """Get headers for subscription API write requests."""
    headers = {
        "Authorization": f"Bearer {token}",
        "x-api-key": settings.NOWPAYMENTS_API_KEY,
        "Content-Type": "application/json",
    }
    logger.debug(f"Subscription headers prepared (Auth starts with: Bearer {token[:10]}..., API key prefix: {settings.NOWPAYMENTS_API_KEY[:10]}...)")
    return headers

async def get_plan_amount(plan: str, interval: str, db: AsyncSession) -> float:
    """Fetch plan amount from database."""
    result = await db.execute(
        select(Pricing.amount).where(
            Pricing.plan == plan,
            Pricing.interval == interval
        )
    )
    amount = result.scalar()
    if amount is None:
        raise HTTPException(status_code=500, detail=f"Price not configured for {plan} {interval}")
    return amount

async def get_or_create_plan(title: str, amount: float, interval_days: int) -> str:
    """Get or create a subscription plan in NowPayments."""
    # Append amount to title to handle price changes (e.g., "Pro Monthly - $9.99")
    # This ensures unique plans when prices update, without reusing old ones.
    unique_title = f"{title} - ${amount:.2f}"
    
    async with httpx.AsyncClient() as client:
        logger.debug(f"Listing plans with API key (prefix: {settings.NOWPAYMENTS_API_KEY[:10]}...)")
        resp = await client.get(
            f"{NOWPAYMENTS_BASE_URL}/subscriptions/plans",
            params={"limit": 100},
            headers=API_KEY_HEADERS
        )
        if resp.status_code != 200:
            logger.error(f"List plans failed with API key: {resp.status_code} - {resp.text}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list plans: Invalid API key or permissions."
            )

        plans = resp.json().get("result", [])
        for plan in plans:
            if plan.get("title") == unique_title:
                logger.info(f"Found existing plan: {unique_title} (ID: {plan['id']})")
                return plan["id"]

        token = await get_nowpayments_token()
        headers = get_subscription_headers(token)
        payload = {
            "title": unique_title,  # Use unique_title here
            "interval_day": interval_days,
            "amount": amount,
            "currency": "usd",
            "ipn_callback_url": f"{settings.BASE_URL}/payments/webhook",
            "success_url": f"{settings.BASE_URL}/dashboard?payment=success",
            "cancel_url": f"{settings.BASE_URL}/plans",
        }
        logger.info(f"Creating new plan: {payload}")
        resp = await client.post(
            f"{NOWPAYMENTS_BASE_URL}/subscriptions/plans",
            json=payload,
            headers=headers
        )
        if resp.status_code not in [200, 201]:
            logger.error(f"Create plan failed: {resp.status_code} - {resp.text}")
            raise HTTPException(status_code=500, detail=f"Failed to create plan: {resp.text}")
        
        plan_data = resp.json()
        plan_id = plan_data.get("result", {}).get("id")
        if not plan_id:
            raise HTTPException(status_code=500, detail=f"No plan ID in create response: {plan_data}")
        logger.info(f"Created new plan: {unique_title} (ID: {plan_id})")
        return plan_id

async def is_email_subscribed_to_plan(email: str, plan_id: str, token: str) -> bool:
    """Check if the email is already subscribed to the given plan."""
    headers = get_subscription_headers(token)
    async with httpx.AsyncClient() as client:
        params = {
            "email": email,
            "status": "PAID",
            "subscription_plan_id": plan_id,
            "limit": 1
        }
        logger.debug(f"Checking PAID subscriptions for email: {email}, plan: {plan_id}")
        resp = await client.get(
            f"{NOWPAYMENTS_BASE_URL}/subscriptions",
            params=params,
            headers=headers
        )
        if resp.status_code != 200:
            logger.error(f"Failed to list PAID subscriptions: {resp.status_code} - {resp.text}")
            raise HTTPException(status_code=500, detail=f"Failed to check subscriptions: {resp.text}")
        
        data = resp.json().get("result", [])
        is_subscribed = len(data) > 0
        if is_subscribed:
            logger.info(f"Email {email} already has PAID subscription to plan {plan_id}")
        return is_subscribed

async def cleanup_orphan_subscriptions(email: str, plan_id: str, token: str):
    """Delete any non-paid subscriptions for the email + plan."""
    headers = get_subscription_headers(token)
    async with httpx.AsyncClient() as client:
        params = {"email": email, "limit": 10}
        logger.debug(f"Fetching subscriptions for cleanup: email {email}, plan {plan_id}")
        resp = await client.get(
            f"{NOWPAYMENTS_BASE_URL}/subscriptions",
            params=params,
            headers=headers
        )
        if resp.status_code != 200:
            logger.warning(f"Failed to list subscriptions for cleanup: {resp.status_code} - {resp.text}")
            return

        data = resp.json().get("result", [])
        deleted_count = 0
        for sub in data:
            sub_plan_id = sub.get("subscription_plan_id") or sub.get("plan_id")
            if sub_plan_id == plan_id:
                sub_id = sub.get("id")
                if not sub_id:
                    continue
                del_resp = await client.delete(
                    f"{NOWPAYMENTS_BASE_URL}/subscriptions/{sub_id}",
                    headers=headers
                )
                if del_resp.status_code in [200, 204]:
                    logger.info(f"Deleted orphan subscription {sub_id} for email {email}")
                    deleted_count += 1
                else:
                    logger.warning(f"Failed to delete orphan {sub_id}: {del_resp.status_code} - {del_resp.text}")

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} orphan subscription(s) for {email}")

@router.post("/checkout")
async def create_subscription(
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    body = await request.json()
    plan = body.get("plan")
    interval = body.get("interval", "monthly")

    if not plan:
        raise HTTPException(status_code=400, detail="plan required")

    if not current_user.email:
        raise HTTPException(status_code=400, detail="Email required for subscriptions")

    amount = await get_plan_amount(plan, interval, db)

    # Fetch discount
    result_discount = await db.execute(
        select(Discount).where(Discount.id == 1)
    )
    db_discount = result_discount.scalar_one_or_none()
    perc = 0.0
    if db_discount and db_discount.enabled and (not db_discount.expiry or db_discount.expiry > date.today()):
        perc = db_discount.percentage

    # Get monthly for calculation
    monthly_amount = await get_plan_amount(plan, 'monthly', db)

    # Apply discount (first-month equivalent reduction)
    if perc > 0:
        if interval == 'monthly':
            amount *= (1 - perc / 100)
        else:
            discount_amount = monthly_amount * (perc / 100)
            amount -= discount_amount
        amount = max(amount, 0.01)  # Avoid zero/negative

    interval_days = 30 if interval == "monthly" else 365
    plan_title = f"{plan.title()} {interval.title()}"
    if perc > 0:
        plan_title += f" ({int(perc)}% promo)"  # Make plan unique for discounted
    plan_type = f"{plan}_{interval}"
    plan_id = await get_or_create_plan(plan_title, amount, interval_days)

    result = await db.execute(
        select(Subscription).where(
            Subscription.user_id == current_user.id,
            Subscription.plan_type == plan_type,
            Subscription.status == 'active'
        )
    )
    existing_active = result.scalars().first()
    if existing_active:
        logger.info(f"Existing active subscription found for user {current_user.id} and plan {plan_type}")
        raise HTTPException(
            status_code=400,
            detail="You are already subscribed to this plan."
        )

    result = await db.execute(
        select(Subscription).where(
            Subscription.user_id == current_user.id,
            Subscription.plan_type == plan_type,
            Subscription.status.in_(['pending', 'paused'])
        )
    )
    pending_paused_subs = result.scalars().all()
    token = await get_nowpayments_token()
    headers = get_subscription_headers(token)
    for sub in pending_paused_subs:
        if sub.nowpayments_sub_id:
            async with httpx.AsyncClient() as client:
                del_resp = await client.delete(
                    f"{NOWPAYMENTS_BASE_URL}/subscriptions/{sub.nowpayments_sub_id}",
                    headers=headers
                )
                if del_resp.status_code in [200, 204]:
                    logger.info(f"Deleted pending/paused NowPayments sub {sub.nowpayments_sub_id}")
                else:
                    logger.warning(f"Failed to delete pending/paused sub: {del_resp.status_code}")
        await db.delete(sub)
    if pending_paused_subs:
        await db.commit()
        logger.info(f"Cleaned up {len(pending_paused_subs)} pending/paused subscription(s)")

    if await is_email_subscribed_to_plan(current_user.email, plan_id, token):
        raise HTTPException(
            status_code=400,
            detail="This email is already subscribed to the selected plan."
        )

    await cleanup_orphan_subscriptions(current_user.email, plan_id, token)

    order_id = f"{current_user.id}_{plan_type}"
    order_description = f"iTrade {plan_title} subscription"
    payload = {
        "subscription_plan_id": plan_id,
        "email": current_user.email,
    }
    async with httpx.AsyncClient() as client:
        logger.info(f"Creating subscription with payload: {payload}")
        resp = await client.post(
            f"{NOWPAYMENTS_BASE_URL}/subscriptions",
            json=payload,
            headers=headers
        )
        if resp.status_code not in [200, 201]:
            error_text = resp.text
            logger.error(f"Create subscription failed: {resp.status_code} - {error_text}")
            raise HTTPException(status_code=500, detail=f"Failed to create subscription: {error_text}")

        response_data = resp.json()
        result = response_data.get("result")
        if isinstance(result, list):
            if not result:
                raise HTTPException(status_code=500, detail="Empty result list in response")
            sub_data = result[0]
        else:
            sub_data = result or {}
        np_sub_id = sub_data.get("id")
        if not np_sub_id:
            raise HTTPException(status_code=500, detail=f"No subscription ID in response: {response_data}")

        db_sub = Subscription(
            user_id=current_user.id,
            nowpayments_plan_id=plan_id,
            nowpayments_sub_id=np_sub_id,
            plan_type=plan_type,
            interval_days=interval_days,
            amount_usd=amount,
            status='pending',
            start_date=datetime.utcnow(),
            next_billing_date=datetime.utcnow() + timedelta(days=interval_days),
            order_id=order_id,
            order_description=order_description,
        )
        db.add(db_sub)
        await db.commit()

    return {
        "message": f"Subscription created! Check your email ({current_user.email}) for the payment link from NowPayments.",
        "subscription_id": np_sub_id,
        "email": current_user.email
    }

@router.post("/webhook")
async def nowpayments_webhook(request: Request, db: AsyncSession = Depends(get_session)):
    body = await request.body()
    sig = request.headers.get("x-nowpayments-sig")
    if not verify_ipn(body, sig, settings.NOWPAYMENTS_IPN_SECRET):
        logger.warning("Invalid IPN signature received")
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        logger.error("Invalid JSON in webhook body")
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    payment_status = data.get("payment_status")
    payment_id = data.get("payment_id")
    subscription_id = data.get("subscription_id")

    if not payment_id or not subscription_id:
        logger.info(f"Webhook ignored: missing payment_id or subscription_id")
        return {"status": "ignored"}

    if payment_status in ["finished", "failed", "refunded"]:
        result = await db.execute(
            select(Subscription).where(
                Subscription.nowpayments_sub_id == subscription_id
            )
        )
        db_sub = result.scalars().first()
        if not db_sub:
            logger.info(f"Webhook ignored: subscription {subscription_id} not found")
            return {"status": "ignored"}

        user_id = db_sub.user_id
        plan_type = db_sub.plan_type

        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalars().first()
        if not user:
            logger.info(f"Webhook ignored: user {user_id} not found")
            return {"status": "ignored"}

        result = await db.execute(
            select(Payment).where(Payment.nowpayments_payment_id == payment_id)
        )
        existing_payment = result.scalars().first()

        pay_date = None
        if data.get("pay_date"):
            pay_date = datetime.fromisoformat(data["pay_date"].replace("Z", "+00:00"))

        if existing_payment:
            existing_payment.status = payment_status
            existing_payment.paid_at = pay_date
            existing_payment.updated_at = datetime.utcnow()
        else:
            db_payment = Payment(
                user_id=user_id,
                subscription_id=db_sub.id,
                nowpayments_payment_id=payment_id,
                amount_usd=float(data.get("price_amount", 0)),
                amount_paid_crypto=float(data.get("gross_amount", 0)),
                crypto_currency=data.get("currency"),
                status=payment_status,
                invoice_url=data.get("invoice_url"),
                paid_at=pay_date
            )
            db.add(db_payment)

        if payment_status == "finished":
            db_sub.status = "active"
            if db_sub.next_billing_date is None or db_sub.start_date == db_sub.next_billing_date:
                db_sub.next_billing_date = datetime.utcnow() + timedelta(days=db_sub.interval_days)
            else:
                db_sub.next_billing_date += timedelta(days=db_sub.interval_days)
            user.plan = plan_type
        elif payment_status in ["failed", "refunded"]:
            db_sub.status = "paused"
            user.plan = "starter"

        db_sub.updated_at = datetime.utcnow()
        user.updated_at = datetime.utcnow()
        await db.commit()

        logger.info(f"Webhook processed: payment {payment_id} status {payment_status} for subscription {subscription_id}")

    return {"status": "ok"}

def verify_ipn(body: bytes, sig: str, secret: str) -> bool:
    """Verify IPN signature using HMAC SHA512."""
    expected = hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha512
    ).hexdigest()
    return hmac.compare_digest(sig.lower(), expected.lower())

async def get_current_subscription(db: AsyncSession, user_id: int):
    """Fetch the current active subscription for the user."""
    result = await db.execute(
        select(Subscription).where(
            Subscription.user_id == user_id,
            Subscription.status == 'active'
        ).order_by(Subscription.start_date.desc())
    )
    return result.scalars().first()
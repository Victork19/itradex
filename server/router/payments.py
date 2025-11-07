import hmac
import hashlib
import json
import logging
import httpx  # ADDED: Import for HTTP client
from typing import Optional
from datetime import datetime, timedelta, date
import time
import asyncio
import base64
import re  # For parsing order_id in webhook

from fastapi import APIRouter, Depends, HTTPException, Request, Path, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from config import settings
from auth import get_current_user
from database import get_session
from models.models import User, Subscription, Payment, Pricing, Discount, PointTransaction, UpgradeTpConfig, BetaInvite
import models  # For models.User in marketplace

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
    """Obtain JWT token for NowPayments API with caching and retries."""
    global _token_cache, _token_expiry
    current_time = time.time()

    if _token_cache and _token_expiry and current_time < _token_expiry - 30:
        logger.debug("Using cached JWT token")
        return _token_cache["token"]

    if not hasattr(settings, 'NOWPAYMENTS_EMAIL') or not hasattr(settings, 'NOWPAYMENTS_PASSWORD'):
        raise HTTPException(status_code=500, detail="NowPayments credentials not configured")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            timeout = httpx.Timeout(60.0, connect=30.0)
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
                resp.raise_for_status()
                token_data = resp.json()
                logger.info(f"Auth response: {token_data}")
                token = token_data.get("token") or token_data.get("result", {}).get("token")
                if not token:
                    raise HTTPException(status_code=500, detail=f"No token in auth response. Response: {token_data}")
                
                try:
                    payload_b64 = token.split('.')[1]
                    payload_b64 += '=' * (4 - len(payload_b64) % 4)
                    payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode('utf-8'))
                    token_exp_unix = payload.get('exp')
                    if token_exp_unix:
                        _token_expiry = token_exp_unix
                        logger.info(f"Token expires at Unix {token_exp_unix} ({datetime.fromtimestamp(token_exp_unix)})")
                    else:
                        _token_expiry = current_time + 240
                except Exception as e:
                    logger.warning(f"Failed to parse JWT exp: {e}, using fallback 4 min")
                    _token_expiry = current_time + 240
                
                _token_cache = {"token": token}
                logger.info(f"Successfully obtained new NowPayments JWT token (expires in ~5 min)")
                return token

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error obtaining token (attempt {attempt+1}): {e.response.status_code} - {e.response.text}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise HTTPException(status_code=e.response.status_code, detail=f"Auth failed: {e.response.text}")
        except (httpx.ConnectTimeout, httpx.TimeoutException):
            logger.error(f"Timeout/Connect error to NowPayments API (attempt {attempt+1})")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise HTTPException(status_code=503, detail="Payment service temporarily unavailable — please try again in a moment.")
        except Exception as e:
            logger.error(f"Unexpected error obtaining token (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise HTTPException(status_code=500, detail=f"Failed to connect to payment service: {str(e)}")

    raise HTTPException(status_code=500, detail="Max retries exceeded for NowPayments authentication")

def get_invoice_headers(token: str):
    """Get headers for invoice API requests."""
    headers = {
        "Authorization": f"Bearer {token}",
        "x-api-key": settings.NOWPAYMENTS_API_KEY,
        "Content-Type": "application/json",
    }
    logger.debug(f"Invoice headers prepared (Auth starts with: Bearer {token[:10]}..., API key prefix: {settings.NOWPAYMENTS_API_KEY[:10]}...)")
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

async def get_upgrade_tp_amount(base_plan: str, db: AsyncSession) -> int:
    """Fetch upgrade TP amount for the plan from database."""
    config_id = 1 if base_plan == 'pro' else 2
    result = await db.execute(
        select(UpgradeTpConfig.amount).where(
            UpgradeTpConfig.plan == base_plan,
            UpgradeTpConfig.id == config_id
        )
    )
    amount = result.scalar()
    return amount if amount is not None else 0

async def has_used_beta_code(user_id: int, db: AsyncSession) -> bool:
    """Check if the user signed up using a beta invite code."""
    result = await db.execute(
        select(BetaInvite.id).where(
            BetaInvite.used_by_id == user_id
        )
    )
    return result.scalar() is not None

async def create_direct_invoice(amount: float, order_id: str, order_description: str, token: str, success_params: str = "") -> str:
    """Create a direct invoice for immediate payment (crypto-only)."""
    headers = get_invoice_headers(token)
    async with httpx.AsyncClient() as client:
        invoice_payload = {
            "price_amount": amount,
            "price_currency": "usd",
            # Omitted 'pay_currency' to allow user to select crypto on invoice page (crypto-only)
            "order_id": order_id,
            "order_description": order_description,
            "ipn_callback_url": f"{settings.BASE_URL}/payments/webhook",
            # FIXED: Static success_url (no placeholder—NowPayments doesn't replace in payload; use sub_id lookup)
            "success_url": f"{settings.BASE_URL}/payment-success?payment=success{success_params}",
            "cancel_url": f"{settings.BASE_URL}/plans",
        }
        logger.info(f"Creating direct invoice: {invoice_payload}")
        resp = await client.post(
            f"{NOWPAYMENTS_BASE_URL}/invoice",
            json=invoice_payload,
            headers=headers
        )
        if resp.status_code not in [200, 201]:
            logger.error(f"Direct invoice creation failed: {resp.status_code} - {resp.text}")
            raise HTTPException(status_code=500, detail="Failed to generate payment link — please contact support.")
        
        invoice_data = resp.json()
        invoice_url = invoice_data.get("invoice_url")  # Flat response
        if not invoice_url:
            logger.error(f"No invoice_url in response: {invoice_data}")
            raise HTTPException(status_code=500, detail="Failed to generate payment link — please contact support.")
        
        # Extract invoice_id from response for logging/DB if needed
        invoice_id = invoice_data.get("invoice_id", "unknown")
        logger.info(f"Created direct invoice (crypto-only): {invoice_url} (ID: {invoice_id})")
        return invoice_url

def extract_invoice_id(invoice_url: str) -> Optional[str]:
    """Extract invoice ID from NowPayments invoice URL."""
    match = re.search(r'/invoice/([a-f0-9-]+)', invoice_url)
    return match.group(1) if match else None

async def get_invoice_status(invoice_url: str, token: str) -> dict:
    """Get the status of an existing invoice."""
    invoice_id = extract_invoice_id(invoice_url)
    if not invoice_id:
        logger.warning(f"Could not extract invoice ID from URL: {invoice_url}")
        return {"payment_status": "expired"}

    headers = get_invoice_headers(token)
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{NOWPAYMENTS_BASE_URL}/invoice/{invoice_id}",
                headers=headers
            )
            if resp.status_code == 200:
                status_data = resp.json()
                logger.info(f"[STATUS POLL] Fetched status for invoice {invoice_id}: {status_data.get('payment_status')}")
                return status_data
            else:
                logger.warning(f"Failed to fetch invoice status: {resp.status_code} - {resp.text}")
                return {"payment_status": "expired"}
        except Exception as e:
            logger.error(f"Error fetching invoice status: {e}")
            return {"payment_status": "expired"}

@router.post("/checkout")
async def create_subscription(
    request: Request,
    trader_id: Optional[int] = None,
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

    # UPDATED: Fetch discount if user signed up with referral OR beta code
    perc = 0.0
    eligible_for_discount = current_user.referred_by is not None or await has_used_beta_code(current_user.id, db)
    if eligible_for_discount:
        result_discount = await db.execute(
            select(Discount).where(Discount.id == 1)
        )
        db_discount = result_discount.scalar_one_or_none()
        if db_discount and db_discount.enabled and (not db_discount.expiry or db_discount.expiry > date.today()):
            perc = db_discount.percentage
            discount_reason = "referral" if current_user.referred_by is not None else "beta_code"
            logger.info(f"Applied {discount_reason} discount: {perc}% for user {current_user.id}")

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
        plan_title += f" ({int(perc)}% promo)"
    plan_type = f"{plan}_{interval}"

    # Check for existing active
    result = await db.execute(
        select(Subscription).where(
            Subscription.user_id == current_user.id,
            Subscription.plan_type == plan_type,
            Subscription.status == 'active'
        )
    )
    if result.scalars().first():
        raise HTTPException(
            status_code=400,
            detail="You are already subscribed to this plan."
        )

    # Cleanup pending/paused for this plan_type
    result_cleanup = await db.execute(
        select(Subscription).where(
            Subscription.user_id == current_user.id,
            Subscription.plan_type == plan_type,
            Subscription.status.in_(['pending', 'paused'])
        )
    )
    old_subs = result_cleanup.scalars().all()
    if old_subs:
        for old_sub in old_subs:
            await db.execute(
                update(Subscription).where(Subscription.id == old_sub.id).values(status='cancelled')
            )
            logger.info(f"Cancelled old sub {old_sub.id} for {plan_type} during new checkout")
    await db.commit()

    # FIXED: Create pending sub FIRST (no invoice_url yet), commit for real ID
    db_sub = Subscription(
        user_id=current_user.id,
        trader_id=trader_id,
        plan_type=plan_type,
        interval_days=interval_days,
        amount_usd=amount,
        status='pending',
        start_date=datetime.utcnow(),
        next_billing_date=datetime.utcnow() + timedelta(days=interval_days),
        order_id=f"{current_user.id}_{plan_type}",  # Temp; will set full after
        order_description=f"iTrade {plan_title} {'initial' if trader_id is None else 'marketplace'} payment",
        # renewal_url set after invoice
    )
    db.add(db_sub)
    await db.commit()  # Commit to get real db_sub.id

    # Now use real sub_id in params
    success_params = f"&sub_id={db_sub.id}"

    token = await get_nowpayments_token()
    order_id = db_sub.order_id  # Use the one from sub
    order_description = db_sub.order_description

    # Generate invoice (now with real sub_id)
    try:
        invoice_url = await create_direct_invoice(amount, order_id, order_description, token, success_params)
        logger.info(f"Invoice created for sub {db_sub.id}: {invoice_url}")
    except Exception as e:
        logger.error(f"Invoice creation error: {e}")
        # Rollback sub on failure
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to generate payment link — please contact support.")

    # Update sub with invoice_url
    db_sub.renewal_url = invoice_url
    await db.commit()

    return {
        "message": f"Payment link ready! Complete your {plan_title} subscription now.",
        "subscription_id": db_sub.id,  # Real DB ID
        "email": current_user.email,
        "invoice_url": invoice_url
    }

@router.post("/renew/{sub_id}")
async def renew_subscription(
    sub_id: int = Path(..., description="Subscription ID to renew"),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    # Fetch and validate sub
    result = await db.execute(
        select(Subscription).where(
            Subscription.id == sub_id,
            Subscription.user_id == current_user.id,
            Subscription.status.in_(['active', 'paused'])  # Allow renew from paused
        )
    )
    db_sub = result.scalars().first()
    if not db_sub:
        raise HTTPException(status_code=404, detail="Subscription not found or access denied.")

    if db_sub.status == 'active' and db_sub.next_billing_date > datetime.utcnow() + timedelta(days=1):
        raise HTTPException(status_code=400, detail="Subscription is active and not due for renewal yet.")

    # FIXED: Create pending Payment FIRST for tracking (no invoice_url yet)
    order_id = f"{current_user.id}_{db_sub.plan_type}_renew"  # e.g., "1_pro_monthly_renew"
    order_description = f"Renewal for {db_sub.plan_type} subscription {sub_id}"
    db_payment = Payment(
        user_id=current_user.id,
        subscription_id=sub_id,
        amount_usd=db_sub.amount_usd,
        status='generated',  # Custom: awaiting payment
        order_id=order_id,
        # invoice_url set after
    )
    db.add(db_payment)
    await db.commit()  # Commit to get db_payment.id if needed, but mainly for consistency

    success_params = f"&sub_id={sub_id}"  # Already real

    token = await get_nowpayments_token()

    # Generate renewal invoice
    try:
        invoice_url = await create_direct_invoice(db_sub.amount_usd, order_id, order_description, token, success_params)
        logger.info(f"Renewal invoice created for sub {sub_id}: {invoice_url}")
    except Exception as e:
        logger.error(f"Renewal invoice error: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to generate renewal link — please contact support.")

    # Update sub and payment with URL
    db_sub.status = 'pending_renewal'
    db_sub.renewal_url = invoice_url
    db_sub.updated_at = datetime.utcnow()
    db_payment.invoice_url = invoice_url
    await db.commit()

    # Optional: Trigger in-app notification (e.g., set flag or send your own email/push)

    return {
        "message": f"Renewal link ready for {db_sub.plan_type}! Pay to extend your subscription.",
        "subscription_id": sub_id,
        "invoice_url": invoice_url
    }

@router.post("/marketplace/checkout")
async def create_marketplace_subscription(
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    body = await request.json()
    trader_id = body.get("trader_id")
    interval = body.get("interval", "monthly")
    amount = body.get("amount", 10.0)  # This will now be pre-discounted from frontend

    if not trader_id:
        raise HTTPException(status_code=400, detail="trader_id required")

    if not current_user.email:
        raise HTTPException(status_code=400, detail="Email required for subscriptions")

    trader = await db.get(User, trader_id)
    if not trader or not trader.is_trader:
        raise HTTPException(status_code=404, detail="Trader not found or not available")

    # UPDATED: Fetch marketplace discount if user signed up with referral OR beta code
    perc = 0.0
    eligible_for_discount = current_user.referred_by is not None or await has_used_beta_code(current_user.id, db)
    if eligible_for_discount:
        result_marketplace_discount = await db.execute(
            select(Discount).where(Discount.id == 2)
        )
        db_marketplace_discount = result_marketplace_discount.scalar_one_or_none()
        if db_marketplace_discount and db_marketplace_discount.enabled and (not db_marketplace_discount.expiry or db_marketplace_discount.expiry > date.today()):
            perc = db_marketplace_discount.percentage
            discount_reason = "referral" if current_user.referred_by is not None else "beta_code"
            logger.info(f"[PAYMENTS DEBUG] Applied {discount_reason} marketplace discount: {perc}% for user {current_user.id}, trader {trader_id}, incoming amount: ${amount:.2f}")

    # FIXED: If amount matches original price, apply discount; else assume pre-discounted (from frontend)
    original_price = trader.marketplace_price or 19.99
    if abs(amount - original_price) < 0.01:  # Incoming is original → apply discount
        if perc > 0:
            discount_amount = amount * (perc / 100)
            amount -= discount_amount
            amount = max(amount, 0.01)
            logger.info(f"[PAYMENTS DEBUG] Discounted ${original_price:.2f} → ${amount:.2f} ({perc}% off)")
    else:
        # Already discounted (e.g., from frontend calc) — no re-apply
        logger.info(f"[PAYMENTS DEBUG] Amount ${amount:.2f} assumed pre-discounted (original was ${original_price:.2f})")

    # Check existing active
    existing_sub = await db.execute(
        select(Subscription).where(
            Subscription.user_id == current_user.id,
            Subscription.trader_id == trader_id,
            Subscription.status == 'active'
        )
    )
    if existing_sub.scalar():
        raise HTTPException(status_code=400, detail="Already subscribed to this trader")

    # Check for existing pending sub and validate invoice
    pending_result = await db.execute(
        select(Subscription).where(
            Subscription.user_id == current_user.id,
            Subscription.trader_id == trader_id,
            Subscription.status == 'pending'
        ).order_by(Subscription.start_date.desc())
    )
    pending_sub = pending_result.scalar_one_or_none()

    if pending_sub and pending_sub.renewal_url:
        token = await get_nowpayments_token()
        status_data = await get_invoice_status(pending_sub.renewal_url, token)
        payment_status = status_data.get("payment_status")
        if payment_status in ["new", "waiting"]:
            logger.info(f"Reusing existing payable invoice for user {current_user.id}, trader {trader_id}")
            return {
                "message": f"Payment link ready for subscription to {trader.full_name or trader.username}! (Reusing existing)",
                "subscription_id": pending_sub.id,
                "trader_id": trader_id,
                "email": current_user.email,
                "invoice_url": pending_sub.renewal_url
            }

    # If no valid pending or expired, cleanup old pending/paused
    result_cleanup = await db.execute(
        select(Subscription).where(
            Subscription.user_id == current_user.id,
            Subscription.trader_id == trader_id,
            Subscription.status.in_(['pending', 'paused'])
        )
    )
    old_subs = result_cleanup.scalars().all()
    if old_subs:
        for old_sub in old_subs:
            await db.execute(
                update(Subscription).where(Subscription.id == old_sub.id).values(status='cancelled')
            )
            logger.info(f"Cancelled old marketplace sub {old_sub.id} for trader {trader_id} during new checkout")
    await db.commit()

    interval_days = 30 if interval == "monthly" else 365
    plan_title = f"Sub to {trader.full_name or trader.username} - ${amount:.2f} {interval.title()}"
    if perc > 0:
        plan_title += f" ({int(perc)}% promo)"
    plan_type = f"marketplace_{trader_id}_{interval}"

    # FIXED: Create pending sub FIRST (no invoice_url yet), commit for real ID
    db_sub = Subscription(
        user_id=current_user.id,
        trader_id=trader_id,
        plan_type=plan_type,
        interval_days=interval_days,
        amount_usd=amount,  # Now the final discounted amount
        status='pending',
        start_date=datetime.utcnow(),
        next_billing_date=datetime.utcnow() + timedelta(days=interval_days),
        order_id=f"{current_user.id}_{plan_type}",  # Temp
        order_description=f"Subscription to {trader.full_name or trader.username}",
        # renewal_url set after invoice
    )
    db.add(db_sub)
    await db.commit()  # Commit to get real db_sub.id

    # Now use real sub_id in params
    success_params = f"&sub_id={db_sub.id}&trader_id={trader_id}"

    token = await get_nowpayments_token()
    order_id = db_sub.order_id
    order_description = db_sub.order_description

    # Generate new invoice (now with real sub_id)
    try:
        invoice_url = await create_direct_invoice(amount, order_id, order_description, token, success_params)
        logger.info(f"Marketplace invoice created for sub {db_sub.id}: {invoice_url}")
    except Exception as e:
        logger.error(f"Marketplace invoice error: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to generate payment link — please contact support.")

    # Update sub with invoice_url
    db_sub.renewal_url = invoice_url
    await db.commit()

    return {
        "message": f"Payment link ready for subscription to {trader.full_name or trader.username}!",
        "subscription_id": db_sub.id,  # Real ID
        "trader_id": trader_id,
        "email": current_user.email,
        "invoice_url": invoice_url
    }

# NEW: Manual verification endpoint (call from /payment-success polling on 'finished')
@router.post("/verify/{sub_id}")
async def verify_subscription(
    sub_id: int = Path(..., description="Subscription ID to verify"),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    result = await db.execute(
        select(Subscription).where(Subscription.id == sub_id, Subscription.user_id == current_user.id)
    )
    db_sub = result.scalar_one_or_none()
    if not db_sub or db_sub.status != 'pending':
        raise HTTPException(status_code=400, detail="Invalid subscription for verification")

    if not db_sub.renewal_url:
        raise HTTPException(status_code=400, detail="No invoice URL for verification")

    token = await get_nowpayments_token()
    status_data = await get_invoice_status(db_sub.renewal_url, token)
    payment_status = status_data.get("payment_status", "").lower().replace(" ", "_").replace("-", "_")

    if payment_status != "finished":
        logger.warning(f"Verification failed for sub {sub_id}: status '{payment_status}' (not 'finished')")
        raise HTTPException(status_code=400, detail=f"Payment not confirmed yet (status: {payment_status})")

    # Mimic webhook success logic
    user = await db.get(User, current_user.id)
    db_sub.status = "active"
    db_sub.next_billing_date = db_sub.start_date + timedelta(days=db_sub.interval_days)
    db_sub.renewal_url = None
    db_sub.updated_at = datetime.utcnow()
    if user:
        user.plan = db_sub.plan_type
        user.updated_at = datetime.utcnow()
    await db.commit()

    logger.info(f"Manual verification succeeded for sub {sub_id}: activated via polling")
    return {"message": "Subscription verified and activated!"}

@router.post("/webhook")
async def nowpayments_webhook(request: Request, db: AsyncSession = Depends(get_session)):
    # FIXED: Read body ONCE, then derive preview as str (avoids multiple reads + type error)
    body = await request.body()
    body_str = body.decode('utf-8', errors='ignore')  # Safe decode for preview/logging
    body_preview = body_str[:500] + "..." if len(body_str) > 500 else body_str
    logger.info(f"[WEBHOOK ENTRY] Incoming webhook: headers={dict(request.headers)}, body_preview={body_preview}")
    
    sig = request.headers.get("x-nowpayments-sig")
    if not verify_ipn(body, sig, settings.NOWPAYMENTS_IPN_SECRET):  # Use raw bytes for HMAC
        logger.warning("Invalid IPN signature received")
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        data = json.loads(body)  # Parse raw bytes as JSON
    except json.JSONDecodeError:
        logger.error("Invalid JSON in webhook body")
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    payment_status = data.get("payment_status")
    payment_id = data.get("payment_id")
    order_id = data.get("order_id")

    # FIXED: Normalize to lowercase underscore (handles spaces/upper)
    if payment_status:
        payment_status = payment_status.lower().replace(" ", "_").replace("-", "_")
    logger.info(f"[WEBHOOK DEBUG] Normalized payment_status: '{payment_status}' (original: '{data.get('payment_status')}')")

    if not payment_id or not order_id:
        logger.info(f"Webhook ignored: missing payment_id or order_id")
        return {"status": "ignored"}

    # ADDED: Log full payload for debugging (remove/sanitize in high-traffic prod)
    logger.info(f"[WEBHOOK DEBUG] Full payload for {payment_id}: {json.dumps(data, indent=2)}")

    # Parse order_id early for user_id/plan_type
    match = re.match(r"^(\d+)_(.+?)(?:_renew)?$", order_id)
    if not match:
        logger.warning(f"Invalid order_id format: {order_id}")
        return {"status": "ignored"}
    user_id_str, plan_type = match.groups()
    user_id = int(user_id_str)
    is_marketplace = plan_type.startswith("marketplace_")
    trader_id = None
    if is_marketplace:
        trader_match = re.match(r"marketplace_(\d+)_(.+)", plan_type)
        if trader_match:
            trader_id = int(trader_match.group(1))
            logger.info(f"[WEBHOOK DEBUG] Parsed marketplace: trader_id={trader_id}, extra={trader_match.group(2)}")  # e.g., 'monthly'

    # ADDED: Fallback search by exact order_id if plan_type match fails
    order_result = await db.execute(select(Subscription).where(Subscription.order_id == order_id))
    db_sub_from_order = order_result.scalars().first()

    # Find matching sub (for Payment linking)
    where_clause = [
        Subscription.user_id == user_id,
        Subscription.plan_type == plan_type,
        Subscription.status.in_(['pending', 'active'])
    ]
    if trader_id:
        where_clause.append(Subscription.trader_id == trader_id)
    result = await db.execute(
        select(Subscription).where(*where_clause).order_by(Subscription.id.desc())
    )
    db_sub = result.scalars().first()
    
    # Use fallback if needed
    if not db_sub and db_sub_from_order:
        db_sub = db_sub_from_order
        logger.info(f"[WEBHOOK DEBUG] Using fallback sub from exact order_id match: {db_sub.id}")
    
    if not db_sub:
        logger.info(f"Webhook ignored: no matching sub for order_id {order_id} (looked for plan_type='{plan_type}')")
        return {"status": "ignored"}

    logger.info(f"[WEBHOOK DEBUG] Matched sub {db_sub.id} (status: {db_sub.status}, plan_type: {db_sub.plan_type}) for order_id {order_id}")

    # ALWAYS update/create Payment record
    result = await db.execute(select(Payment).where(Payment.nowpayments_payment_id == payment_id))
    existing_payment = result.scalars().first()

    pay_date = None
    if data.get("pay_date"):
        pay_date = datetime.fromisoformat(data["pay_date"].replace("Z", "+00:00"))

    # FIXED: Compute USD equiv if missing (use ratio to avoid external API calls)
    expected_usd = float(data.get("price_amount", db_sub.amount_usd))
    pay_amount_crypto = float(data.get("pay_amount", 0))  # Expected crypto amount
    actually_paid_crypto = float(data.get("actually_paid", existing_payment.amount_paid_crypto if existing_payment else 0))
    actually_paid_usd = float(data.get("actually_paid_at_fiat", 0))

    if actually_paid_usd == 0 and pay_amount_crypto > 0:
        # Compute proportional USD: (paid_crypto / expected_crypto) * expected_usd
        paid_ratio = actually_paid_crypto / pay_amount_crypto
        actually_paid_usd = paid_ratio * expected_usd
        logger.info(f"[USD COMPUTE] Missing fiat; computed ${actually_paid_usd:.4f} from ratio {paid_ratio:.4f} "
                    f"({actually_paid_crypto} / {pay_amount_crypto} MATIC × ${expected_usd:.2f})")

    if existing_payment:
        existing_payment.status = payment_status  # Now normalized lowercase
        existing_payment.paid_at = pay_date
        existing_payment.amount_paid_crypto = actually_paid_crypto  # FIXED: From actually_paid
        existing_payment.amount_paid_usd = actually_paid_usd  # NEW: USD equiv for partials
        existing_payment.updated_at = datetime.utcnow()
        logger.info(f"[WEBHOOK DEBUG] Updated existing Payment {existing_payment.id} for {payment_id}")
    else:
        db_payment = Payment(
            user_id=user_id,
            subscription_id=db_sub.id,
            nowpayments_payment_id=payment_id,
            amount_usd=float(data.get("price_amount", 0)),
            amount_paid_crypto=actually_paid_crypto,  # FIXED: From actually_paid
            amount_paid_usd=actually_paid_usd,  # NEW: USD equiv for partials
            crypto_currency=data.get("pay_currency"),  # FIXED: Use pay_currency (standard field)
            status=payment_status,  # Normalized lowercase
            invoice_url=data.get("invoice_url"),
            paid_at=pay_date
        )
        db.add(db_payment)
        logger.info(f"[WEBHOOK DEBUG] Created new Payment for {payment_id}, linked to sub {db_sub.id}")

    await db.commit()  # Commit Payment early

    # SIMPLIFIED: Handle partials via NowPayments covering (trust "finished" if within tolerance)
    # For true partials (< covering %), log and optionally notify user to top-up
    if payment_status == "partially_paid":
        # IMPROVED: Log full payload for debugging (remove in prod)
        logger.info(f"[DEBUG] Partial payload for {payment_id}: {json.dumps(data, indent=2)}")

        diff_usd = expected_usd - actually_paid_usd
        logger.info(f"[PARTIAL DEBUG] Expected: ${expected_usd:.2f}, Paid USD equiv: ${actually_paid_usd:.2f}, Shortfall: ${diff_usd:.2f}")

        # Optional: Email user "Partial received—top up ${diff_usd:.2f} to complete"
        # TODO: Integrate your email service here

    # Handle sub/trader only for final statuses
    if payment_status in ["finished", "failed", "refunded"]:
        # ADDED: Tolerance check for 'finished' but slight underpay (e.g., volatility)
        if payment_status == "finished":
            paid = actually_paid_usd  # Now reliable
            expected = expected_usd
            if paid < expected * 0.95:  # 5% tolerance for underpay
                logger.warning(f"[TOLERANCE DEBUG] Finished but underpaid for {payment_id}: ${paid:.2f} vs ${expected:.2f} (short by ${(expected - paid):.2f}) - treating as partial")
                # Downgrade to partial handling (no sub activation)
                payment_status = "partially_paid"
                # Re-update payment status
                if existing_payment:
                    existing_payment.status = payment_status
                else:
                    db_payment.status = payment_status
                await db.commit()
            else:
                logger.info(f"[TOLERANCE DEBUG] Finished payment within tolerance: ${paid:.2f} >= ${expected * 0.95:.2f}")

        if payment_status == "partially_paid":
            logger.info(f"Webhook partial/update: payment {payment_id} status {payment_status} for order {order_id} (sub {db_sub.id}) - awaiting full/covering payment")
            # Optional: Trigger user email "Partial payment received—send remaining to complete!"
            return {"status": "ok"}

        # Handle trader earnings on success
        if trader_id and payment_status == "finished":
            trader = await db.get(User, trader_id)
            if trader:
                earnings = db_sub.amount_usd * 0.7
                trader.marketplace_earnings += earnings
                trader.monthly_earnings += earnings
                await db.commit()
                logger.info(f"Credited trader {trader_id} ${earnings:.2f} from {order_id}")

        # Update sub status FIRST (before user fetch)
        user_plan = None
        if payment_status == "finished":
            db_sub.status = "active"
            if "_renew" in order_id:
                db_sub.next_billing_date += timedelta(days=db_sub.interval_days)
            else:
                # Initial: Set from start_date
                db_sub.next_billing_date = db_sub.start_date + timedelta(days=db_sub.interval_days)
            db_sub.renewal_url = None  # Clear after payment
            user_plan = plan_type
            logger.info(f"[SUB UPDATE] Activated sub {db_sub.id} as active (initial/renew: {'renew' if '_renew' in order_id else 'initial'})")
        elif payment_status in ["failed", "refunded"]:
            db_sub.status = "paused"
            user_plan = "starter"
            logger.info(f"[SUB UPDATE] Paused sub {db_sub.id} due to {payment_status}")

        db_sub.updated_at = datetime.utcnow()

        # NEW: Grant TP on plan upgrade (only for platform plans, initial activation, not renewals/marketplace)
        if payment_status == "finished" and not is_marketplace and "_renew" not in order_id:
            base_plan = plan_type.split('_')[0]  # e.g., 'pro' or 'elite'
            if base_plan in ['pro', 'elite']:
                tp_amount = await get_upgrade_tp_amount(base_plan, db)
                if tp_amount > 0:
                    user = await db.get(User, user_id)
                    if user:
                        user.trade_points += tp_amount
                        upgrade_tx = PointTransaction(
                            user_id=user_id,
                            type='plan_upgrade',
                            amount=tp_amount,
                            description=f"{base_plan.title()} plan upgrade bonus"
                        )
                        db.add(upgrade_tx)
                        logger.info(f"Granted {tp_amount} TP for {base_plan} upgrade to user {user_id}")
                        await db.commit()

        # Now update user if exists
        user = await db.get(User, user_id)
        if user:
            user.plan = user_plan
            user.updated_at = datetime.utcnow()
            logger.info(f"[USER UPDATE] Set user {user_id} plan to {user_plan}")
        else:
            logger.error(f"[USER ERROR] Sub {db_sub.id} updated but user {user_id} not found - manual intervention needed!")

        await db.commit()

        logger.info(f"Webhook processed: {payment_status} for payment {payment_id} for order {order_id} (sub {db_sub.id})")
    else:
        # For partials/waiting: Log and maybe notify
        logger.info(f"Webhook partial/update: payment {payment_id} status {payment_status} for order {order_id} (sub {db_sub.id}) - awaiting full/covering payment")
        # Optional: Trigger user email "Partial payment received—send remaining to complete!"

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

# Optional: Cron job example (add to your main app startup)
async def auto_generate_renewals(db: AsyncSession):
    """Cron: Generate renewals for subs due in 3 days."""
    due_date = datetime.utcnow() + timedelta(days=3)
    result = await db.execute(
        select(Subscription).where(
            Subscription.status == 'active',
            Subscription.next_billing_date <= due_date
        )
    )
    subs = result.scalars().all()
    token = await get_nowpayments_token()
    for sub in subs:
        if sub.renewal_url:  # Already generated
            continue
        order_id = f"{sub.user_id}_{sub.plan_type}_renew"
        order_description = f"Auto-renewal for {sub.plan_type} sub {sub.id}"
        success_params = f"&sub_id={sub.id}"
        try:
            invoice_url = await create_direct_invoice(sub.amount_usd, order_id, order_description, token, success_params)
            # Create pending Payment
            db_payment = Payment(
                user_id=sub.user_id,
                subscription_id=sub.id,
                amount_usd=sub.amount_usd,
                status='generated',
                order_id=order_id,
                invoice_url=invoice_url
            )
            db.add(db_payment)
            sub.renewal_url = invoice_url
            sub.status = 'pending_renewal'  # Custom flag
            await db.commit()
            logger.info(f"Auto-generated renewal for sub {sub.id}: {invoice_url}")
            # TODO: Send in-app notification (e.g., update user notifications table)
        except Exception as e:
            logger.error(f"Auto-renewal failed for sub {sub.id}: {e}")

# NEW: Manual completion endpoint (for admins/tests) - Secure with admin dependency
# TODO: Implement get_admin_user() similar to get_current_user but check is_admin
@router.post("/admin/complete-payment")
async def manual_complete_payment(
    payment_id: str = Body(...),  # e.g., "5102708066"
    reason: str = Body("Test/manual override"),
    db: AsyncSession = Depends(get_session),
    # current_user=Depends(get_admin_user)  # Uncomment and implement
):
    # Fetch payment and sub
    result = await db.execute(select(Payment).where(Payment.nowpayments_payment_id == payment_id))
    db_payment = result.scalar_one_or_none()
    if not db_payment:
        raise HTTPException(404, "Payment not found")
    
    # Verify status
    if db_payment.status not in ["partially_paid", "failed"]:
        raise HTTPException(400, "Only partial/failed can be completed")
    
    # Link to sub if needed
    if not db_payment.subscription_id:
        # Parse from order_id or error
        raise HTTPException(400, "No linked subscription")
    
    result = await db.execute(select(Subscription).where(Subscription.id == db_payment.subscription_id))
    db_sub = result.scalar_one_or_none()
    if not db_sub:
        raise HTTPException(404, "Subscription not found")
    
    # Complete
    db_payment.status = "finished_manual"
    db_payment.notes = reason
    db_payment.updated_at = datetime.utcnow()
    
    db_sub.status = "active"
    db_sub.next_billing_date = db_sub.start_date + timedelta(days=db_sub.interval_days) if "_renew" not in db_payment.order_id else db_sub.next_billing_date + timedelta(days=db_sub.interval_days)
    db_sub.renewal_url = None
    db_sub.updated_at = datetime.utcnow()
    
    # Update user
    result = await db.execute(select(User).where(User.id == db_sub.user_id))
    user = result.scalar_one_or_none()
    if user:
        user.plan = db_sub.plan_type
        user.updated_at = datetime.utcnow()
    
    await db.commit()
    
    logger.info(f"Manually completed {payment_id} for sub {db_sub.id}: {reason}")
    return {"message": "Payment completed, subscription activated!"}

# NEW: Public status API for polling (no auth)
@router.get("/sub-status/{sub_id}")
async def sub_status(sub_id: int, db: AsyncSession = Depends(get_session)):
    result = await db.execute(
        select(Subscription.status, User.email).join(User, Subscription.user_id == User.id).where(Subscription.id == sub_id)
    )
    row = result.first()
    if row:
        return {"active": row.status == "active", "email": row.email or ""}
    return {"active": False, "email": ""}
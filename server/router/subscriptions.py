from fastapi import APIRouter, Depends, HTTPException, Request, status, Header, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, desc, func, and_
from sqlalchemy.orm import joinedload
from datetime import datetime, timedelta
from typing import Any
from math import ceil

from templates_config import templates
from models.models import User, Subscription, Payment  # FIXED: Direct import for clarity
from database import get_session
import auth


# Shared helpers
from .payments import get_nowpayments_token, create_direct_invoice  # FIXED: Assume payments.py is sibling module
from config import get_settings

settings = get_settings()
router = APIRouter(prefix="/subscriptions", tags=["Subscriptions"])


# ----------------------------------------------------------------------
# Helper – is the invoice still alive?
# ----------------------------------------------------------------------
def _invoice_is_fresh(payment: Payment) -> bool:
    """
    NowPayments invoices expire after ~1 hour.
    We keep the original link as long as it is younger than 55 minutes.
    """
    if not payment.created_at:
        return False
    age = datetime.utcnow() - payment.created_at
    return age < timedelta(minutes=55)


# ----------------------------------------------------------------------
# Helper: Render "Pay Now" button HTML (reusable for HTMX)
# ----------------------------------------------------------------------
def _render_pay_now_button(invoice_url: str, message: str = "Pay Now") -> str:
    return f'''
    <a href="{invoice_url}" class="btn btn-success action-btn w-100" target="_blank">
        <i class="bi bi-credit-card me-1"></i> {message}
    </a>
    <span class="text-success small ms-2">{message} link ready!</span>
    '''


# ----------------------------------------------------------------------
# NEW: Get full stats (independent of pagination/filter)
# ----------------------------------------------------------------------
async def get_stats(db: AsyncSession, user_id: int) -> dict[str, Any]:
    # active_count
    res = await db.execute(
        select(func.count(Subscription.id))
        .where(and_(Subscription.user_id == user_id, Subscription.status == "active"))
    )
    active_count = res.scalar() or 0

    # monthly_total
    res = await db.execute(
        select(func.sum(Subscription.amount_usd))
        .where(and_(Subscription.user_id == user_id, Subscription.status == "active"))
    )
    monthly_total = res.scalar() or 0.0

    # total_payments
    res = await db.execute(
        select(func.count(Payment.id))
        .join(Subscription, Payment.subscription_id == Subscription.id)
        .where(Subscription.user_id == user_id)
    )
    total_payments = res.scalar() or 0

    # successful
    res = await db.execute(
        select(func.count(Payment.id))
        .join(Subscription, Payment.subscription_id == Subscription.id)
        .where(
            and_(
                Subscription.user_id == user_id,
                Payment.status.in_(["finished", "finished_auto"])
            )
        )
    )
    successful = res.scalar() or 0

    success_rate = round((successful / total_payments * 100), 1) if total_payments else 0.0

    # platform_count
    res = await db.execute(
        select(func.count(Subscription.id))
        .where(and_(Subscription.user_id == user_id, Subscription.trader_id.is_(None)))
    )
    platform_count = res.scalar() or 0

    # marketplace_count
    res = await db.execute(
        select(func.count(Subscription.id))
        .where(and_(Subscription.user_id == user_id, Subscription.trader_id.is_not(None)))
    )
    marketplace_count = res.scalar() or 0

    return {
        "active_count": active_count,
        "monthly_total": round(monthly_total, 2),
        "total_payments": int(total_payments),
        "success_rate": success_rate,
        "platform_count": platform_count,
        "marketplace_count": marketplace_count,
    }


# ----------------------------------------------------------------------
# GET /subscriptions/
# ----------------------------------------------------------------------
@router.get("/", response_class=HTMLResponse)
async def list_subscriptions(
    request: Request,
    sub_type: str | None = Query(None, description="Filter by sub_type: platform, marketplace"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=50, description="Items per page"),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(auth.get_current_user)
):
    # Full stats
    stats = await get_stats(db, current_user.id)

    # Build where clause for filtered/paginated query
    where_clause = Subscription.user_id == current_user.id
    if sub_type == "platform":
        where_clause = and_(where_clause, Subscription.trader_id.is_(None))
    elif sub_type == "marketplace":
        where_clause = and_(where_clause, Subscription.trader_id.is_not(None))

    # Total count for pagination
    total_res = await db.execute(
        select(func.count(Subscription.id)).where(where_clause)
    )
    total_count = total_res.scalar() or 0
    total_pages = ceil(total_count / limit) if limit > 0 else 0

    # Paginated subscriptions
    offset = (page - 1) * limit
    subs_res = await db.execute(
        select(Subscription)
        .where(where_clause)
        .order_by(desc(Subscription.start_date))
        .offset(offset)
        .limit(limit)
    )
    subs = subs_res.scalars().all()

    enriched = []

    for sub in subs:
        # trader name
        trader = await db.get(User, sub.trader_id) if sub.trader_id else None
        trader_name = trader.full_name or trader.username if trader else "Platform Plan"

        # FIXED: Improved display for marketplace
        display_plan = f"Marketplace: {trader_name}" if sub.trader_id else sub.plan_type.replace("_", " ").title()

        # NEW: Explicit sub_type for template clarity
        sub_type_enriched = "marketplace" if sub.trader_id else "platform"

        # payments (last 10 for UI)
        pays_res = await db.execute(
            select(Payment)
            .where(Payment.subscription_id == sub.id)
            .order_by(desc(Payment.created_at))
        )
        payments = pays_res.scalars().all()

        # FIXED: Include partials in total_paid (uses amount_paid_usd now available)
        paid = round(sum(
            p.amount_usd if p.status in ["finished", "finished_auto"] 
            else (p.amount_paid_usd if p.status == "partially_paid" else 0)
            for p in payments
        ), 2)

        start = sub.start_date.strftime("%b %d, %Y") if sub.start_date else "N/A"
        next_b = sub.next_billing_date.strftime("%b %d, %Y") if sub.next_billing_date else "N/A"

        # Pay URL for pending states if fresh
        pay_url = None
        fake_payment = type("obj", (), {"created_at": sub.updated_at, "invoice_url": sub.renewal_url})
        if sub.status in ["pending", "pending_renewal"] and sub.renewal_url and _invoice_is_fresh(fake_payment):
            pay_url = sub.renewal_url

        enriched.append({
            "id": sub.id,
            "trader_name": trader_name,
            "plan_type": display_plan,  # FIXED: Better display
            "sub_type": sub_type_enriched,  # NEW: Explicit type for template
            "amount_usd": sub.amount_usd,
            "status": sub.status,
            "start_date": start,
            "next_billing": next_b,
            "total_paid": paid,
            "renewal_url": sub.renewal_url,
            "pay_url": pay_url,
            "payments": [
                {
                    "id": p.id,
                    "amount_usd": p.amount_usd,
                    "status": p.status,
                    "paid_at": p.paid_at.strftime("%b %d, %Y") if p.paid_at else "Pending",
                    "invoice_url": p.invoice_url,
                    "fresh": _invoice_is_fresh(p) if p.invoice_url else False,
                    # FIXED: Partial progress (now uses amount_paid_usd directly)
                    "partial_amount": f"${p.amount_paid_usd:.2f}" if p.status == "partially_paid" and p.amount_paid_usd > 0 else None,
                    "progress": f"{(p.amount_paid_usd / p.amount_usd * 100):.0f}%" if p.status == "partially_paid" and p.amount_usd > 0 and p.amount_paid_usd > 0 else "Partial",
                } for p in payments[-10:]
            ],
            "has_more_payments": len(payments) > 10,
        })

    return templates.TemplateResponse(
        "subscriptions.html",
        {
            "request": request,
            "subscriptions": enriched,
            "current_user": current_user,
            "now": datetime.utcnow(),
            "stats": stats,
            "sub_type": sub_type,
            "page": page,
            "limit": limit,
            "total_count": total_count,
            "total_pages": total_pages,
        },
    )


# ----------------------------------------------------------------------
# POST /subscriptions/{sub_id}/cancel
# ----------------------------------------------------------------------
@router.post("/{sub_id}/cancel")
async def cancel_subscription(
    sub_id: int,
    request: Request,  # Add Request to detect HTMX
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(auth.get_current_user),
):
    sub = await db.get(Subscription, sub_id)
    if not sub or sub.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Subscription not found")
    if sub.status != "active":
        raise HTTPException(status_code=400, detail="Only active subscriptions can be cancelled")

    sub.status = "cancelled"
    sub.updated_at = datetime.utcnow()
    await db.commit()

    # HTMX-friendly: Return partial HTML for the card if HTMX, else redirect
    if request.headers.get("HX-Request"):
        return HTMLResponse(
            '''
            <div class="alert alert-danger d-flex align-items-center">
                <i class="bi bi-check-circle me-2"></i>
                <span>Subscription cancelled successfully.</span>
            </div>
            ''',
            status_code=200
        )
    else:
        return RedirectResponse(url="/subscriptions/", status_code=status.HTTP_303_SEE_OTHER)


# ----------------------------------------------------------------------
# NEW: POST /subscriptions/{sub_id}/cancel_payment  (cancel pending payment/renewal)
# ----------------------------------------------------------------------
@router.post("/{sub_id}/cancel_payment")
async def cancel_pending_payment(
    sub_id: int,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(auth.get_current_user),
):
    sub = await db.get(Subscription, sub_id)
    if not sub or sub.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Subscription not found")
    if sub.status not in ("pending", "pending_renewal"):
        raise HTTPException(status_code=400, detail="Only pending payments can be cancelled")

    # Find the most recent pending payment and cancel it
    res = await db.execute(
        select(Payment)
        .where(
            Payment.subscription_id == sub_id,
            Payment.status.in_(["pending", "generated", "partially_paid"])
        )
        .order_by(desc(Payment.created_at))
    )
    payment = res.scalars().first()
    if payment:
        payment.status = "cancelled"
        payment.updated_at = datetime.utcnow()
        db.add(payment)

    # Adjust subscription status
    if sub.status == "pending":
        sub.status = "cancelled"
    elif sub.status == "pending_renewal":
        sub.status = "active"

    sub.renewal_url = None
    sub.updated_at = datetime.utcnow()
    await db.commit()

    # HTMX-friendly: Return partial HTML (alert replaces card temporarily)
    if request.headers.get("HX-Request"):
        action = "renewal" if sub.status == "active" else "subscription"
        return HTMLResponse(
            f'''
            <div class="alert alert-info d-flex align-items-center p-4">
                <i class="bi bi-check-circle me-2"></i>
                <span>Payment {action} cancelled successfully.</span>
            </div>
            ''',
            status_code=200
        )
    else:
        return RedirectResponse(url="/subscriptions/", status_code=status.HTTP_303_SEE_OTHER)


# ----------------------------------------------------------------------
# POST /subscriptions/{sub_id}/renew   (manual renewal)
# ----------------------------------------------------------------------
@router.post("/{sub_id}/renew")
async def manual_renew_subscription(
    sub_id: int,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(auth.get_current_user),
):
    sub = await db.get(Subscription, sub_id)
    if not sub or sub.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Subscription not found")
    if sub.status not in ("active", "pending_renewal", "pending"):
        raise HTTPException(status_code=400, detail="Not eligible for renewal")

    is_htmx = request.headers.get("HX-Request") is not None

    # If we already have a fresh renewal URL → just return the button
    if sub.renewal_url:
        fake_payment = type("obj", (), {"created_at": sub.updated_at, "invoice_url": sub.renewal_url})
        if _invoice_is_fresh(fake_payment):
            if is_htmx:
                return HTMLResponse(_render_pay_now_button(sub.renewal_url, "Pay Now"))
            else:
                return RedirectResponse(url="/subscriptions/", status_code=status.HTTP_303_SEE_OTHER)

    # FIXED: Use consistent order_id for renewals (matches webhook parsing)
    token = await get_nowpayments_token()
    order_id = f"{current_user.id}_{sub.plan_type}_renew"  # No manual/timestamp – reuse for consistency
    invoice_url = await create_direct_invoice(
        amount=sub.amount_usd,
        order_id=order_id,
        order_description=f"Manual renewal – Sub {sub.id}",
        token=token,
        success_params=f"&sub_id={sub.id}",
    )

    # Create Payment (now with order_id)
    payment = Payment(
        user_id=current_user.id,
        subscription_id=sub.id,
        amount_usd=sub.amount_usd,
        status="generated",
        order_id=order_id,  # Now valid!
        invoice_url=invoice_url,
        created_at=datetime.utcnow(),  # Explicit for safety
    )
    db.add(payment)

    sub.renewal_url = invoice_url
    sub.status = "pending_renewal"
    sub.updated_at = datetime.utcnow()
    await db.commit()

    if is_htmx:
        return HTMLResponse(_render_pay_now_button(invoice_url, "New invoice ready!"))
    else:
        return RedirectResponse(url="/subscriptions/", status_code=status.HTTP_303_SEE_OTHER)


# ----------------------------------------------------------------------
# POST /subscriptions/{sub_id}/retry   (regenerate any pending/failed)
# ----------------------------------------------------------------------
@router.post("/{sub_id}/retry")
async def retry_or_regenerate_payment(
    sub_id: int,
    request: Request,  # FIXED: Add Request for consistency (though not used yet)
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(auth.get_current_user),
):
    sub = await db.get(Subscription, sub_id)
    if not sub or sub.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Subscription not found")

    # Find the newest payment that needs a new invoice
    res = await db.execute(
        select(Payment)
        .where(
            Payment.subscription_id == sub_id,
            Payment.status.in_(["failed", "pending", "generated"]),
        )
        .order_by(desc(Payment.created_at))
    )
    payment = res.scalars().first()
    if not payment:
        raise HTTPException(status_code=400, detail="No payment to regenerate")

    # --------------------------------------------------------------
    # If the existing URL is still fresh → just return it
    # --------------------------------------------------------------
    if payment.invoice_url and _invoice_is_fresh(payment):
        return HTMLResponse(
            f'''
            <a href="{payment.invoice_url}" class="btn btn-sm btn-success action-btn" target="_blank">
                <i class="bi bi-credit-card"></i> Pay Now
            </a>
            <span class="text-success small ms-2">Valid link</span>
            '''
        )

    # --------------------------------------------------------------
    # FIXED: Use consistent order_id based on sub type (matches webhook parsing)
    # --------------------------------------------------------------
    is_renewal = sub.status == "pending_renewal"
    base_order_id = sub.order_id or f"{current_user.id}_{sub.plan_type}"
    if is_renewal and not base_order_id.endswith("_renew"):
        base_order_id += "_renew"
    # Use base without timestamp for consistency (NowPayments handles duplicates via payment_id)

    token = await get_nowpayments_token()
    invoice_url = await create_direct_invoice(
        amount=payment.amount_usd,
        order_id=base_order_id,  # FIXED: Consistent
        order_description=f"Regenerated payment – Sub {sub.id}",
        token=token,
        success_params=f"&sub_id={sub.id}",
    )

    payment.status = "generated"  # Reset to generated
    payment.invoice_url = invoice_url
    payment.order_id = base_order_id  # FIXED: Update to consistent
    payment.updated_at = datetime.utcnow()
    await db.commit()

    return HTMLResponse(
        f'''
        <a href="{invoice_url}" class="btn btn-sm btn-success action-btn" target="_blank">
            <i class="bi bi-credit-card"></i> Pay Now
        </a>
        <span class="text-success small ms-2">New invoice generated!</span>
        '''
    )


@router.get("/current")
async def get_current_subscription(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(auth.get_current_user)
):
    # Get the latest subscription
    query = (
        select(Subscription)
        .where(Subscription.user_id == current_user.id)
        .order_by(desc(Subscription.start_date))
        .limit(1)
    )
    result = await db.execute(query)
    sub = result.scalars().first()

    if not sub or sub.status not in ["active", "pending"]:
        return {
            "status": "free",
            "plan": "starter"
        }

    # FIXED: Handle marketplace plan_type parsing
    if sub.plan_type.startswith("marketplace_"):
        plan = "marketplace"
        interval = sub.plan_type.split("_")[-1]  # e.g., 'monthly'
    else:
        plan_parts = sub.plan_type.split("_") if "_" in sub.plan_type else [sub.plan_type, "monthly"]
        plan = plan_parts[0]
        interval = plan_parts[1] if len(plan_parts) > 1 else "monthly"

    status = "active" if sub.status == "active" else sub.status

    return {
        "status": status,
        "plan": plan,
        "interval": interval,
        "amount": float(sub.amount_usd),
        "next_billing": sub.next_billing_date.isoformat() if sub.next_billing_date else None,
        "subscription_id": sub.id
    }
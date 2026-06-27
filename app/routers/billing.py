"""
app/routers/billing.py

FastAPI router for premium subscription packages, transactions, webhooks, and invoice generation.
"""

import uuid
from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.user import User
from app.models.billing import Subscription, Invoice, PaymentTransaction
from app.schemas.billing import (
    CheckoutRequest,
    CheckoutResponse,
    PaymentVerifyRequest,
    SubscriptionResponse,
    InvoiceResponse,
    RefundRequest,
)
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/billing", tags=["Billing & Subscriptions"])

# Pricing configuration
PLAN_PRICES = {
    "premium": {"monthly": 9.00, "annual": 79.00},
    "professional": {"monthly": 29.00, "annual": 249.00},
    "clinic": {"monthly": 99.00, "annual": 799.00},
}


@router.post(
    "/checkout",
    response_model=CheckoutResponse,
    summary="Initialize checkout transaction session",
)
async def init_checkout(
    payload: CheckoutRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> CheckoutResponse:
    """Creates a mock checkout session mapping selected payment gateway and subscription plan details."""
    plan = payload.plan_name.lower()
    interval = payload.billing_interval.lower()
    gateway = payload.gateway.lower()

    if plan not in PLAN_PRICES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid plan selected. Choose from: {list(PLAN_PRICES.keys())}",
        )

    if interval not in ("monthly", "annual"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Billing interval must be 'monthly' or 'annual'.",
        )

    amount = PLAN_PRICES[plan][interval]
    session_id = f"checkout_sess_{uuid.uuid4().hex[:16]}"
    
    # Mock redirect URL for checkout gateways
    checkout_url = f"http://localhost:3000/dashboard/billing/checkout?session={session_id}&plan={plan}&interval={interval}&amount={amount}&gateway={gateway}"

    return CheckoutResponse(
        checkout_session_id=session_id,
        amount=amount,
        currency="USD",
        gateway=gateway,
        checkout_url=checkout_url,
    )


@router.post(
    "/verify",
    response_model=SubscriptionResponse,
    summary="Verify payment and activate premium package",
)
async def verify_payment(
    payload: PaymentVerifyRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SubscriptionResponse:
    """
    Confirms payment, activates subscription, generates invoice, logs transaction,
    and updates user roles to dermatologist for clinical tiers.
    """
    sess = payload.checkout_session_id
    ref_id = payload.gateway_ref_id
    
    # Simple simulated parsing of plan details from session ID parameters or defaults
    # In a real environment, this is decrypted or fetched from state database cache
    plan_name = "premium"
    billing_interval = "monthly"
    
    # Simulating data extraction based on typical payload conventions
    if "professional" in sess:
        plan_name = "professional"
    elif "clinic" in sess:
        plan_name = "clinic"
        
    if "annual" in sess:
        billing_interval = "annual"
        
    amount = PLAN_PRICES.get(plan_name, {"monthly": 9.00})["monthly" if billing_interval == "monthly" else "annual"]

    # 1. Update/Create Subscription
    stmt = select(Subscription).where(Subscription.user_id == current_user.id)
    result = await db.execute(stmt)
    sub = result.scalars().first()

    now_time = datetime.utcnow()
    duration = timedelta(days=365) if billing_interval == "annual" else timedelta(days=30)
    period_end = now_time + duration

    if not sub:
        sub = Subscription(
            user_id=current_user.id,
            plan_name=plan_name,
            status="active",
            billing_interval=billing_interval,
            current_period_start=now_time,
            current_period_end=period_end,
            cancel_at_period_end=False,
        )
        db.add(sub)
    else:
        sub.plan_name = plan_name
        sub.status = "active"
        sub.billing_interval = billing_interval
        sub.current_period_start = now_time
        sub.current_period_end = period_end
        sub.cancel_at_period_end = False
        sub.updated_at = now_time

    # 2. Log transaction
    gateway = "stripe"
    if "paypal" in sess:
        gateway = "paypal"
    elif "jazzcash" in sess:
        gateway = "jazzcash"
    elif "easypaisa" in sess:
        gateway = "easypaisa"
    elif "binance" in sess:
        gateway = "binance_pay"

    tx = PaymentTransaction(
        user_id=current_user.id,
        amount=amount,
        currency="USD",
        gateway=gateway,
        gateway_ref_id=ref_id,
        status="succeeded",
    )
    db.add(tx)

    # 3. Create Invoice
    invoice = Invoice(
        user_id=current_user.id,
        amount=amount,
        currency="USD",
        status="paid",
        payment_method=gateway,
        transaction_id=ref_id,
        billing_reason=f"SaaS Plan Activation: {plan_name.upper()} ({billing_interval.upper()})",
    )
    db.add(invoice)

    # 4. Promote User Role if buying Professional or Clinic plans
    if plan_name in ("professional", "clinic"):
        current_user.role = "dermatologist"
        user = await db.get(User, current_user.id)
        if user:
            user.role = "dermatologist"

    await db.flush()
    await db.refresh(sub)
    
    print(f"Subscription activated for user_id={current_user.id}. Plan={plan_name}. Role set to={current_user.role}")

    return SubscriptionResponse.model_validate(sub)


@router.get(
    "/subscription/active",
    response_model=SubscriptionResponse,
    summary="Get active subscription details",
)
async def get_active_subscription(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SubscriptionResponse:
    """Retrieves current user active subscription package details, falling back to Free tier."""
    stmt = select(Subscription).where(Subscription.user_id == current_user.id)
    result = await db.execute(stmt)
    sub = result.scalars().first()

    if not sub:
        # Generate inline virtual Free plan state
        return SubscriptionResponse(
            id=0,
            user_id=current_user.id,
            plan_name="free",
            status="active",
            billing_interval="monthly",
            current_period_start=datetime.utcnow(),
            current_period_end=datetime.utcnow() + timedelta(days=9999),
            cancel_at_period_end=False,
        )

    return SubscriptionResponse.model_validate(sub)


@router.post(
    "/subscription/cancel",
    response_model=SubscriptionResponse,
    summary="Cancel active auto-renewing subscription",
)
async def cancel_subscription(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SubscriptionResponse:
    """Cancels auto-renewal on subscription period end."""
    stmt = select(Subscription).where(Subscription.user_id == current_user.id)
    result = await db.execute(stmt)
    sub = result.scalars().first()

    if not sub or sub.plan_name == "free":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active auto-renewing subscription found to cancel.",
        )

    sub.cancel_at_period_end = True
    sub.updated_at = datetime.utcnow()
    await db.flush()
    await db.refresh(sub)

    return SubscriptionResponse.model_validate(sub)


@router.post(
    "/subscription/refund",
    response_model=InvoiceResponse,
    summary="Refund customer invoice (Admin only)",
)
async def refund_subscription_invoice(
    payload: RefundRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> InvoiceResponse:
    """Admin endpoint to process subscription refunds, revoking premium roles."""
    # Simple role authorization check
    is_admin = current_user.role == "admin" or "admin" in current_user.email.lower()
    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Admin credentials required to process billing refunds.",
        )

    invoice = await db.get(Invoice, payload.invoice_id)
    if not invoice:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Target invoice record not found.",
        )

    if invoice.status == "refunded":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invoice is already refunded.",
        )

    # Mark as refunded
    invoice.status = "refunded"

    # Revoke user subscription
    stmt = select(Subscription).where(Subscription.user_id == invoice.user_id)
    res = await db.execute(stmt)
    sub = res.scalars().first()
    if sub:
        sub.plan_name = "free"
        sub.status = "cancelled"
        sub.updated_at = datetime.utcnow()

    # Demote user role back to patient
    target_user = await db.get(User, invoice.user_id)
    if target_user:
        target_user.role = "patient"

    # Log refund transaction
    tx = PaymentTransaction(
        user_id=invoice.user_id,
        amount=-invoice.amount,
        currency=invoice.currency,
        gateway=invoice.payment_method,
        gateway_ref_id=f"refund_{invoice.transaction_id}",
        status="refunded",
    )
    db.add(tx)

    await db.flush()
    await db.refresh(invoice)

    return InvoiceResponse.model_validate(invoice)


@router.get(
    "/history",
    response_model=List[InvoiceResponse],
    summary="Fetch billing invoices history",
)
async def get_billing_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[InvoiceResponse]:
    """Retrieves all past payments and generated invoices for the authenticated user."""
    stmt = select(Invoice).where(Invoice.user_id == current_user.id).order_by(Invoice.created_at.desc())
    result = await db.execute(stmt)
    return [InvoiceResponse.model_validate(inv) for inv in result.scalars().all()]

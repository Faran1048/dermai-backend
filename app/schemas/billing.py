"""
app/schemas/billing.py

Pydantic v2 schemas for subscription checkouts, verification, and responses.
"""

from datetime import datetime
from pydantic import BaseModel, Field


class CheckoutRequest(BaseModel):
    plan_name: str = Field(..., examples=["premium", "professional", "clinic"])
    billing_interval: str = Field(..., examples=["monthly", "annual"])
    gateway: str = Field(..., examples=["stripe", "paypal", "jazzcash", "easypaisa", "binance_pay"])


class CheckoutResponse(BaseModel):
    checkout_session_id: str
    amount: float
    currency: str
    gateway: str
    checkout_url: str


class PaymentVerifyRequest(BaseModel):
    checkout_session_id: str
    gateway_ref_id: str
    status: str = "succeeded"


class SubscriptionResponse(BaseModel):
    id: int
    user_id: int
    plan_name: str
    status: str
    billing_interval: str
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool

    class Config:
        from_attributes = True


class InvoiceResponse(BaseModel):
    id: int
    user_id: int
    amount: float
    currency: str
    status: str
    payment_method: str
    transaction_id: str
    billing_reason: str
    created_at: datetime

    class Config:
        from_attributes = True


class RefundRequest(BaseModel):
    invoice_id: int
    reason: str | None = None

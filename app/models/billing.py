"""
app/models/billing.py

SQLAlchemy ORM models for billing, subscriptions, invoices, and payment logs.
"""

from datetime import datetime
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True, nullable=False
    )
    plan_name: Mapped[str] = mapped_column(String(50), default="free", nullable=False)  # 'free', 'premium', 'professional', 'clinic'
    status: Mapped[str] = mapped_column(String(50), default="active", nullable=False)  # 'active', 'past_due', 'cancelled', 'unpaid'
    billing_interval: Mapped[str] = mapped_column(String(50), default="monthly", nullable=False)  # 'monthly', 'annual'
    current_period_start: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    current_period_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    cancel_at_period_end: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationship to user
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])  # noqa: F821

    def __repr__(self) -> str:
        return f"<Subscription id={self.id} user_id={self.user_id} plan={self.plan_name!r}>"


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), default="USD", nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="paid", nullable=False)  # 'paid', 'unpaid', 'refunded'
    payment_method: Mapped[str] = mapped_column(String(50), nullable=False)  # 'stripe', 'paypal', 'jazzcash', 'easypaisa', 'binance_pay'
    transaction_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    billing_reason: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])  # noqa: F821

    def __repr__(self) -> str:
        return f"<Invoice id={self.id} amount={self.amount} status={self.status}>"


class PaymentTransaction(Base):
    __tablename__ = "payment_transactions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), default="USD", nullable=False)
    gateway: Mapped[str] = mapped_column(String(50), nullable=False)  # 'stripe', 'paypal', 'jazzcash', 'easypaisa', 'binance_pay'
    gateway_ref_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="succeeded", nullable=False)  # 'pending', 'succeeded', 'failed', 'refunded'
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])  # noqa: F821

    def __repr__(self) -> str:
        return f"<PaymentTransaction id={self.id} gateway={self.gateway!r} status={self.status}>"

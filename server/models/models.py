# Updated models/models.py
from sqlalchemy import Column, Integer, String, Text, JSON, Float, ForeignKey, DateTime, Boolean, Date, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base
from enum import Enum

# Enums for trade attributes
class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class AssetType(Enum):
    FOREX = "FOREX"
    CRYPTO = "CRYPTO"

    @classmethod
    def _missing_(cls, value):
        # FIXED: Handle lowercase inputs gracefully (e.g., 'forex' -> FOREX)
        if isinstance(value, str):
            upper_value = value.upper()
            for member in cls:
                if member.value == upper_value:
                    return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")

# NEW: Custom Enum type for case-insensitive loading
class CaseInsensitiveSQLEnum(SQLEnum):
    def result_processor(self, dialect, coltype):
        parent_processor = super().result_processor(dialect, coltype)
        def process(value):
            # Uppercase before parent's validation/lookup
            if isinstance(value, str):
                value = value.upper()
            if parent_processor:
                value = parent_processor(value)
            return value
        return process

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    email = Column(String, unique=True, index=True, nullable=False)
    account_balance = Column(Float, default=10000.0)  # Current account size in USD (auto-updated from trades)
    initial_deposit = Column(Float, default=10000.0, nullable=False)  # Fixed starting balance (user-settable for equity curve)
    password_hash = Column(String, nullable=True)
    referral_code = Column(String, unique=True, index=True)
    referred_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    is_admin = Column(Integer, default=0)

    plan = Column(String, default='starter', nullable=False)  # e.g., 'starter', 'pro', 'elite'
    insights_credits = Column(Integer, default=0, nullable=False)
    ai_chats_used = Column(Integer, default=0, nullable=False)
    bio = Column(String, nullable=True)
    trading_style = Column(String, nullable=True)  # e.g., 'scalping', 'swing'
    goals = Column(String, nullable=True)
    psych_zone = Column(String, nullable=True)  # Psychological state or trading zone
    strategy = Column(String, nullable=True)
    strategy_desc = Column(Text, nullable=True)
    preferred_strategies = Column(Text, nullable=True)  # e.g., "Momentum, Breakouts, Scalping" (FIXED: Added field)

    # NEW: For marketplace
    is_trader_pending = Column(Boolean, default=False)
    is_trader = Column(Boolean, default=False)  # Opt-in to be discoverable
    win_rate = Column(Float, default=0.0)  # Cached/computed win rate
    marketplace_price = Column(Float, default=19.99, nullable=False)  # Monthly subscription price for this trader

    # NEW: For marketplace earnings and payouts
    marketplace_earnings = Column(Float, default=0.0)
    monthly_earnings = Column(Float, default=0.0)
    wallet_address = Column(String, nullable=True)
    last_payout_date = Column(Date, nullable=True)
    payout_threshold = Column(Float, default=50.0)

    ai_detect = Column(Boolean, default=True)  # Enable AI trade detection
    risk_per_trade = Column(Float, nullable=True, default=1.0)  # Default risk % per trade
    daily_loss_percent = Column(Float, nullable=True, default=5.0)  # Daily loss limit %
    daily_loss_limit = Column(Float, nullable=True, default=500.0)  # Daily loss $ limit
    stop_loss = Column(Boolean, default=True)  # Enforce stop-loss
    no_revenge = Column(Boolean, default=False)  # Prevent revenge trading
    
    notes = Column(Text, nullable=True)
    preferred_timeframes = Column(JSON, default=list)  # e.g., ["M15", "H1"]
    risk_tolerance = Column(Integer, default=5)  # 1-10 scale
    recommendations = Column(JSON, default=dict)  # AI-generated recommendations
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    trades = relationship("Trade", back_populates="owner", cascade="all, delete-orphan")
    insights = relationship("TradeInsight", back_populates="user", cascade="all, delete-orphan")
    # FIXED: Specify foreign_keys to resolve multiple FK paths (user_id vs trader_id)
    subscriptions = relationship("Subscription", back_populates="user", foreign_keys="Subscription.user_id", cascade="all, delete-orphan")

class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    symbol = Column(String, nullable=True)  # e.g., EUR/USD, BTC/USDT
    trade_date = Column(DateTime, nullable=True)  # ISO date-time of trade (REQUIRED for accurate equity curve)
    entry_price = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    sl_price = Column(Float, nullable=True)  # Stop-loss price
    tp_price = Column(Float, nullable=True)  # Take-profit price
    direction = Column(SQLEnum(TradeDirection), nullable=True)
    position_size = Column(Float, nullable=True)  # Lots (forex) or units (crypto)
    leverage = Column(Float, nullable=True)  # e.g., 10.0 for 10x
    pnl = Column(Float, nullable=True)  # Profit/loss in % of account (e.g., 5.0 for +5%)
    notes = Column(String, nullable=True)  # Trade notes, including pip distance for forex
    session = Column(String, nullable=True)  # e.g., London, NY
    strategy = Column(String, nullable=True)  # e.g., Breakout, Reversal
    fees = Column(Float, default=0.0)  # Trading fees
    ai_log = Column(Text, nullable=True)  # AI extraction log
    chart_url = Column(String, nullable=True)  # URL to saved chart image
    risk_percentage = Column(Float, nullable=True)  # % of account risked
    risk_amount = Column(Float, nullable=True)  # $ risked on SL
    reward_amount = Column(Float, nullable=True)  # $ on TP
    r_r_ratio = Column(Float, nullable=True)  # Risk:Reward ratio
    suggestion = Column(String, nullable=True)  # AI-generated trade suggestion
    tags = Column(JSON, nullable=True)  # Custom tags
    raw_ai_response = Column(Text, nullable=True)  # Raw AI output
    confidence = Column(Float, nullable=True)  # AI confidence score
    asset_type = Column(CaseInsensitiveSQLEnum(AssetType), nullable=True)  # UPDATED: Use custom type
    owner = relationship("User", back_populates="trades")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class TradeInsight(Base):
    __tablename__ = "trade_insights"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    total_trades = Column(Integer, nullable=False)  # Number of trades analyzed
    insights_json = Column(Text, nullable=False)  # JSON string of insights
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="insights")

class Subscription(Base):
    __tablename__ = "subscriptions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    trader_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # For marketplace subs
    order_id = Column(String, nullable=True)
    order_description = Column(String, nullable=True)
    plan_type = Column(String, nullable=False)  # e.g., pro_monthly, marketplace_trader_1_monthly
    interval_days = Column(Integer, nullable=False)  # Billing interval
    amount_usd = Column(Float, nullable=False)
    status = Column(String, default='pending')  # pending, active, paused, cancelled
    start_date = Column(DateTime, default=datetime.utcnow)
    next_billing_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    renewal_url = Column(String, nullable=True)  # NEW: For storing generated invoice URLs
    user = relationship("User", back_populates="subscriptions", foreign_keys="Subscription.user_id")
    payments = relationship("Payment", back_populates="subscription", cascade="all, delete-orphan")

class Payment(Base):
    __tablename__ = "payments"
    id = Column(Integer, primary_key=True, index=True)
    subscription_id = Column(Integer, ForeignKey("subscriptions.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    nowpayments_payment_id = Column(String, unique=True, nullable=True)
    amount_usd = Column(Float, nullable=False)
    amount_paid_crypto = Column(Float, nullable=True)
    amount_paid_usd = Column(Float, default=0.0, nullable=False)
    crypto_currency = Column(String, nullable=True)
    status = Column(String, default='pending')
    invoice_url = Column(String, nullable=True)
    order_id = Column(String(255))
    paid_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User")
    subscription = relationship("Subscription", back_populates="payments")

class Pricing(Base):
    __tablename__ = "pricing"
    id = Column(Integer, primary_key=True, index=True)
    plan = Column(String, index=True)  # e.g., starter, premium, marketplace
    interval = Column(String, index=True)  # e.g., monthly, yearly
    amount = Column(Float)  # Price in USD

class Discount(Base):
    __tablename__ = "discounts"
    id = Column(Integer, primary_key=True, index=True)
    enabled = Column(Boolean, default=False)
    percentage = Column(Float, default=0.0)
    expiry = Column(Date, nullable=True)  # Discount expiration date

class EligibilityConfig(Base):
    __tablename__ = 'eligibility_config'
    id = Column(Integer, primary_key=True, index=True)
    min_trades = Column(Integer, default=50)
    min_win_rate = Column(Float, default=80.0)
    max_marketplace_price = Column(Float, default=99.99)
    is_active = Column(Boolean, default=True)

class UploadLimits(Base):
    __tablename__ = "upload_limits"
    id = Column(Integer, primary_key=True, index=True)
    plan = Column(String, index=True, unique=True, nullable=False)  # e.g., 'starter', 'pro', 'elite'
    monthly_limit = Column(Integer, default=3, nullable=False)  # Number of uploads per month
    batch_limit = Column(Integer, default=3, nullable=False)  # Max files per batch
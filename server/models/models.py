# Updated models/models.py
from sqlalchemy import Column,Index, Integer, String, Text, JSON, Float, ForeignKey, DateTime, Boolean, Date, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from database import Base
from enum import Enum
from sqlalchemy import UniqueConstraint  # NEW: For unique constraints on Referral

# Enums for trade attributes (unchanged)
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

# UPDATED: User model - Added trade_points, tier for referrals, chat limits tracking
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
    # UPDATED: Centralize to trade_points (replaces insights_credits; migrate on upgrade)
    trade_points = Column(Integer, default=3, nullable=False)  # Unified TP balance (starts at 3 for free tier)
    ai_chats_used = Column(Integer, default=0, nullable=False)  # Tracks text/voice chats (reset monthly via cron)
    # NEW: Referral tier (gamification)
    referral_tier = Column(String, default='rookie', nullable=False)  # e.g., 'rookie', 'pro_trader', 'elite_alpha'
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
    
    beta_invites_owned = relationship("BetaInvite", foreign_keys="BetaInvite.owner_id", back_populates="owner")
    beta_invites_used = relationship("BetaInvite", foreign_keys="BetaInvite.used_by_id")
    trades = relationship("Trade", back_populates="owner", cascade="all, delete-orphan")
    insights = relationship("TradeInsight", back_populates="user", cascade="all, delete-orphan")
    # FIXED: Specify foreign_keys to resolve multiple FK paths (user_id vs trader_id)
    subscriptions = relationship("Subscription", back_populates="user", foreign_keys="Subscription.user_id", cascade="all, delete-orphan")
    # NEW: Relationships for referrals and points
    referrals = relationship("Referral", back_populates="referrer", foreign_keys="Referral.referrer_id", cascade="all, delete-orphan")
    point_transactions = relationship("PointTransaction", back_populates="user", cascade="all, delete-orphan")

# Trade model (unchanged)
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

# Notification model (unchanged)
class Notification(Base):
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    type = Column(String, default="info")  # e.g., "approval", "rejection", "info"
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# TradeInsight model (unchanged)
class TradeInsight(Base):
    __tablename__ = "trade_insights"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    total_trades = Column(Integer, nullable=False)  # Number of trades analyzed
    insights_json = Column(Text, nullable=False)  # JSON string of insights
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="insights")

# Subscription model (unchanged)
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
    # NEW: Link to referrals for marketplace (if sub via ref)
    referral = relationship("Referral", back_populates="subscription", uselist=False)  # Optional 1:1

# Payment model (unchanged, but add ref tracking if needed)
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

# Pricing model (unchanged)
class Pricing(Base):
    __tablename__ = "pricing"
    id = Column(Integer, primary_key=True, index=True)
    plan = Column(String, index=True)  # e.g., starter, premium, marketplace
    interval = Column(String, index=True)  # e.g., monthly, yearly
    amount = Column(Float)  # Price in USD

# Discount model (unchanged)
class Discount(Base):
    __tablename__ = "discounts"
    id = Column(Integer, primary_key=True, index=True)
    enabled = Column(Boolean, default=False)
    percentage = Column(Float, default=0.0)
    expiry = Column(Date, nullable=True)  # Discount expiration date

# EligibilityConfig model (unchanged)
class EligibilityConfig(Base):
    __tablename__ = 'eligibility_config'
    id = Column(Integer, primary_key=True, index=True)
    min_trades = Column(Integer, default=50)
    min_win_rate = Column(Float, default=80.0)
    max_marketplace_price = Column(Float, default=99.99)
    is_active = Column(Boolean, default=True)
    trader_share_percent = Column(
        Float,
        default=70.0,
        nullable=False,
        comment="Percentage of marketplace subscription revenue that goes to the trader (0–100)"
    )

# UPDATED: UploadLimits - Now ties to Trade Points (TP cost per upload)
class UploadLimits(Base):
    __tablename__ = "upload_limits"
    id = Column(Integer, primary_key=True, index=True)
    plan = Column(String, index=True, unique=True, nullable=False)  # e.g., 'starter', 'pro', 'elite'
    monthly_limit = Column(Integer, default=3, nullable=False)  # Number of uploads per month (TP-gated)
    batch_limit   = Column(Integer, default=5, nullable=False) 
    tp_cost = Column(Integer, default=1, nullable=False)  # TP cost per upload (1 default)

# UPDATED: InsightsLimits - Extend for chats; TP-gated
class InsightsLimits(Base):
    __tablename__ = 'insights_limits'

    id = Column(Integer, primary_key=True, index=True)
    plan = Column(String(50), unique=True, nullable=False)  # e.g., 'starter', 'pro', 'elite'
    monthly_limit = Column(Integer, default=0, nullable=False)  # Monthly generation limit
    # NEW: For AI chats (text/voice)
    monthly_chat_limit = Column(Integer, default=5, nullable=False)  # e.g., 5 text chats/mo for free
    voice_chat_limit = Column(Integer, default=3, nullable=False)  # Voice-specific cap
    tp_cost_insight = Column(Integer, default=1, nullable=False)  # TP per insight
    tp_cost_chat = Column(Integer, default=1, nullable=False)  # TP per chat session

# NEW: Referral model - Tracks refs, earnings, tiers
class Referral(Base):
    __tablename__ = "referrals"
    id = Column(Integer, primary_key=True, index=True)
    referrer_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    referee_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)  # The new user
    status = Column(String, default='pending', nullable=False)  # pending, active, churned
    commission_rate = Column(Float, default=0.0, nullable=False)  # DISABLED: Set to 0% (no commissions earned)
    commission_earned = Column(Float, default=0.0, nullable=False)  # Running total $ (will remain 0)
    points_earned = Column(Integer, default=0, nullable=False)  # TP from this ref
    tier_bonus = Column(Float, default=1.0, nullable=False)  # Multiplier from referrer's tier (e.g., 1.2 for Pro Hunter)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # NEW: Optional link to first sub (for marketplace refs)
    subscription_id = Column(Integer, ForeignKey("subscriptions.id"), nullable=True)
    
    # Relationships
    referrer = relationship("User", foreign_keys=[referrer_id], back_populates="referrals")
    referee = relationship("User", foreign_keys=[referee_id])
    subscription = relationship("Subscription", back_populates="referral")
    
    # Constraint: No self-referrals
    __table_args__ = (UniqueConstraint('referrer_id', 'referee_id', name='unique_ref_pair'),)

# NEW: PointTransaction model - Logs all TP movements
class PointTransaction(Base):
    __tablename__ = "point_transactions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    type = Column(String, nullable=False)  # e.g., 'upload', 'insight', 'chat', 'ref_earn', 'ref_redeem', 'payout'
    amount = Column(Integer, nullable=False)  # Positive for earn, negative for spend
    description = Column(String, nullable=True)  # e.g., "Earned 500 TP from ref signup"
    related_ref_id = Column(Integer, ForeignKey("referrals.id"), nullable=True)  # Link to ref if applicable
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="point_transactions")
    referral = relationship("Referral")

class InitialTpConfig(Base):
    __tablename__ = "initial_tp_configs"

    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Integer, default=3, nullable=False)

class UpgradeTpConfig(Base):
    __tablename__ = "upgrade_tp_configs"

    id = Column(Integer, primary_key=True, index=True)
    plan = Column(String, index=True)  # 'pro' or 'elite'
    amount = Column(Integer, default=0, nullable=False)

class AiChatLimits(Base):
    """
    AI Chat limits per plan (monthly limit and TP cost per chat).
    """
    __tablename__ = "ai_chat_limits"

    id = Column(Integer, primary_key=True, index=True)
    plan = Column(String(50), nullable=False, unique=True)  # e.g., 'starter', 'pro', 'elite'
    monthly_limit = Column(Integer, default=5)  # Monthly chat sessions allowed
    tp_cost = Column(Integer, default=1)  # Trade Points cost per chat (0 for free)

class BetaInvite(Base):
    __tablename__ = "beta_invites"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(10), unique=True, index=True, nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    used_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships (optional, for easier querying)
    owner = relationship("User", foreign_keys=[owner_id], back_populates="beta_invites_owned")
    used_by = relationship("User", foreign_keys=[used_by_id])

    __table_args__ = (
        Index('idx_beta_invites_owner', 'owner_id'),  # Explicit for owner_id
        Index('idx_beta_invites_used_by', 'used_by_id'),  # Explicit for used_by_id
    )

# NEW: BetaConfig model for toggling beta mode
class BetaConfig(Base):
    __tablename__ = "beta_configs"

    id = Column(Integer, primary_key=True, index=True)
    is_active = Column(Boolean, default=True, nullable=False)  # Whether beta mode is on
    required_for_signup = Column(Boolean, default=True, nullable=False)  # Require code for signup
    award_points_on_use = Column(Integer, default=3, nullable=False)  # TP awarded to inviter/referrer on successful signup

class BetaReferralTpConfig(Base):
    __tablename__ = 'beta_referral_tp_configs'

    id = Column(Integer, primary_key=True, index=True)
    starter_tp = Column(Integer, default=5, nullable=False)
    pro_tp = Column(Integer, default=20, nullable=False)
    elite_tp = Column(Integer, default=45, nullable=False)

# Tier bonus mapping (can be used in utils)
REFERRAL_TIER_BONUSES = {
    'rookie': 1.0,
    'pro_trader': 1.2,
    'elite_alpha': 1.5,
}
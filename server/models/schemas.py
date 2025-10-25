#models/schemas.py
from pydantic import BaseModel, EmailStr, ConfigDict, Field, model_validator
from typing import Optional, List, Any, Dict
from datetime import datetime


class SignupRequest(BaseModel):
    username: str
    full_name: Optional[str]
    email: EmailStr
    password: str
    referral_code: Optional[str] = None 

class LoginRequest(BaseModel):
    username: str
    password: str
    remember_me: Optional[bool] = False

class GenericResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

class TradeBase(BaseModel):
    symbol: Optional[str]
    trade_date: Optional[str]
    entry_price: Optional[float]
    exit_price: Optional[float]
    pnl: Optional[float]
    notes: Optional[str]

# Fixed: Full TradeResponse mirroring model + frontend (all fields, exact names, no duplicates)
class TradeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    owner_id: int
    symbol: Optional[str] = None
    trade_date: Optional[datetime] = None  # Single instance, datetime
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    sl_price: Optional[float] = None  # Added
    tp_price: Optional[float] = None  # Added
    direction: Optional[str] = None  # SQLEnum serializes as str ('LONG'/'SHORT')
    position_size: Optional[float] = None  # Fixed: 'position_size' not 'size'
    leverage: Optional[float] = None  # Added
    pnl: Optional[float] = None
    notes: Optional[str] = None
    session: Optional[str] = None
    strategy: Optional[str] = None
    fees: Optional[float] = None
    ai_log: Optional[str] = None  # Text -> str (Pydantic handles)
    chart_url: Optional[str] = None
    risk_percentage: Optional[float] = None  # Fixed: 'risk_percentage' not 'risk'
    risk_amount: Optional[float] = None  # Added
    reward_amount: Optional[float] = None  # Added
    r_r_ratio: Optional[float] = None  # Added
    suggestion: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None  # JSON -> dict
    raw_ai_response: Optional[str] = None  # Added (Text -> str)
    confidence: Optional[float] = None
    asset_type: Optional[str] = None  # SQLEnum serializes as str ('FOREX'/'CRYPTO') - Added

    @model_validator(mode='before')
    @classmethod
    def normalize_asset_type(cls, data):
        if isinstance(data, dict) and 'asset_type' in data:
            at = data['asset_type']
            if at and isinstance(at, str):
                # Normalize to uppercase for consistency with enum storage
                data['asset_type'] = at.upper()
        return data

# Added: PaginatedTrades for router (as used in get_trades)
class PaginatedTrades(BaseModel):
    trades: List[TradeResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    win_rate: float = 0.0
    avg_pl: float = 0.0

class SymbolStats(BaseModel):
    symbol: str
    trades: int
    total_pnl: float
    avg_pnl: float

class AIInsights(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    actions: List[str]

class InsightsResponse(BaseModel):
    total_trades: int
    total_pnl: float
    average_pnl: float
    median_pnl: float
    win_rate: float
    best_symbol: Optional[SymbolStats] = None
    worst_symbol: Optional[SymbolStats] = None
    ai_insights: AIInsights

class BaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[dict[str, Any]] = None

class ProfileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    full_name: Optional[str] = None
    email: str
    bio: Optional[str] = None
    trading_style: Optional[str] = None
    goals: Optional[str] = None
    created_at: Optional[datetime] = None
    lifetime_pnl: float
    win_rate: float
    total_trades: int
    best_trade: Optional[TradeResponse] = None
    worst_trade: Optional[TradeResponse] = None
    top_tickers: List[str]

class ProfileUpdateRequest(BaseModel):
    bio: Optional[str] = None
    trading_style: Optional[str] = None
    goals: Optional[str] = None

# Fixed: Full TradeUpdate mirroring model + frontend (all fields, exact names)
class TradeUpdate(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    trade_date: Optional[datetime] = None
    symbol: Optional[str] = None
    asset_type: Optional[str] = None  # Added

    @model_validator(mode='before')
    @classmethod
    def normalize_asset_type(cls, data):
        if isinstance(data, dict) and 'asset_type' in data:
            at = data['asset_type']
            if at and isinstance(at, str):
                # Normalize to uppercase for consistency with enum storage
                data['asset_type'] = at.upper()
        return data

    session: Optional[str] = None
    strategy: Optional[str] = None
    direction: Optional[str] = None  # Added
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    position_size: Optional[float] = None  # Fixed: 'position_size' not 'size'
    risk_percentage: Optional[float] = None  # Fixed: 'risk_percentage' not 'risk'
    sl_price: Optional[float] = None  # Added
    tp_price: Optional[float] = None  # Added
    leverage: Optional[float] = None  # Added
    risk_amount: Optional[float] = None  # Added
    reward_amount: Optional[float] = None  # Added
    r_r_ratio: Optional[float] = None  # Added
    notes: Optional[str] = None
    fees: Optional[float] = None  # Added (if editable)
    suggestion: Optional[str] = None  # Added (if editable)
    chart_url: Optional[str] = None  # Added (if editable)
    tags: Optional[Dict[str, Any]] = None  # Added (JSON)
    raw_ai_response: Optional[str] = None  # Added (if editable)
    confidence: Optional[float] = None  # Added (if editable)
    ai_log: Optional[str] = None  # Added (if editable)
# server/app_utils/upload_utils.py
import io
import os
import re
import json
import base64
import logging
import asyncio
from typing import Optional, Tuple, Dict, Any, List
from enum import Enum
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import uuid

import httpx
from pydantic import BaseModel, Field, field_validator, ConfigDict, FieldValidationInfo
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update, case
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from fastapi import HTTPException

# Local imports
from models import models

# NEW: Import the spend function from app_utils
from app_utils.points import spend_trade_points

from config import get_settings

settings = get_settings()

# Optional dependencies with fallbacks
try:
    from jsonschema import validate as jsonschema_validate, ValidationError as JsonSchemaValidationError
    HAS_JSONSCHEMA = True
except ImportError:
    jsonschema_validate = None
    JsonSchemaValidationError = Exception
    HAS_JSONSCHEMA = False

# OpenAI client
try:
    import openai
    HAS_OPENAI_LIB = True
    openai_client = openai.AsyncOpenAI(
        api_key=getattr(settings, "OPENAI_API_KEY", None),
        timeout=getattr(settings, "OPENAI_TIMEOUT", 60),
        max_retries=getattr(settings, "OPENAI_MAX_RETRIES", 10),
    )
    if not openai_client.api_key:
        raise ValueError("OPENAI_API_KEY missing")
except ImportError:
    openai_client = None
    HAS_OPENAI_LIB = False

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Static directory for trade images
TRADES_DIR = Path("static/trades")
TRADES_DIR.mkdir(parents=True, exist_ok=True)

# Enums
class ExtractionSource(Enum):
    VISION = "openai_vision"
    FAILED = "extraction_failed"

class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class AssetType(Enum):
    FOREX = "FOREX"
    CRYPTO = "CRYPTO"

# Pydantic schema for trade data
class TradeSchema(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "symbol": "EUR/USD",
                "trade_date": "2025-10-22T17:02:00",
                "entry_price": 1.0850,
                "exit_price": None,
                "sl_price": 1.0900,
                "tp_price": 1.0750,
                "direction": "SHORT",
                "position_size": 0.2,
                "leverage": 20.0,
                "pnl": None,
                "notes": "Short on resistance; 50 pips SL; margin: $108.50",
                "session": "London",
                "strategy": "Breakout",
                "risk_percentage": 0.2,
                "risk_amount": 10.0,
                "reward_amount": 20.0,
                "r_r_ratio": 2.0,
                "suggestion": "Solid $10 risk at 1:2 R:R; trail SL after 1:1.",
                "chart_url": "/static/trades/example.jpg",
                "is_trade_confirmation": True,
                "asset_type": "FOREX",
                # New fields example
                "margin_required": 108.50,
                "notional_value": 2170.0,
                "calculator_note": "Auto-calculated: $10 risk → 0.04 BTC @ 10x",
                "auto_calculated": True,
                "calculator_version": "1.1"
            }
        }
    )
    
    symbol: Optional[str] = None
    trade_date: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    direction: Optional[str] = None
    position_size: Optional[float] = None
    leverage: Optional[float] = None
    pnl: Optional[float] = None
    notes: Optional[str] = None
    session: Optional[str] = None
    strategy: Optional[str] = None
    risk_percentage: Optional[float] = None
    risk_amount: Optional[float] = None
    reward_amount: Optional[float] = None
    r_r_ratio: Optional[float] = None
    suggestion: Optional[str] = None
    chart_url: Optional[str] = None
    is_trade_confirmation: bool = False
    asset_type: Optional[str] = None

    # New risk calculator fields
    margin_required: Optional[float] = None
    notional_value: Optional[float] = None
    calculator_note: Optional[str] = None
    auto_calculated: Optional[bool] = None
    calculator_version: Optional[str] = Field(default="1.1", description="Version of the risk calculator used")

    @field_validator("trade_date", mode="before")
    @classmethod
    def validate_date(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError(f"trade_date must be a string, got {type(v)}: {v}")
        if not re.match(r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?", v):
            raise ValueError(f"Invalid ISO date format: {v}")
        return v

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v):
        if v is not None:
            v_upper = v.upper()
            if v_upper not in [member.value for member in TradeDirection]:
                raise ValueError(f"Invalid direction: {v}. Must be 'LONG' or 'SHORT'.")
            return v_upper
        return v

    @field_validator("asset_type")
    @classmethod
    def validate_asset_type(cls, v):
        if v is not None:
            v_upper = v.upper()
            if v_upper not in [member.value for member in AssetType]:
                raise ValueError(f"Invalid asset_type: {v}. Must be 'FOREX' or 'CRYPTO'.")
            return v_upper
        return v

    @field_validator("leverage", mode="before")
    @classmethod
    def validate_leverage(cls, v):
        if v is None:
            return None
        try:
            v_float = float(v)
            if v_float < 1 or v_float > 100:
                logger.warning("Invalid leverage %s; setting to null", v)
                return None
            return v_float
        except (ValueError, TypeError):
            logger.warning("Invalid leverage type %s: %s; setting to null", type(v), v)
            return None

    @field_validator("entry_price", "exit_price", "sl_price", "tp_price", "position_size", "pnl", "risk_percentage", "risk_amount", "reward_amount", "r_r_ratio", "margin_required", "notional_value", mode="before")
    @classmethod
    def validate_numeric(cls, v, info: FieldValidationInfo):
        if v is None:
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid {info.field_name}: must be a number, got {type(v)}: {v}")

# Configuration
OPENAI_MODEL = getattr(settings, "OPENAI_MODEL", "gpt-4o")
MAX_FILE_SIZE_BYTES = getattr(settings, "MAX_UPLOAD_SIZE_BYTES", 10 * 1024 * 1024)
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg"}
ENABLE_FUZZY_CACHE = getattr(settings, "ENABLE_FUZZY_CACHE", False)

if not HAS_OPENAI_LIB:
    raise RuntimeError("openai lib required for AI extraction")

# JSON schema for OpenAI structured output
TRADE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "symbol": {"type": ["string", "null"]},
        "trade_date": {"type": ["string", "null"], "format": "date-time"},
        "entry_price": {"type": ["number", "null"]},
        "exit_price": {"type": ["number", "null"]},
        "sl_price": {"type": ["number", "null"]},
        "tp_price": {"type": ["number", "null"]},
        "direction": {"type": ["string", "null"], "enum": ["LONG", "SHORT"]},
        "position_size": {"type": ["number", "null"]},
        "leverage": {"type": ["number", "null"]},
        "pnl": {"type": ["number", "null"]},
        "notes": {"type": ["string", "null"]},
        "session": {"type": ["string", "null"]},
        "strategy": {"type": ["string", "null"]},
        "risk_percentage": {"type": ["number", "null"]},
        "risk_amount": {"type": ["number", "null"]},
        "reward_amount": {"type": ["number", "null"]},
        "r_r_ratio": {"type": ["number", "null"]},
        "suggestion": {"type": ["string", "null"]},
        "is_trade_confirmation": {"type": "boolean"},
        "asset_type": {"type": ["string", "null"], "enum": ["FOREX", "CRYPTO"]},
        # New fields for risk calculator
        "margin_required": {"type": ["number", "null"]},
        "notional_value": {"type": ["number", "null"]},
        "calculator_note": {"type": ["string", "null"]},
        "auto_calculated": {"type": ["boolean", "null"]},
        "calculator_version": {"type": ["string", "null"]}
    },
    # FIXED: Reduced required fields to only the essential boolean; allows partial extractions with nulls
    "required": ["is_trade_confirmation"],
    "additionalProperties": False,
}

# System prompt - FIXED: Added emphasis on partial extractions and using nulls
SYSTEM_PROMPT = (
    "You are a 20-year veteran Wall Street forex trader and crypto perps specialist. Extract every detail dynamically from screenshots—no fixed values. "
    "Audit like a compliance officer: Focus on personal trades or annotated charts only (look for entry/exit arrows, SL/TP horizontal lines, P&L labels, risk markers). "
    "Set is_trade_confirmation=true only for verifiable trades with clear prices/dates; false for ambiguous charts or partial info. "
    "Be precise and extract exact numbers from visual labels, axes, and annotations: "
    "- Symbol: From chart title/header (e.g., EUR/USD=forex, BTC/USDT=crypto—set asset_type accordingly; infer from 'lots/pips' for forex, 'perp/qty' for crypto). Use null if unclear. "
    "- Trade_date: ISO format from timestamp axis or order ticket (e.g., 2025-10-22T17:02:00; use current date if unclear). Use null if no date visible. "
    "- Prices: Raw decimals from y-axis/labels (entry: green/up arrow for LONG, red/down arrow for SHORT; exit: close label; SL: red line below entry for LONG or above for SHORT; TP: green line above entry for LONG or below for SHORT—preserve 4-5 decimal places). Use null if no clear lines/labels. "
    "- Direction: Prioritize visual cues first: 'LONG' if buy/green/upward arrow, entry/SL/TP relation shows TP > entry > SL; 'SHORT' if sell/red/downward arrow, resistance labels, or SL > entry > TP. Use null if ambiguous. "
      "For shorts at resistance: Look for red sell signals, downward arrows at horizontal resistance bars/lines, bearish candles at highs, TP clearly below entry. Double-check: if entry is at a high/resistance with downward bias, favor SHORT even if prices ambiguous. "
      "Infer from candle colors/arrows if unlabeled, but cross-verify with SL/TP positions—do not rely solely on one. Use null if uncertain. "
    "- Position_size: Lots (forex, e.g., 0.01 from ticket) or qty/units (crypto, e.g., 0.1 BTC); null if missing. "
    "- Leverage: 'x' value from margin labels (e.g., 20x); null if absent. "
    "- SL/TP: Exact prices from dashed/solid lines or labels; append pip distances to notes for forex (e.g., 'SL: 1.0900 (50 pips below entry)'). Use null if no lines visible. "
    "- Risk: risk_amount as $ loss to SL (e.g., from 'Risk: $10' or calc if position/SL shown); risk_percentage if % account labeled. Use null if unclear. Leave position_size null if risk shown but size missing (compute later). "
    "- Reward_amount: $ target to TP from labels. Use null if none. "
    "- R:R: From labels or price distances (reward/risk); null if <1:1 or impossible to compute, flag poor setups in notes. "
    "- Session: Infer from x-axis time (UTC-based: London 8-17, NY 13-22, Tokyo 0-9, Sydney 22-7 UTC); use 'UTC' if unclear or 24/7 crypto. Use null if no time axis. "
    "- Strategy: From annotations (e.g., 'Pinbar reversal at S/R', 'Breakout above EMA'); infer from patterns (e.g., head&shoulders=Reversal). Use null or 'Unknown' for resistance shorts if unclear: note 'Short at resistance' or similar. "
    "- Notes: All visible text/details (P&L, margin, pips, timeframe, indicators like RSI/MACD values); include extraction confidence (e.g., 'High conf on prices, low on strategy'). Flag if direction inference uncertain. Always explain any nulls or ambiguities here. "
    "- Suggestion: Pro risk management insight (e.g., 'Solid 1:2 R:R on $10 risk; trail SL to BE after 1:1; avoid >50x lev on crypto'). Target R:R >=1:2, safe lev (<=20x forex, <=10x crypto). Use null if no data for suggestion. "
    "Handle common platforms: TradingView (clean labels), MT4/5 (ticket panels), Thinkorswim (colored zones). "
    "Null for absent/unclear (e.g., no exit=unrealized trade); prioritize: symbol > direction/entry/SL/TP > date/session. "
    "If chart quality low or no clear trade, set is_trade_confirmation=false and explain in notes why extraction is partial. "
    "If unable to extract all fields, use null and detail limitations in notes—do not omit keys. "
    "Always output complete JSON with nulls for missing/unclear; be accurate—double-check numbers from visuals. Even partial extractions are valuable for manual review."
)

# In-memory symbol cache
_symbol_cache: Optional[Dict[str, list]] = None

def normalize_plan(plan: str) -> str:
    """Normalize composite plan_type (e.g., 'pro_monthly') to base plan (e.g., 'pro') for feature granting."""
    if '_' in plan:
        return plan.split('_')[0]
    return plan

def is_valid_trade(trade_dict: Dict[str, Any]) -> bool:
    """Check if trade has at least 2 key fields populated."""
    key_fields = ['symbol', 'entry_price', 'sl_price', 'tp_price', 'direction']
    return sum(1 for f in key_fields if trade_dict.get(f) is not None) >= 2

def validate_and_infer_direction(parsed: Dict[str, Any]) -> str:
    """Post-extraction validation and inference for direction based on price relations and visual cues in notes."""
    direction = parsed.get("direction", "").upper()
    entry = parsed.get("entry_price")
    sl = parsed.get("sl_price")
    tp = parsed.get("tp_price")
    notes_lower = (parsed.get("notes", "") or "").lower()

    if direction in ["LONG", "SHORT"] and (entry is None or sl is None or tp is None):
        return direction

    if entry is not None and sl is not None and tp is not None:
        if tp > entry > sl:
            return "LONG"
        elif sl > entry > tp:
            return "SHORT"

    elif entry is not None and tp is not None:
        if entry < tp:
            return "LONG"
        else:
            return "SHORT"

    if "short" in notes_lower or "resistance" in notes_lower or "sell" in notes_lower:
        return "SHORT"
    elif "long" in notes_lower or "buy" in notes_lower or "support" in notes_lower:
        return "LONG"

    return direction if direction in ["LONG", "SHORT"] else "LONG"

async def get_cached_symbols(db: AsyncSession) -> list:
    global _symbol_cache
    if _symbol_cache is not None and not ENABLE_FUZZY_CACHE:
        return list(_symbol_cache.get("symbols", []))
    try:
        # FIXED: Use subquery with ROW_NUMBER to get latest symbol per unique symbol
        subq = (
            select(
                models.Trade.symbol,
                models.Trade.created_at,
                func.row_number().over(
                    partition_by=models.Trade.symbol,
                    order_by=models.Trade.created_at.desc()
                ).label("rn")
            )
            .subquery()
        )
        q = await db.execute(
            select(subq.c.symbol)
            .where(subq.c.rn == 1)
            .order_by(subq.c.created_at.desc())
            .limit(500)
        )
        symbols = [r[0] for r in q.fetchall() if r[0]]
        if not ENABLE_FUZZY_CACHE:
            _symbol_cache = {"symbols": symbols}
        return symbols
    except Exception as e:
        logger.error("Failed to fetch symbols: %s", e)
        return []

async def get_monthly_upload_count(db: AsyncSession, user_id: int) -> int:
    now = datetime.utcnow()
    start_of_month = now.replace(day=1)
    try:
        count = await db.execute(
            select(func.count(models.Trade.id))
            .where(
                models.Trade.owner_id == user_id,
                models.Trade.created_at >= start_of_month
            )
        )
        return count.scalar() or 0
    except Exception as e:
        logger.error("Failed to count monthly uploads: %s", e)
        return 0

async def get_plan_limits(db: AsyncSession, plan: str) -> dict:
    lowered_plan = plan.lower()
    result = await db.execute(
        select(models.UploadLimits).where(models.UploadLimits.plan == lowered_plan)
    )
    limits = result.scalar_one_or_none()

    if limits:
        return {
            "monthly_upload_limit": limits.monthly_limit,
            "batch_upload_limit": limits.batch_limit,
            "tp_cost_upload": limits.tp_cost,
        }

    # FIXED: Corrected defaults to match model (monthly=3, batch=5 for starter)
    defaults = {
        "starter": {"monthly_upload_limit": 3, "batch_upload_limit": 5, "tp_cost_upload": 1},
        "pro": {"monthly_upload_limit": 20, "batch_upload_limit": 10, "tp_cost_upload": 0},
        "elite": {"monthly_upload_limit": 50, "batch_upload_limit": 20, "tp_cost_upload": 0},
    }
    return defaults.get(lowered_plan, defaults["starter"])

# REMOVED: spend_trade_points function (now imported from app_utils.points)

async def enforce_upload_limits(db: AsyncSession, current_user: models.User, num_files: int, action_type: str = "upload") -> None:
    normalized_plan = normalize_plan(current_user.plan)
    limits = await get_plan_limits(db, normalized_plan)
    
    monthly_count = await get_monthly_upload_count(db, current_user.id)
    monthly_limit = limits["monthly_upload_limit"]
    if action_type == "upload" and num_files > limits["batch_upload_limit"]:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size exceeds plan limit ({limits['batch_upload_limit']} max). Upgrade for more."
        )
    projected_count = monthly_count + num_files
    if projected_count > monthly_limit:
        remaining = max(0, monthly_limit - monthly_count)
        raise HTTPException(
            status_code=429,
            detail=f"Monthly upload limit reached ({monthly_limit} total). {remaining} remaining this month. Upgrade for more."
        )
    
    tp_cost = limits.get(f"tp_cost_{action_type}", 0) * num_files if action_type in ["upload", "insight"] else 0
    if tp_cost > 0:
        # CHANGED: Call the imported async function for points spending (no redis here, so None)
        await spend_trade_points(db, current_user, action_type, tp_cost)
        await db.refresh(current_user)

def validate_and_preprocess_image(contents: bytes, max_pixels: int = 1_000_000) -> bytes:
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    try:
        img = Image.open(io.BytesIO(contents))
        content_type = img.format.lower() if img.format else "unknown"
        if f"image/{content_type}" not in ALLOWED_IMAGE_TYPES:
            raise ValueError("Unsupported image type")
        w, h = img.size
        if w * h > max_pixels:
            ratio = (max_pixels / (w * h)) ** 0.5
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=60, optimize=True)
        compressed_bytes = buffer.getvalue()
        logger.debug("Image compressed: %d -> %d bytes", len(contents), len(compressed_bytes))
        return compressed_bytes
    except Exception as e:
        logger.error("Image preprocessing failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

def save_image_locally(image_bytes: bytes, filename: str) -> str:
    file_path = TRADES_DIR / filename
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    return f"/static/trades/{filename}"

def normalize_symbol(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = re.sub(r"[^\w/]", "", str(raw).strip().upper())
    s = re.sub(r"([A-Z0-9]+)([A-Z0-9]{3,})", r"\1/\2", s)
    s = re.sub(r"(LIVE|V\d+|YG|TEST|DEMO)$", "", s, flags=re.IGNORECASE)
    s = s.strip("/ ")
    return s if s else None

def infer_asset_type(symbol: Optional[str], notes: Optional[str] = None) -> Optional[str]:
    if not symbol:
        return None
    symbol_upper = symbol.upper()
    forex_pairs = ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
    if any(pair in symbol_upper for pair in forex_pairs) and '/' in symbol_upper:
        return "FOREX"
    if any(crypto in symbol_upper for crypto in ['BTC', 'ETH', 'USDT', 'USDC']):
        return "CRYPTO"
    if notes:
        notes_lower = notes.lower()
        if 'lot' in notes_lower or 'pip' in notes_lower:
            return "FOREX"
        if 'perp' in notes_lower or 'contract' in notes_lower:
            return "CRYPTO"
    return None

async def fuzzy_match_symbol(db: AsyncSession, candidate: str, symbols: list) -> str:
    if not candidate or not symbols:
        return candidate
    from difflib import get_close_matches
    matches = get_close_matches(candidate, symbols, n=1, cutoff=0.8)
    return matches[0] if matches else candidate

def get_pip_value(asset_type: str, symbol: str) -> float:
    if asset_type == "FOREX":
        return 10.0
    elif asset_type == "CRYPTO":
        return 1.0
    return 1.0

# ──────────────────────────────────────────────────────────────
# CRYPTO FUTURES RISK CALCULATOR (USDT-margined perpetuals)
# ──────────────────────────────────────────────────────────────
def calculate_crypto_futures_risk(
    balance: float,
    risk_percent: float | None,
    risk_amount: float | None,
    entry_price: float,
    sl_price: float,
    leverage: float | None,
) -> dict:
    """
    Core calculator from PDR Section 3.
    Returns enriched fields for TradeSchema.
    """
    if not entry_price or not sl_price:
        return {}

    price_diff = abs(entry_price - sl_price)
    if price_diff <= 0:
        return {"notes": "Invalid: SL cannot be at or beyond entry"}

    # Priority: use risk_amount > risk_percent > default 1%
    if risk_amount:
        risk_usd = risk_amount
    elif risk_percent:
        risk_usd = balance * (risk_percent / 100)
    else:
        risk_usd = balance * 0.01  # 1% default
        risk_percent = 1.0

    # Position size in coins
    position_size = risk_usd / price_diff
    notional = entry_price * position_size
    margin = notional / (leverage or 10)  # default 10x

    # Verify risk %
    verified_risk_pct = (price_diff * position_size / balance) * 100

    return {
        "position_size": round(position_size, 6),
        "risk_amount": round(risk_usd, 2),
        "risk_percentage": round(verified_risk_pct, 3),
        "margin_required": round(margin, 2),
        "notional_value": round(notional, 2),
        "leverage_used": leverage or 10,
        "calculator_note": f"Auto-calculated: ${risk_usd} risk → {position_size:.6f} coins @ {leverage or 10}x"
    }

def compute_trade_metrics(parsed: Dict[str, Any], account_balance: float = 10000.0) -> Dict[str, Any]:
    """
    Enhanced trade metrics engine:
    - Fixes direction inference
    - Auto-calculates position size, risk, margin for CRYPTO futures using calculate_crypto_futures_risk
    - Preserves full FOREX logic
    - Adds R:R, notes, suggestions
    - Uses provided account_balance
    """
    # ──────────────────────────────────────────────────────────────
    # 1. DIRECTION INFERENCE & CORRECTION
    # ──────────────────────────────────────────────────────────────
    original_direction = parsed.get("direction", "").upper()
    parsed["direction"] = validate_and_infer_direction(parsed)
    if parsed["direction"] != original_direction:
        parsed["notes"] = (
            parsed.get("notes", "") + f" (Direction auto-corrected to {parsed['direction']})"
        ).strip()

    # ──────────────────────────────────────────────────────────────
    # 2. CORE DATA
    # ──────────────────────────────────────────────────────────────
    entry = parsed.get("entry_price")
    sl = parsed.get("sl_price")
    tp = parsed.get("tp_price")
    direction = parsed["direction"] or "LONG"
    leverage = parsed.get("leverage")
    risk_amount = parsed.get("risk_amount")
    risk_percent = parsed.get("risk_percentage")
    position_size = parsed.get("position_size")

    # Asset type inference
    asset_type = parsed.get("asset_type") or infer_asset_type(
        parsed.get("symbol"), parsed.get("notes")
    )
    parsed["asset_type"] = asset_type

    # Session fallback
    if parsed.get("session") in [None, "N/A", ""]:
        parsed["session"] = "UTC"

    # ──────────────────────────────────────────────────────────────
    # 3. CRYPTO FUTURES RISK CALCULATOR (PDR Section 3)
    # ──────────────────────────────────────────────────────────────
    if asset_type == "CRYPTO" and entry and sl and entry != sl:
        calc = calculate_crypto_futures_risk(
            balance=account_balance,
            risk_percent=risk_percent,
            risk_amount=risk_amount,
            entry_price=entry,
            sl_price=sl,
            leverage=leverage,
        )
        if calc:
            # Apply only if not already set
            if not position_size:
                parsed["position_size"] = calc["position_size"]
            if not risk_amount:
                parsed["risk_amount"] = calc["risk_amount"]
            parsed["risk_percentage"] = calc["risk_percentage"]
            parsed["margin_required"] = calc["margin_required"]
            parsed["notional_value"] = calc["notional_value"]
            parsed["calculator_note"] = calc["calculator_note"]
            parsed["auto_calculated"] = True
            parsed["calculator_version"] = "1.1"

            # Rich note
            parsed["notes"] = (
                parsed.get("notes", "") + "; " + calc["calculator_note"]
            ).strip()

            # R:R Ratio
            if tp:
                reward_dist = abs(tp - entry)
                risk_dist = abs(entry - sl)
                if risk_dist > 0:
                    rr = round(reward_dist / risk_dist, 2)
                    parsed["r_r_ratio"] = rr
                    if rr >= 2:
                        parsed["suggestion"] = (
                            f"Excellent {rr}:1 R:R — strong setup. Trail SL after 1:1."
                        )
                    elif rr >= 1.5:
                        parsed["suggestion"] = (
                            f"Good {rr}:1 R:R — consider partial profit at 1:1."
                        )
                    else:
                        parsed["suggestion"] = (
                            f"Tight {rr}:1 R:R — move SL to breakeven quickly."
                        )

    # ──────────────────────────────────────────────────────────────
    # 4. FOREX LOGIC (Your original pip-based system)
    # ──────────────────────────────────────────────────────────────
    elif asset_type == "FOREX" and entry and sl:
        risk_dist = entry - sl if direction == "LONG" else sl - entry
        if risk_dist <= 0:
            parsed["notes"] = (parsed.get("notes", "") + " Invalid SL distance").strip()
        else:
            pip_dist = abs(risk_dist) * 10000
            parsed["notes"] = (
                parsed.get("notes", "") + f"; SL distance: {pip_dist:.1f} pips"
            ).strip()

            if tp:
                reward_dist = abs(tp - entry)
                if risk_dist > 0:
                    rr = round(reward_dist / risk_dist, 2)
                    parsed["r_r_ratio"] = rr

            # Reuse your existing risk_amount → position_size logic
            if risk_amount and not position_size and risk_dist != 0:
                position_size = risk_amount / (abs(risk_dist) * 10000 * 10)
                parsed["position_size"] = round(position_size, 4)
                parsed["notes"] = (
                    parsed.get("notes", "") + f" (Position sized for ${risk_amount:.2f} risk)"
                ).strip()

    # ──────────────────────────────────────────────────────────────
    # 5. FINAL TOUCHES
    # ──────────────────────────────────────────────────────────────
    if leverage and leverage > 1:
        lev_note = f" Leverage: {leverage}x"
        if asset_type == "CRYPTO" and leverage > 20:
            lev_note += " — high risk, consider reducing"
        elif asset_type == "FOREX" and leverage > 50:
            lev_note += " — very aggressive"
        parsed["notes"] = (parsed.get("notes", "") + lev_note).strip()

    # Default suggestion if none
    if not parsed.get("suggestion"):
        parsed["suggestion"] = "Review SL/TP placement and risk size before execution."

    return parsed

def compute_confidence(parsed: Dict[str, Any], source: ExtractionSource) -> Dict[str, float]:
    weights = {ExtractionSource.VISION: 0.95, ExtractionSource.FAILED: 0.0}
    base = weights.get(source, 0.0)
    conf = {}
    fields = [
        "symbol", "trade_date", "entry_price", "exit_price", "sl_price", "tp_price", 
        "direction", "position_size", "leverage", "pnl", "notes", "session", 
        "strategy", "risk_percentage", "risk_amount", "reward_amount", "r_r_ratio", 
        "suggestion", "asset_type", "is_trade_confirmation", "margin_required", 
        "notional_value", "auto_calculated", "calculator_version"
    ]
    for field in fields:
        conf[field] = round(base if parsed.get(field) is not None or field == "is_trade_confirmation" else 0.0, 2)
    conf["overall"] = round(sum(conf.values()) / len(conf), 2)
    return conf

async def _call_openai_with_lib(messages: list, response_format: Dict[str, Any], model: str = OPENAI_MODEL) -> Dict[str, Any]:
    try:
        raw_response = await openai_client.chat.completions.with_raw_response.create(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=0.0,
            max_tokens=1024,  # FIXED: Increased from 512 for more detailed notes/explanations
        )
        remaining_tokens = int(raw_response.headers.get('x-ratelimit-remaining-tokens', 0)) if raw_response.headers else 0
        if remaining_tokens < 50000:
            logger.warning("Low tokens remaining: %d; consider tier bump", remaining_tokens)
        return raw_response.parse().model_dump()
    except openai.RateLimitError as e:
        logger.warning("OpenAI rate limit hit: %s. Extended backoff...", e)
        await asyncio.sleep(120)
        raise
    except Exception as e:
        logger.warning("OpenAI call failed: %s", e)
        raise

@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential_jitter(initial=2, exp_base=2, max=120, jitter=1),
    retry=retry_if_exception_type(httpx.HTTPStatusError)
)
async def _call_openai_fallback(payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_keepalive_connections=5)) as client:
        resp = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("retry-after", 30))
            logger.warning("429 detected; waiting %d seconds", retry_after)
            await asyncio.sleep(retry_after)
        resp.raise_for_status()
        data = resp.json()
        # FIXED: Check for error in response body even on 200
        if isinstance(data, dict) and "error" in data:
            msg = data["error"].get("message", str(data["error"]))
            raise HTTPException(status_code=resp.status_code, detail=f"OpenAI API error: {msg}")
        return data

async def call_openai_vision(image_bytes: bytes, max_attempts: int = 3, account_balance: float = 10000.0, redis_client: Optional[Any] = None) -> Tuple[Optional[TradeSchema], Dict[str, Any]]:
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
            ]
        }
    ]
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "trade_extraction",
            # FIXED: Set strict=False to allow more flexible partial outputs
            "strict": False,
            "schema": TRADE_JSON_SCHEMA
        }
    }
    
    for attempt in range(max_attempts):
        try:
            if HAS_OPENAI_LIB:
                raw = await _call_openai_with_lib(messages, response_format)
            else:
                payload = {
                    "model": OPENAI_MODEL,
                    "messages": messages,
                    "response_format": response_format,
                    "temperature": 0.0,
                    "max_tokens": 1024,  # FIXED: Increased for better handling of notes
                }
                raw = await _call_openai_fallback(payload)
            
            # FIXED: Validate structure and JSON content
            if "choices" not in raw or not raw["choices"]:
                raise ValueError("Invalid OpenAI response structure: no choices")
            
            content = raw["choices"][0]["message"]["content"]
            if not content:
                raise ValueError("Empty content from OpenAI")
            
            # FIXED: Robust JSON parsing with error handling
            try:
                parsed_dict = json.loads(content)
            except json.JSONDecodeError as json_err:
                logger.error("OpenAI returned invalid JSON on attempt %d: %s | Content preview: %s", attempt + 1, json_err, content[:500])
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    raise HTTPException(status_code=500, detail="AI returned invalid JSON response after retries")
            
            parsed = TradeSchema(**parsed_dict) if parsed_dict else None
            if parsed:
                computed_dict = compute_trade_metrics(parsed.dict(), account_balance=account_balance)
                parsed = TradeSchema(**computed_dict)
                logger.info("Vision extraction succeeded on attempt %d", attempt + 1)
                return parsed, raw
        except openai.RateLimitError as e:
            error_detail = str(e).split("'")[1] if "'" in str(e) else str(e)  # Extract clean error msg (e.g., "Rate limit reached")
            logger.warning("OpenAI RateLimitError on attempt %d: %s", attempt + 1, error_detail)
            if redis_client:  # NEW: Set user-specific backoff in Redis
                user_key = f"rate_limit_backoff:{account_balance}"  # Use account_balance as user proxy; replace with user_id if avail
                await redis_client.set(user_key, json.dumps({"error": error_detail, "retry_after": 300}), ex=600)  # 5min backoff hint, expire 10min
            if attempt < max_attempts - 1:
                sleep_time = 2 ** attempt + (60 if "rate" in error_detail.lower() else 0)  # Extra 60s on rate limits
                await asyncio.sleep(sleep_time)
            else:
                raise HTTPException(503, f"OpenAI rate limit exceeded: {error_detail}. Wait 5-10min or check usage at https://platform.openai.com/usage")
        except HTTPException:
            raise  # Re-raise HTTPExceptions directly
        except Exception as e:
            logger.warning("Vision attempt %d failed: %s", attempt + 1, e)
            if attempt < max_attempts - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                raise HTTPException(503, f"AI extraction failed: {str(e)[:100]}... (check logs for details)")
    
    logger.error("Vision extraction failed after %d attempts", max_attempts)
    return None, {"error": f"Max attempts ({max_attempts}) exceeded after backoffs"}
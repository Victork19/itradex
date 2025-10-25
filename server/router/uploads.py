# router/uploads.py

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
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Body
from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict, FieldValidationInfo
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select, func
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

# Local imports (adjust paths as needed)
from models import models, schemas
import auth
from database import get_session
from config import settings

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
    logger = logging.getLogger(__name__)
    logger.warning("openai lib not available; install 'openai' for AI extraction")

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["Uploads"])

# Static directory for trade images
TRADES_DIR = Path("static/trades")
TRADES_DIR.mkdir(parents=True, exist_ok=True)

# Enums for clarity and type safety
class ExtractionSource(Enum):
    VISION = "openai_vision"
    FAILED = "extraction_failed"

class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class AssetType(Enum):
    FOREX = "FOREX"
    CRYPTO = "CRYPTO"

# Pydantic schema for trade data - Enhanced validation
class TradeSchema(BaseModel):
    model_config = ConfigDict(
        extra="ignore",  # FIXED: Changed from "forbid" to "ignore" to allow extra fields like _is_partial from frontend/AI
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
                "asset_type": "FOREX"
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

    @field_validator("entry_price", "exit_price", "sl_price", "tp_price", "position_size", "pnl", "risk_percentage", "risk_amount", "reward_amount", "r_r_ratio", mode="before")
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
        "asset_type": {"type": ["string", "null"], "enum": ["FOREX", "CRYPTO"]}
    },
    "required": ["symbol", "trade_date", "entry_price", "exit_price", "sl_price", "tp_price", "direction", "position_size", "leverage", "pnl", "notes", "session", "strategy", "risk_percentage", "risk_amount", "reward_amount", "r_r_ratio", "suggestion", "is_trade_confirmation", "asset_type"],
    "additionalProperties": False,
}

# Enhanced Pro trader system prompt
SYSTEM_PROMPT = (
    "You are a 20-year veteran Wall Street forex trader and crypto perps specialist. Extract every detail dynamically from screenshots—no fixed values. "
    "Audit like a compliance officer: Focus on personal trades or annotated charts only (look for entry/exit arrows, SL/TP horizontal lines, P&L labels, risk markers). "
    "Set is_trade_confirmation=true only for verifiable trades with clear prices/dates; false for ambiguous charts. "
    "Be precise and extract exact numbers from visual labels, axes, and annotations: "
    "- Symbol: From chart title/header (e.g., EUR/USD=forex, BTC/USDT=crypto—set asset_type accordingly; infer from 'lots/pips' for forex, 'perp/qty' for crypto). "
    "- Trade_date: ISO format from timestamp axis or order ticket (e.g., 2025-10-22T17:02:00; use current date if unclear). "
    "- Prices: Raw decimals from y-axis/labels (entry: green/up arrow; exit: close label; SL: red/below line; TP: green/above line—preserve 4-5 decimal places). "
    "- Direction: 'LONG' for buy/green/upward moves (entry < TP); 'SHORT' for sell/red/downward (entry > TP). Infer from candle colors/arrows if unlabeled. "
    "- Position_size: Lots (forex, e.g., 0.01 from ticket) or qty/units (crypto, e.g., 0.1 BTC); null if missing. "
    "- Leverage: 'x' value from margin labels (e.g., 20x); null if absent. "
    "- SL/TP: Exact prices from dashed/solid lines or labels; append pip distances to notes for forex (e.g., 'SL: 1.0900 (50 pips below entry)'). "
    "- Risk: risk_amount as $ loss to SL (e.g., from 'Risk: $10' or calc if position/SL shown); risk_percentage if % account labeled. Leave position_size null if risk shown but size missing (compute later). "
    "- Reward_amount: $ target to TP from labels. "
    "- R:R: From labels or price distances (reward/risk); null if <1:1, flag poor setups. "
    "- Session: Infer from x-axis time (UTC-based: London 8-17, NY 13-22, Tokyo 0-9, Sydney 22-7 UTC); use 'UTC' if unclear or 24/7 crypto. "
    "- Strategy: From annotations (e.g., 'Pinbar reversal at S/R', 'Breakout above EMA'); infer from patterns (e.g., head&shoulders=Reversal). "
    "- Notes: All visible text/details (P&L, margin, pips, timeframe, indicators like RSI/MACD values); include extraction confidence (e.g., 'High conf on prices, low on strategy'). "
    "- Suggestion: Pro risk management insight (e.g., 'Solid 1:2 R:R on $10 risk; trail SL to BE after 1:1; avoid >50x lev on crypto'). Target R:R >=1:2, safe lev (<=20x forex, <=10x crypto). "
    "Handle common platforms: TradingView (clean labels), MT4/5 (ticket panels), Thinkorswim (colored zones). "
    "Null for absent/unclear (e.g., no exit=unrealized trade); prioritize: symbol > direction/entry/SL/TP > date/session. "
    "If chart quality low or no clear trade, set is_trade_confirmation=false and explain in notes. "
    "Always output complete JSON with nulls for missing; be accurate—double-check numbers from visuals."
)

# In-memory symbol cache
_symbol_cache: Optional[Dict[str, list]] = None

def is_valid_trade(trade_dict: Dict[str, Any]) -> bool:
    """Check if trade has at least 2 key fields populated."""
    key_fields = ['symbol', 'entry_price', 'sl_price', 'tp_price', 'direction']
    return sum(1 for f in key_fields if trade_dict.get(f) is not None) >= 2

async def get_cached_symbols(db: AsyncSession) -> list:
    """Fetch recent symbols for fuzzy matching."""
    global _symbol_cache
    if _symbol_cache is not None and not ENABLE_FUZZY_CACHE:
        return list(_symbol_cache.get("symbols", []))
    try:
        q = await db.execute(
            select(models.Trade.symbol)
            .distinct()
            .order_by(models.Trade.created_at.desc())
            .limit(500)
        )
        symbols = [r[0] for r in q.fetchall() if r[0]]
        if ENABLE_FUZZY_CACHE:
            pass  # TODO: Redis integration
        else:
            _symbol_cache = {"symbols": symbols}
        return symbols
    except Exception as e:
        logger.error("Failed to fetch symbols: %s", e)
        return []

async def get_monthly_upload_count(db: AsyncSession, user_id: int) -> int:
    """Count trades uploaded this month for the user."""
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

def get_plan_limits(plan: str) -> Tuple[int, int]:
    """Get monthly and batch limits based on plan."""
    if plan in ['free', 'starter']:
        return 3, 3
    elif plan in ['premium', 'pro', 'elite']:
        return float('inf'), 10
    else:
        return 3, 3

async def enforce_upload_limits(db: AsyncSession, current_user: models.User, num_files: int) -> None:
    """Enforce plan-based upload limits."""
    monthly_limit, batch_limit = get_plan_limits(current_user.plan)
    monthly_count = await get_monthly_upload_count(db, current_user.id)
    if num_files > batch_limit:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size exceeds plan limit ({batch_limit} max). Upgrade for more."
        )
    projected_count = monthly_count + num_files
    if monthly_limit != float('inf') and projected_count > monthly_limit:
        remaining = max(0, monthly_limit - monthly_count)
        raise HTTPException(
            status_code=402,
            detail=f"Monthly upload limit reached ({monthly_limit} total). {remaining} remaining this month. Upgrade for unlimited."
        )

def validate_and_preprocess_image(contents: bytes, max_pixels: int = 1_000_000) -> bytes:
    """Validate and compress image for token efficiency."""
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
    """Save image to static/trades and return URL."""
    file_path = TRADES_DIR / filename
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    return f"/static/trades/{filename}"

def normalize_symbol(raw: Optional[str]) -> Optional[str]:
    """Clean symbol: Uppercase, standardize separators."""
    if not raw:
        return None
    s = re.sub(r"[^\w/]", "", str(raw).strip().upper())
    s = re.sub(r"([A-Z0-9]+)([A-Z0-9]{3,})", r"\1/\2", s)
    s = re.sub(r"(LIVE|V\d+|YG|TEST|DEMO)$", "", s, flags=re.IGNORECASE)
    s = s.strip("/ ")
    return s if s else None

def infer_asset_type(symbol: Optional[str], notes: Optional[str] = None) -> Optional[str]:
    """Infer asset_type from symbol or notes."""
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
    """Fuzzy match symbol against known symbols."""
    if not candidate or not symbols:
        return candidate
    from difflib import get_close_matches
    matches = get_close_matches(candidate, symbols, n=1, cutoff=0.8)
    return matches[0] if matches else candidate

def get_pip_value(asset_type: str, symbol: str) -> float:
    """Get pip value: $10/lot for forex majors, $1/unit for crypto."""
    if asset_type == "FOREX":
        return 10.0
    elif asset_type == "CRYPTO":
        return 1.0
    return 1.0

def compute_trade_metrics(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Compute metrics: back-calc position_size if needed, add pip distance for forex."""
    entry = parsed.get("entry_price")
    sl = parsed.get("sl_price")
    tp = parsed.get("tp_price")
    direction = parsed.get("direction") or "LONG"
    leverage = parsed.get("leverage")
    risk_amount = parsed.get("risk_amount")
    position_size = parsed.get("position_size")
    
    asset_type = parsed.get("asset_type") or infer_asset_type(parsed.get("symbol"), parsed.get("notes"))
    parsed["asset_type"] = asset_type
    if parsed.get("session") is None or parsed.get("session") == "N/A":
        parsed["session"] = "UTC"
    if not all([entry is not None, sl is not None]):
        return parsed
    
    risk_dist = entry - sl if direction == "LONG" else sl - entry
    reward_dist = (tp - entry if direction == "LONG" else entry - tp) if tp is not None else 0
    
    if risk_dist <= 0:
        return parsed
    
    if parsed.get("r_r_ratio") is None and reward_dist > 0:
        parsed["r_r_ratio"] = round(reward_dist / risk_dist, 2)
    
    if risk_amount is not None and position_size is None and risk_dist != 0:
        pip_value = get_pip_value(asset_type, parsed.get("symbol", ""))
        position_size = risk_amount / (abs(risk_dist) * pip_value)
        parsed["position_size"] = round(position_size, 4)
        parsed["notes"] = parsed.get("notes", "") + f" (Position sized for ${risk_amount:.2f} risk)"
    
    if parsed.get("reward_amount") is None and reward_dist > 0 and position_size is not None:
        pip_value = get_pip_value(asset_type, parsed.get("symbol", ""))
        parsed["reward_amount"] = round(position_size * reward_dist * pip_value, 2)
    
    if leverage and leverage > 1:
        parsed["notes"] = parsed.get("notes", "") + f"; Leverage: {leverage}x"
    
    if asset_type == "FOREX" and risk_dist != 0:
        pip_dist = abs(risk_dist) * 10000
        parsed["notes"] = parsed.get("notes", "") + f"; SL distance: {pip_dist:.1f} pips"
    
    if risk_amount:
        rr = parsed.get("r_r_ratio") or 1.0
        base_sugg = parsed.get("suggestion", "")
        risk_tip = f"Risk ${risk_amount:.2f} {'solid' if risk_amount <= 50 else 'high—scale down'}"
        rr_tip = f"; R:R {rr}: {'excellent (>2)' if rr >= 2 else 'improve (>1.5)' if rr >= 1.5 else 'tighten'}"
        leverage_tip = f"; Leverage {leverage}x {'safe' if leverage <= 20 else 'risky—reduce'}" if leverage and leverage > 1 else ""
        parsed["suggestion"] = f"{base_sugg} {risk_tip}{rr_tip}{leverage_tip}."
    
    return parsed

def compute_confidence(parsed: Dict[str, Any], source: ExtractionSource) -> Dict[str, float]:
    """Compute confidence scores for extracted fields."""
    weights = {ExtractionSource.VISION: 0.95, ExtractionSource.FAILED: 0.0}
    base = weights.get(source, 0.0)
    conf = {}
    fields = ["symbol", "trade_date", "entry_price", "exit_price", "sl_price", "tp_price", "direction", "position_size", "leverage", "pnl", "notes", "session", "strategy", "risk_percentage", "risk_amount", "reward_amount", "r_r_ratio", "suggestion", "asset_type", "is_trade_confirmation"]
    for field in fields:
        conf[field] = round(base if parsed.get(field) is not None or field == "is_trade_confirmation" else 0.0, 2)
    conf["overall"] = round(sum(conf.values()) / len(conf), 2)
    return conf

async def _call_openai_with_lib(messages: list, response_format: Dict[str, Any], model: str = OPENAI_MODEL) -> Dict[str, Any]:
    """Call OpenAI with official client and retries."""
    try:
        raw_response = await openai_client.chat.completions.with_raw_response.create(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=0.0,
            max_tokens=512,
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
    """Fallback OpenAI call with httpx and retry logic."""
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
        return resp.json()

async def call_openai_vision(image_bytes: bytes, max_attempts: int = 3) -> Tuple[Optional[TradeSchema], Dict[str, Any]]:
    """Extract trade data from image using OpenAI vision API."""
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
            "strict": True,
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
                    "max_tokens": 512,
                }
                raw = await _call_openai_fallback(payload)
            
            content = raw["choices"][0]["message"]["content"]
            parsed_dict = json.loads(content) if content else None
            parsed = TradeSchema(**parsed_dict) if parsed_dict else None
            if parsed:
                computed_dict = compute_trade_metrics(parsed.dict())
                parsed = TradeSchema(**computed_dict)
                logger.info("Vision extraction succeeded on attempt %d", attempt + 1)
                return parsed, raw
        except Exception as e:
            logger.warning("Vision attempt %d failed: %s", attempt + 1, e)
            if attempt < max_attempts - 1:
                await asyncio.sleep(2 ** attempt)
    
    logger.error("Vision extraction failed after %d attempts", max_attempts)
    return None, {"error": "Max attempts exceeded"}

@router.get("/monthly_uploads")
async def get_monthly_uploads(db: AsyncSession = Depends(get_session), current_user: models.User = Depends(auth.get_current_user)):
    """Get count of trades uploaded this month."""
    count = await get_monthly_upload_count(db, current_user.id)
    return {"count": count}

@router.post("/extract_batch")
async def extract_batch_trades(
    files: List[UploadFile] = File(..., media_type="image/*"),
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user),
):
    """Extract trades from multiple screenshots, deduplicate, and return for review."""
    num_files = len(files)
    await enforce_upload_limits(db, current_user, num_files)
    correlation_id = asyncio.current_task().get_name()
    logger.info("Batch extract started: user=%s, files=%d", current_user.id, num_files, extra={"corr_id": correlation_id})
    
    symbols_cache = await get_cached_symbols(db)
    extracted = []
    source = ExtractionSource.VISION
    
    for file in files:
        contents = await file.read()
        if not contents:
            continue
        ext = file.filename.split('.')[-1]
        filename = f"{uuid.uuid4()}.{ext}"
        chart_url = save_image_locally(contents, filename)
        processed_bytes = validate_and_preprocess_image(contents)
        parsed, raw_response = await call_openai_vision(processed_bytes)
        
        if not parsed:
            partial = {
                "symbol": None,
                "trade_date": None,
                "entry_price": None,
                "exit_price": None,
                "sl_price": None,
                "tp_price": None,
                "direction": None,
                "position_size": None,
                "leverage": None,
                "pnl": None,
                "notes": "Manual review needed - AI couldn't extract details from chart",
                "session": "UTC",
                "strategy": None,
                "risk_percentage": None,
                "risk_amount": None,
                "reward_amount": None,
                "r_r_ratio": None,
                "suggestion": "Upload a clearer screenshot with entry/exit/SL/TP/risk markers next time.",
                "chart_url": chart_url,
                "asset_type": None,
                "_confidence": 0.0,
                "_is_partial": True
            }
            extracted.append(partial)
            logger.warning(f"Partial entry created for {file.filename}", extra={"corr_id": correlation_id})
            continue
        
        if (parsed.symbol or parsed.entry_price or parsed.sl_price or parsed.tp_price) and not parsed.is_trade_confirmation:
            parsed.is_trade_confirmation = True
            logger.info(f"Overrode confirmation for partial chart in {file.filename}")
        
        if not parsed.is_trade_confirmation:
            partial = parsed.dict()
            partial.pop('is_trade_confirmation', None)
            partial["notes"] = partial.get("notes", "") + " (AI flagged as non-trade; review manually)"
            partial["_confidence"] = 0.3
            partial["_is_partial"] = True
            partial["chart_url"] = chart_url
            extracted.append(partial)
            continue
        
        parsed.symbol = normalize_symbol(parsed.symbol)
        if parsed.symbol:
            parsed.symbol = await fuzzy_match_symbol(db, parsed.symbol, symbols_cache)
        if not parsed.symbol and not parsed.entry_price:
            continue
        conf = compute_confidence(parsed.dict(), source)
        ex_dict = parsed.dict()
        ex_dict.pop('is_trade_confirmation', None)
        ex_dict['_confidence'] = conf['overall']
        ex_dict['chart_url'] = chart_url
        extracted.append(ex_dict)
    
    extracted = [ex for ex in extracted if is_valid_trade(ex)]
    
    if not extracted:
        raise HTTPException(status_code=422, detail="No meaningful trade data extracted from any image. Try clearer screenshots with marked entry/SL/TP.")
    
    groups = defaultdict(list)
    for ex in extracted:
        symbol = ex.get('symbol', '')
        date_part = ex.get('trade_date', '').split('T')[0] if ex.get('trade_date') else ''
        entry = round(ex.get('entry_price') or 0, 4)
        sl = round(ex.get('sl_price') or 0, 4)
        tp = round(ex.get('tp_price') or 0, 4)
        asset = ex.get('asset_type', '')
        lev = round(ex.get('leverage') or 0, 1)
        key = (symbol, date_part, entry, sl, tp, asset, lev)
        groups[key].append(ex)
    
    unique_extracted = [max(group, key=lambda x: x.get('_confidence', 0)) for key, group in groups.items() if group]
    unique_extracted.sort(key=lambda x: x.get('_confidence', 0), reverse=True)
    
    logger.info("Batch extract completed: %d unique trades", len(unique_extracted), extra={"corr_id": correlation_id})
    return unique_extracted

@router.post("/extract", response_model=TradeSchema)
async def extract_trade_screenshot(
    file: UploadFile = File(..., media_type="image/*"),
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user),
):
    """Extract trade from a single screenshot for review."""
    num_files = 1
    await enforce_upload_limits(db, current_user, num_files)
    correlation_id = asyncio.current_task().get_name()
    logger.info("Extract started: user=%s, file=%s", current_user.id, file.filename, extra={"corr_id": correlation_id})
    
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    
    processed_bytes = validate_and_preprocess_image(contents)
    parsed: Optional[TradeSchema] = None
    raw_response: Dict[str, Any] = {}
    source = ExtractionSource.VISION
    symbols_cache = await get_cached_symbols(db)
    
    try:
        parsed, raw_response = await call_openai_vision(processed_bytes)
        if not parsed:
            raise ValueError("AI extraction returned no data")
    except Exception as e:
        logger.error("AI vision failed entirely: %s", e, extra={"corr_id": correlation_id})
        source = ExtractionSource.FAILED
        parsed = TradeSchema(session="UTC")
        raise HTTPException(status_code=503, detail="AI extraction temporarily unavailable (rate limit?) - try again soon")
    
    if not parsed.is_trade_confirmation:
        raise HTTPException(status_code=422, detail="Image not recognized as trade confirmation")
    
    if HAS_JSONSCHEMA:
        try:
            jsonschema_validate(parsed.dict(), TRADE_JSON_SCHEMA)
        except JsonSchemaValidationError as e:
            logger.warning("Schema validation failed: %s; using partial data", e, extra={"corr_id": correlation_id})
    
    parsed.symbol = normalize_symbol(parsed.symbol)
    if parsed.symbol:
        parsed.symbol = await fuzzy_match_symbol(db, parsed.symbol, symbols_cache)
    
    parsed_dict = parsed.dict()
    if not is_valid_trade(parsed_dict):
        raise HTTPException(status_code=422, detail="Incomplete trade data: At least 2 key fields (symbol, direction, entry_price, sl_price, tp_price) required")
    
    ext = file.filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    chart_url = save_image_locally(contents, filename)
    parsed.chart_url = chart_url
    
    conf = compute_confidence(parsed.dict(), source)
    parsed_dict = parsed.dict()
    parsed_dict["_confidence"] = conf
    parsed_dict["_extraction_source"] = source.value
    
    logger.info("Extract completed: symbol=%s, conf=%.2f", parsed.symbol, conf["overall"], extra={"corr_id": correlation_id})
    
    return TradeSchema(**{k: v for k, v in parsed_dict.items() if k not in ["_confidence", "_extraction_source"]})

@router.post("/save_batch", response_model=List[Dict[str, Any]], status_code=status.HTTP_201_CREATED)
async def save_batch_trades(
    trades_data: List[TradeSchema] = Body(...),
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user),
):
    """Save multiple reviewed trades to the database with detailed error handling."""
    if not trades_data:
        raise HTTPException(status_code=400, detail="No trades to save")
    correlation_id = asyncio.current_task().get_name()
    logger.info("Batch save started: user=%s, trades=%d, data=%s", current_user.id, len(trades_data), trades_data, extra={"corr_id": correlation_id})
    
    supported_fields = {
        'symbol', 'trade_date', 'entry_price', 'exit_price', 'sl_price', 'tp_price', 'direction', 'position_size', 'leverage',
        'pnl', 'notes', 'session', 'strategy', 'risk_percentage', 'risk_amount', 'reward_amount', 'r_r_ratio', 'suggestion',
        'fees', 'ai_log', 'chart_url', 'asset_type'
    }
    saved_trades = []
    errors = []
    account_balance = getattr(current_user, 'account_balance', 10000.0)
    
    for idx, trade in enumerate(trades_data):
        try:
            if not trade.symbol:
                logger.warning("Skipping trade at index %d: missing symbol", idx, extra={"corr_id": correlation_id})
                errors.append({"index": idx, "error": "Symbol is required"})
                continue
            
            trade_dict = trade.dict(exclude_none=False)
            logger.debug("Processing trade at index %d: %s", idx, trade_dict, extra={"corr_id": correlation_id})
            
            # Handle legacy mappings
            if 'position_size' not in trade_dict or trade_dict['position_size'] is None:
                trade_dict['position_size'] = trade_dict.get('size')
            if 'risk_percentage' not in trade_dict or trade_dict['risk_percentage'] is None:
                trade_dict['risk_percentage'] = trade_dict.get('risk')
            
            trade_dict_filtered = {k: v for k, v in trade_dict.items() if k in supported_fields}
            
            if 'direction' in trade_dict_filtered and isinstance(trade_dict_filtered['direction'], TradeDirection):
                trade_dict_filtered['direction'] = trade_dict_filtered['direction'].value
            if 'asset_type' in trade_dict_filtered and isinstance(trade_dict_filtered['asset_type'], AssetType):
                trade_dict_filtered['asset_type'] = trade_dict_filtered['asset_type'].value
            
            # Default strategy and session
            if 'strategy' not in trade_dict_filtered or not trade_dict_filtered['strategy']:
                trade_dict_filtered['strategy'] = getattr(current_user, 'strategy', 'Manual')
            if 'session' not in trade_dict_filtered or not trade_dict_filtered['session']:
                trade_dict_filtered['session'] = 'UTC'
            
            if not is_valid_trade(trade_dict_filtered):
                logger.warning("Skipping trade at index %d for %s: insufficient key fields", idx, trade_dict_filtered.get('symbol'), extra={"corr_id": correlation_id})
                errors.append({"index": idx, "error": "Incomplete trade data: At least 2 key fields (symbol, direction, entry_price, sl_price, tp_price) required"})
                continue
            
            if trade_dict_filtered.get('risk_amount') and account_balance > 0:
                trade_dict_filtered['risk_percentage'] = round((trade_dict_filtered['risk_amount'] / account_balance) * 100, 2)
            
            if trade_dict_filtered.get('trade_date'):
                try:
                    dt_str = trade_dict_filtered['trade_date']
                    if 'Z' in dt_str:
                        dt_str = dt_str.replace('Z', '+00:00')
                    trade_dict_filtered['trade_date'] = datetime.fromisoformat(dt_str)
                except ValueError as e:
                    logger.warning("Invalid trade_date format at index %d: %s", idx, trade_dict_filtered['trade_date'], extra={"corr_id": correlation_id})
                    errors.append({"index": idx, "error": f"Invalid trade_date format: {trade_dict_filtered['trade_date']}"})
                    continue
            
            new_trade = models.Trade(**trade_dict_filtered)
            if hasattr(new_trade, "owner_id"):
                new_trade.owner_id = current_user.id
            elif hasattr(new_trade, "user_id"):
                new_trade.user_id = current_user.id
            else:
                new_trade.owner = current_user
            if hasattr(new_trade, "raw_ai_response"):
                new_trade.raw_ai_response = json.dumps({})
            if hasattr(new_trade, "confidence"):
                new_trade.confidence = 0.95
            db.add(new_trade)
            saved_trades.append(new_trade)
        except ValidationError as e:
            error_details = [{"loc": err["loc"], "msg": err["msg"], "type": err["type"]} for err in e.errors()]
            logger.error("Validation failed for trade at index %d: %s, data: %s", idx, error_details, trade_dict, extra={"corr_id": correlation_id})
            errors.append({"index": idx, "error": error_details})
            continue
        except Exception as e:
            logger.error("Failed to save trade at index %d: %s, data: %s", idx, e, trade_dict, extra={"corr_id": correlation_id})
            errors.append({"index": idx, "error": str(e)})
            continue
    
    if errors and not saved_trades:
        raise HTTPException(
            status_code=422,
            detail={"message": "All trades failed validation", "errors": errors}
        )
    
    try:
        await db.commit()
        logger.info("Batch save completed: %d trades saved, %d failed", len(saved_trades), len(errors), extra={"corr_id": correlation_id})
        
        response = [schemas.TradeResponse.model_validate(t, from_attributes=True).dict() for t in saved_trades]
        if errors:
            return {"saved_trades": response, "errors": errors}
        return response
    except Exception as e:
        await db.rollback()
        logger.error("DB commit failed: %s", e, extra={"corr_id": correlation_id})
        raise HTTPException(status_code=500, detail={"message": "Database save failed", "errors": errors})

@router.post("/save", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def save_extracted_trade(
    trade_data: TradeSchema,
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user),
):
    """Save a single reviewed trade to the database."""
    num_files = 1
    await enforce_upload_limits(db, current_user, num_files)
    correlation_id = asyncio.current_task().get_name()
    logger.info("Save started: user=%s, symbol=%s, data=%s", current_user.id, trade_data.symbol, trade_data.dict(), extra={"corr_id": correlation_id})
    
    if not trade_data.symbol:
        raise HTTPException(status_code=422, detail="Symbol is required")
    
    supported_fields = {
        'symbol', 'trade_date', 'entry_price', 'exit_price', 'sl_price', 'tp_price', 'direction', 'position_size', 'leverage',
        'pnl', 'notes', 'session', 'strategy', 'risk_percentage', 'risk_amount', 'reward_amount', 'r_r_ratio', 'suggestion',
        'fees', 'ai_log', 'chart_url', 'asset_type'
    }
    trade_dict = trade_data.dict(exclude_none=False)
    
    if 'position_size' not in trade_dict or trade_dict['position_size'] is None:
        trade_dict['position_size'] = trade_dict.get('size')
    if 'risk_percentage' not in trade_dict or trade_dict['risk_percentage'] is None:
        trade_dict['risk_percentage'] = trade_dict.get('risk')
    
    trade_dict = {k: v for k, v in trade_dict.items() if k in supported_fields and k != "is_trade_confirmation"}
    
    if 'direction' in trade_dict and isinstance(trade_dict['direction'], TradeDirection):
        trade_dict['direction'] = trade_dict['direction'].value
    if 'asset_type' in trade_dict and isinstance(trade_dict['asset_type'], AssetType):
        trade_dict['asset_type'] = trade_dict['asset_type'].value
    
    if 'strategy' not in trade_dict or not trade_dict['strategy']:
        trade_dict['strategy'] = getattr(current_user, 'strategy', 'Manual')
    if 'session' not in trade_dict or not trade_dict['session']:
        trade_dict['session'] = 'UTC'
    
    if not is_valid_trade(trade_dict):
        raise HTTPException(status_code=422, detail="Incomplete trade data: At least 2 key fields (symbol, direction, entry_price, sl_price, tp_price) required")
    
    account_balance = getattr(current_user, 'account_balance', 10000.0)
    if trade_dict.get('risk_amount') and account_balance > 0:
        trade_dict['risk_percentage'] = round((trade_dict['risk_amount'] / account_balance) * 100, 2)
    
    if trade_dict.get('trade_date'):
        try:
            dt_str = trade_dict['trade_date']
            if 'Z' in dt_str:
                dt_str = dt_str.replace('Z', '+00:00')
            trade_dict['trade_date'] = datetime.fromisoformat(dt_str)
        except ValueError:
            logger.warning("Invalid trade_date format: %s", trade_dict['trade_date'], extra={"corr_id": correlation_id})
            del trade_dict['trade_date']
    
    try:
        new_trade = models.Trade(**trade_dict)
        if hasattr(new_trade, "owner_id"):
            new_trade.owner_id = current_user.id
        elif hasattr(new_trade, "user_id"):
            new_trade.user_id = current_user.id
        else:
            new_trade.owner = current_user
        if hasattr(new_trade, "raw_ai_response"):
            new_trade.raw_ai_response = json.dumps({})
        if hasattr(new_trade, "confidence"):
            new_trade.confidence = 0.95
        db.add(new_trade)
        await db.commit()
        logger.info("Trade saved: id=%s, symbol=%s", new_trade.id, new_trade.symbol, extra={"corr_id": correlation_id})
        return schemas.TradeResponse.model_validate(new_trade, from_attributes=True).dict()
    except Exception as e:
        await db.rollback()
        logger.error("DB save failed: %s | Trade data: %s", e, trade_dict, extra={"corr_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Database save failed: {str(e)}")

@router.post("/", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def upload_trade_screenshot(
    file: UploadFile = File(..., media_type="image/*"),
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user),
):
    """Upload, extract, and save trade from a screenshot."""
    num_files = 1
    await enforce_upload_limits(db, current_user, num_files)
    correlation_id = asyncio.current_task().get_name()
    logger.info("Upload started: user=%s, file=%s", current_user.id, file.filename, extra={"corr_id": correlation_id})
    
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    
    processed_bytes = validate_and_preprocess_image(contents)
    parsed: Optional[TradeSchema] = None
    raw_response: Dict[str, Any] = {}
    source = ExtractionSource.VISION
    symbols_cache = await get_cached_symbols(db)
    
    try:
        parsed, raw_response = await call_openai_vision(processed_bytes)
        if not parsed:
            raise ValueError("AI extraction returned no data")
    except Exception as e:
        logger.error("AI vision failed entirely: %s", e, extra={"corr_id": correlation_id})
        source = ExtractionSource.FAILED
        parsed = TradeSchema(session="UTC")
        raise HTTPException(status_code=503, detail="AI extraction temporarily unavailable (rate limit?) - try again soon")
    
    if not parsed.is_trade_confirmation:
        raise HTTPException(status_code=422, detail="Image not recognized as trade confirmation")
    
    if HAS_JSONSCHEMA:
        try:
            jsonschema_validate(parsed.dict(), TRADE_JSON_SCHEMA)
        except JsonSchemaValidationError as e:
            logger.warning("Schema validation failed: %s; using partial data", e, extra={"corr_id": correlation_id})
    
    parsed.symbol = normalize_symbol(parsed.symbol)
    if parsed.symbol:
        parsed.symbol = await fuzzy_match_symbol(db, parsed.symbol, symbols_cache)
    
    parsed_dict = parsed.dict()
    if not is_valid_trade(parsed_dict):
        raise HTTPException(status_code=422, detail="Incomplete trade data: At least 2 key fields (symbol, direction, entry_price, sl_price, tp_price) required")
    
    ext = file.filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    chart_url = save_image_locally(contents, filename)
    parsed.chart_url = chart_url
    
    conf = compute_confidence(parsed.dict(), source)
    parsed_dict = parsed.dict()
    parsed_dict["_confidence"] = conf
    parsed_dict["_extraction_source"] = source.value
    
    try:
        supported_fields = {
            'symbol', 'trade_date', 'entry_price', 'exit_price', 'sl_price', 'tp_price', 'direction', 'position_size', 'leverage',
            'pnl', 'notes', 'session', 'strategy', 'risk_percentage', 'risk_amount', 'reward_amount', 'r_r_ratio', 'suggestion', 'chart_url', 'asset_type'
        }
        trade_dict = parsed_dict.copy()
        if 'position_size' not in trade_dict or trade_dict['position_size'] is None:
            trade_dict['position_size'] = trade_dict.get('size')
        if 'risk_percentage' not in trade_dict or trade_dict['risk_percentage'] is None:
            trade_dict['risk_percentage'] = trade_dict.get('risk')
        
        trade_dict = {k: v for k, v in trade_dict.items() if k in supported_fields}
        
        if 'direction' in trade_dict and isinstance(trade_dict['direction'], TradeDirection):
            trade_dict['direction'] = trade_dict['direction'].value
        if 'asset_type' in trade_dict and isinstance(trade_dict['asset_type'], AssetType):
            trade_dict['asset_type'] = trade_dict['asset_type'].value
        
        if 'strategy' not in trade_dict or not trade_dict['strategy']:
            trade_dict['strategy'] = getattr(current_user, 'strategy', 'Manual')
        if 'session' not in trade_dict or not trade_dict['session']:
            trade_dict['session'] = 'UTC'
        
        account_balance = getattr(current_user, 'account_balance', 10000.0)
        if trade_dict.get('risk_amount') and account_balance > 0:
            trade_dict['risk_percentage'] = round((trade_dict['risk_amount'] / account_balance) * 100, 2)
        
        if trade_dict.get('trade_date'):
            try:
                dt_str = trade_dict['trade_date']
                if 'Z' in dt_str:
                    dt_str = dt_str.replace('Z', '+00:00')
                trade_dict['trade_date'] = datetime.fromisoformat(dt_str)
            except ValueError:
                logger.warning("Invalid trade_date format: %s", trade_dict['trade_date'], extra={"corr_id": correlation_id})
                del trade_dict['trade_date']
        
        new_trade = models.Trade(**trade_dict)
        if hasattr(new_trade, "owner_id"):
            new_trade.owner_id = current_user.id
        elif hasattr(new_trade, "user_id"):
            new_trade.user_id = current_user.id
        else:
            new_trade.owner = current_user
        if hasattr(new_trade, "raw_ai_response"):
            new_trade.raw_ai_response = json.dumps(raw_response)
        if hasattr(new_trade, "confidence"):
            new_trade.confidence = conf["overall"]
        
        db.add(new_trade)
        await db.commit()
        
        logger.info("Trade saved: id=%s, symbol=%s, conf=%.2f", new_trade.id, new_trade.symbol, conf["overall"], extra={"corr_id": correlation_id})
        
        return schemas.TradeResponse.model_validate(new_trade, from_attributes=True).dict()
    except Exception as e:
        await db.rollback()
        logger.error("DB save failed: %s | Trade data: %s", e, trade_dict, extra={"corr_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Database save failed: {str(e)}")
# server/router/uploads.py
import logging
import asyncio
import json
from collections import defaultdict
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Union

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_session
import auth
from models import models, schemas
from app_utils.uploads_utils import *

# NEW: Import the spend function from app_utils
from app_utils.points import spend_trade_points

from redis_client import redis_dependency, get_cache
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["Uploads"])

@router.get("/monthly_uploads")
async def get_monthly_uploads(db: AsyncSession = Depends(get_session), current_user: models.User = Depends(auth.get_current_user)):
    count = await get_monthly_upload_count(db, current_user.id)
    return {"count": count}

@router.get("/limits", response_model=Dict[str, Any])
async def get_upload_limits(
    plan: Optional[str] = None,
    current_user: Optional[models.User] = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    if plan:
        limits = await get_plan_limits(db, plan)
        return limits
    
    plans = ['starter', 'pro', 'elite']
    all_limits = {}
    for p in plans:
        all_limits[p] = await get_plan_limits(db, p)
    return all_limits

@router.get("/points/balance")
async def get_points_balance(current_user: models.User = Depends(auth.get_current_user)):
    return {"trade_points": current_user.trade_points}

@router.post("/points/spend")
async def spend_points(
    payload: dict = Body(...),
    current_user: models.User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    action = payload.get("action")
    amount = payload.get("amount", 1)
    if not action:
        raise HTTPException(status_code=400, detail="Action required (e.g., 'upload', 'insight')")
    # CHANGED: Call the imported async function for points spending (no redis here, so None)
    await spend_trade_points(db, current_user, action, amount)
    await db.refresh(current_user)
    return {"balance": current_user.trade_points, "message": f"Spent {amount} TP on {action}."}

@router.post("/extract_batch")
async def extract_batch_trades(
    files: List[UploadFile] = File(..., media_type="image/*"),
    db: AsyncSession = Depends(get_session),
    redis_client: Optional[Redis] = Depends(redis_dependency),  # NEW: Optional Redis
    current_user: models.User = Depends(auth.get_current_user),
):
    num_files = len(files)
    await enforce_upload_limits(db, current_user, num_files, "upload")
    correlation_id = asyncio.current_task().get_name()
    logger.info("Batch extract started: user=%s, files=%d", current_user.id, num_files, extra={"corr_id": correlation_id})
    
    symbols_cache = await get_cached_symbols(db)
    extracted = []
    source = ExtractionSource.VISION
    account_balance = getattr(current_user, 'account_balance', 10000.0)
    
    for file in files:
        contents = await file.read()
        if not contents:
            continue
        ext = file.filename.split('.')[-1]
        filename = f"{uuid.uuid4()}.{ext}"
        chart_url = save_image_locally(contents, filename)
        processed_bytes = validate_and_preprocess_image(contents)
        parsed, raw_response = await call_openai_vision(processed_bytes, account_balance=account_balance, redis_client=redis_client)
        
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
                # New fields
                "margin_required": None,
                "notional_value": None,
                "calculator_note": None,
                "auto_calculated": None,
                "calculator_version": None,
                "_confidence": 0.0,
                "_is_partial": True
            }
            extracted.append(partial)
            continue
        
        if (parsed.symbol or parsed.entry_price or parsed.sl_price or parsed.tp_price) and not parsed.is_trade_confirmation:
            parsed.is_trade_confirmation = True
        
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
        raise HTTPException(status_code=422, detail="No meaningful trade data extracted from any image.")
    
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
    redis_client: Optional[Redis] = Depends(redis_dependency),  # NEW: Optional Redis
    current_user: models.User = Depends(auth.get_current_user),
):
    await enforce_upload_limits(db, current_user, 1, "upload")
    correlation_id = asyncio.current_task().get_name()
    logger.info("Extract started: user=%s, file=%s", current_user.id, file.filename, extra={"corr_id": correlation_id})
    
    # NEW: Check Redis for recent backoff before calling AI
    user_key = f"rate_limit_backoff:{current_user.id}"
    backoff_data = await get_cache(redis_client, user_key) if redis_client else None
    if backoff_data:
        try:
            backoff = json.loads(backoff_data)
            retry_after = backoff.get("retry_after", 300)
            raise HTTPException(503, f"Recent rate limit detected. Wait {retry_after//60}min before retrying. Error: {backoff.get('error', 'Unknown')}")
        except (json.JSONDecodeError, KeyError):
            pass  # Proceed if invalid
    
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    
    processed_bytes = validate_and_preprocess_image(contents)
    account_balance = getattr(current_user, 'account_balance', 10000.0)
    try:
        parsed, raw_response = await call_openai_vision(
            processed_bytes, account_balance=account_balance, redis_client=redis_client
        )
    except HTTPException:
        raise  # Re-raise enhanced 503 from vision func
    if not parsed:
        # NEW: Fallback to partial extraction
        logger.warning("Falling back to partial mode for user %s", current_user.id, extra={"corr_id": correlation_id})
        parsed = TradeSchema(
            symbol=None, direction=None, notes="AI extraction failed - manual review required",
            chart_url=save_image_locally(contents, f"{uuid.uuid4()}.jpg"),  # Save image anyway
            is_trade_confirmation=False, suggestion="Upload clearer chart or review manually.",
            # Set other fields to defaults/None
            trade_date=None, entry_price=None, exit_price=None, sl_price=None, tp_price=None,
            position_size=None, leverage=None, pnl=None, session="UTC", strategy=None,
            risk_percentage=None, risk_amount=None, reward_amount=None, r_r_ratio=None,
            asset_type=None, margin_required=None, notional_value=None, calculator_note=None,
            auto_calculated=None, calculator_version="1.1"
        )
    
    if not parsed.is_trade_confirmation:
        raise HTTPException(status_code=422, detail="Image not recognized as trade confirmation")
    
    if HAS_JSONSCHEMA:
        try:
            jsonschema_validate(parsed.dict(), TRADE_JSON_SCHEMA)
        except JsonSchemaValidationError as e:
            logger.warning("Schema validation failed: %s; using partial data", e, extra={"corr_id": correlation_id})
    
    symbols_cache = await get_cached_symbols(db)
    parsed.symbol = normalize_symbol(parsed.symbol)
    if parsed.symbol:
        parsed.symbol = await fuzzy_match_symbol(db, parsed.symbol, symbols_cache)
    
    parsed_dict = parsed.dict()
    if not is_valid_trade(parsed_dict):
        raise HTTPException(status_code=422, detail="Incomplete trade data")
    
    ext = file.filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    chart_url = save_image_locally(contents, filename)
    parsed.chart_url = chart_url
    
    conf = compute_confidence(parsed.dict(), ExtractionSource.VISION)
    parsed_dict = parsed.dict()
    parsed_dict["_confidence"] = conf
    parsed_dict["_extraction_source"] = ExtractionSource.VISION.value
    
    return TradeSchema(**{k: v for k, v in parsed_dict.items() if not k.startswith('_')})

# FIXED: Changed response_model to Dict[str, Any] to accommodate error responses
@router.post("/save_batch", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def save_batch_trades(
    trades_data: List[TradeSchema] = Body(...),
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user),
):
    if not trades_data:
        raise HTTPException(status_code=400, detail="No trades to save")
    num_files = len(trades_data)
    await enforce_upload_limits(db, current_user, num_files, "save")
    correlation_id = asyncio.current_task().get_name()
    logger.info("Batch save started: user=%s, trades=%d", current_user.id, len(trades_data), extra={"corr_id": correlation_id})
    
    # FIXED: Separate DB fields from calculator fields (assuming Trade model lacks new fields)
    db_fields = {
        'symbol', 'trade_date', 'entry_price', 'exit_price', 'sl_price', 'tp_price', 'direction', 'position_size', 'leverage',
        'pnl', 'notes', 'session', 'strategy', 'risk_percentage', 'risk_amount', 'reward_amount', 'r_r_ratio', 'suggestion',
        'fees', 'ai_log', 'chart_url', 'asset_type'
    }
    saved_trades = []
    errors = []
    total_pnl_delta = 0.0
    
    for idx, trade in enumerate(trades_data):
        try:
            if not trade.symbol:
                errors.append({"index": idx, "error": "Symbol is required"})
                continue
            
            trade_dict = trade.dict(exclude_none=False)
            
            if 'position_size' not in trade_dict or trade_dict['position_size'] is None:
                trade_dict['position_size'] = trade_dict.get('size')
            if 'risk_percentage' not in trade_dict or trade_dict['risk_percentage'] is None:
                trade_dict['risk_percentage'] = trade_dict.get('risk')
            
            # FIXED: Filter only to DB fields, exclude calculator fields
            trade_dict_filtered = {k: v for k, v in trade_dict.items() if k in db_fields}
            
            if 'direction' in trade_dict_filtered and isinstance(trade_dict_filtered['direction'], TradeDirection):
                trade_dict_filtered['direction'] = trade_dict_filtered['direction'].value
            if 'asset_type' in trade_dict_filtered and isinstance(trade_dict_filtered['asset_type'], AssetType):
                trade_dict_filtered['asset_type'] = trade_dict_filtered['asset_type'].value
            
            if 'strategy' not in trade_dict_filtered or not trade_dict_filtered['strategy']:
                trade_dict_filtered['strategy'] = getattr(current_user, 'strategy', 'Manual')
            if 'session' not in trade_dict_filtered or not trade_dict_filtered['session']:
                trade_dict_filtered['session'] = 'UTC'
            
            if not is_valid_trade(trade_dict_filtered):
                errors.append({"index": idx, "error": "Incomplete trade data"})
                continue
            
            if trade_dict_filtered.get('risk_amount') and current_user.account_balance > 0:
                trade_dict_filtered['risk_percentage'] = round((trade_dict_filtered['risk_amount'] / current_user.account_balance) * 100, 2)
            
            if trade_dict_filtered.get('trade_date'):
                try:
                    dt_str = trade_dict_filtered['trade_date']
                    if 'Z' in dt_str:
                        dt_str = dt_str.replace('Z', '+00:00')
                    parsed_dt = datetime.fromisoformat(dt_str)
                    trade_dict_filtered['trade_date'] = parsed_dt.replace(tzinfo=None)
                except ValueError:
                    errors.append({"index": idx, "error": "Invalid trade_date format"})
                    continue
            
            # P&L recalc
            if (trade_dict_filtered.get('entry_price') is not None and 
                trade_dict_filtered.get('exit_price') is not None and
                trade_dict_filtered.get('position_size') is not None and
                trade_dict_filtered.get('pnl') is None):
                direction_str = str(trade_dict_filtered.get('direction')) if trade_dict_filtered.get('direction') else None
                delta = (trade_dict_filtered['exit_price'] - trade_dict_filtered['entry_price']) if direction_str in (None, 'LONG') else (trade_dict_filtered['entry_price'] - trade_dict_filtered['exit_price'])
                leverage = trade_dict_filtered.get('leverage') or 1.0
                base_pnl = delta * trade_dict_filtered['position_size'] * leverage
                asset_type_str = str(trade_dict_filtered.get('asset_type')) if trade_dict_filtered.get('asset_type') else None
                if asset_type_str == 'FOREX':
                    pip_factor = 100 if 'JPY' in (trade_dict_filtered.get('symbol') or '') else 10000
                    base_pnl *= (abs(delta) * pip_factor / abs(delta)) * 10
                fees_val = trade_dict_filtered.get('fees') or 0.0
                trade_dict_filtered['pnl'] = round(base_pnl - fees_val, 2)
            
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
            
            pnl_delta = new_trade.pnl or 0.0
            if pnl_delta != 0:
                current_user.account_balance += pnl_delta
                total_pnl_delta += pnl_delta
            
            db.add(new_trade)
            saved_trades.append(new_trade)
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})
            continue
    
    # FIXED: Raise if all failed
    if len(saved_trades) == 0 and errors:
        raise HTTPException(status_code=422, detail={"message": "All trades failed", "errors": errors})
    
    try:
        if saved_trades:
            await db.commit()
            if total_pnl_delta != 0:  # Only update if needed
                current_user.account_balance += total_pnl_delta
                await db.commit()  # Single commit for user update
            logger.info("Batch save completed: %d saved, %d failed, P&L delta: %.2f", len(saved_trades), len(errors), total_pnl_delta, extra={"corr_id": correlation_id})
            
            response = [schemas.TradeResponse.model_validate(t, from_attributes=True).dict() for t in saved_trades]
        else:
            response = []
        
        # FIXED: Always return dict for consistency
        result = {"saved_trades": response, "errors": errors}
        return result
    except Exception as commit_e:
        await db.rollback()
        logger.error("Batch commit failed: %s", commit_e, extra={"corr_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Database commit failed: {str(commit_e)}")

@router.post("/save", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def save_extracted_trade(
    trade_data: TradeSchema,
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user),
):
    await enforce_upload_limits(db, current_user, 1, "save")
    correlation_id = asyncio.current_task().get_name()
    logger.info("Save started: user=%s, symbol=%s, data=%s", current_user.id, trade_data.symbol, trade_data.dict(), extra={"corr_id": correlation_id})
    
    if not trade_data.symbol:
        raise HTTPException(status_code=422, detail="Symbol is required")
    
    # FIXED: Separate DB fields from calculator fields
    db_fields = {
        'symbol', 'trade_date', 'entry_price', 'exit_price', 'sl_price', 'tp_price', 'direction', 'position_size', 'leverage',
        'pnl', 'notes', 'session', 'strategy', 'risk_percentage', 'risk_amount', 'reward_amount', 'r_r_ratio', 'suggestion',
        'fees', 'ai_log', 'chart_url', 'asset_type'
    }
    trade_dict = trade_data.dict(exclude_none=False)
    
    if 'position_size' not in trade_dict or trade_dict['position_size'] is None:
        trade_dict['position_size'] = trade_dict.get('size')
    if 'risk_percentage' not in trade_dict or trade_dict['risk_percentage'] is None:
        trade_dict['risk_percentage'] = trade_dict.get('risk')
    
    # FIXED: Filter only to DB fields, exclude calculator fields
    trade_dict = {k: v for k, v in trade_dict.items() if k in db_fields and k != "is_trade_confirmation"}
    
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
            parsed_dt = datetime.fromisoformat(dt_str)
            trade_dict['trade_date'] = parsed_dt.replace(tzinfo=None)
        except ValueError:
            logger.warning("Invalid trade_date format: %s", trade_dict['trade_date'], extra={"corr_id": correlation_id})
            del trade_dict['trade_date']
    
    # Recalc P&L if needed
    if (trade_dict.get('entry_price') is not None and 
        trade_dict.get('exit_price') is not None and
        trade_dict.get('position_size') is not None and
        trade_dict.get('pnl') is None):
        direction_str = str(trade_dict.get('direction')) if trade_dict.get('direction') else None
        delta = (trade_dict['exit_price'] - trade_dict['entry_price']) if direction_str in (None, 'LONG') else (trade_dict['entry_price'] - trade_dict['exit_price'])
        leverage = trade_dict.get('leverage') or 1.0
        base_pnl = delta * trade_dict['position_size'] * leverage
        asset_type_str = str(trade_dict.get('asset_type')) if trade_dict.get('asset_type') else None
        if asset_type_str == 'FOREX':
            pip_factor = 100 if 'JPY' in (trade_dict.get('symbol') or '') else 10000
            base_pnl *= (abs(delta) * pip_factor / abs(delta)) * 10
        fees_val = trade_dict.get('fees') or 0.0
        trade_dict['pnl'] = round(base_pnl - fees_val, 2)
    
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
        
        pnl_delta = new_trade.pnl or 0.0
        if pnl_delta != 0:
            current_user.account_balance += pnl_delta
            logger.info(f"Saved trade {new_trade.id} for user {current_user.id}, added {pnl_delta:.2f} to balance (new: {current_user.account_balance:.2f})", extra={"corr_id": correlation_id})
        
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
    redis_client: Optional[Redis] = Depends(redis_dependency),  # NEW: Optional Redis
    current_user: models.User = Depends(auth.get_current_user),
):
    await enforce_upload_limits(db, current_user, 1, "upload")
    correlation_id = asyncio.current_task().get_name()
    logger.info("Upload started: user=%s, file=%s", current_user.id, file.filename, extra={"corr_id": correlation_id})
    
    # NEW: Check Redis for recent backoff before calling AI
    user_key = f"rate_limit_backoff:{current_user.id}"
    backoff_data = await get_cache(redis_client, user_key) if redis_client else None
    if backoff_data:
        try:
            backoff = json.loads(backoff_data)
            retry_after = backoff.get("retry_after", 300)
            raise HTTPException(503, f"Recent rate limit detected. Wait {retry_after//60}min before retrying. Error: {backoff.get('error', 'Unknown')}")
        except (json.JSONDecodeError, KeyError):
            pass  # Proceed if invalid
    
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    
    processed_bytes = validate_and_preprocess_image(contents)
    parsed: Optional[TradeSchema] = None
    raw_response: Dict[str, Any] = {}
    source = ExtractionSource.VISION
    symbols_cache = await get_cached_symbols(db)
    account_balance = getattr(current_user, 'account_balance', 10000.0)
    
    try:
        parsed, raw_response = await call_openai_vision(
            processed_bytes, account_balance=account_balance, redis_client=redis_client
        )
    except HTTPException:
        raise  # Re-raise enhanced 503 from vision func
    if not parsed:
        # NEW: Fallback to partial upload (manual review mode)
        logger.warning("Falling back to manual mode for user %s", current_user.id, extra={"corr_id": correlation_id})
        parsed = TradeSchema(
            symbol=None, direction=None, notes="AI extraction failed - manual review required",
            chart_url=save_image_locally(contents, f"{uuid.uuid4()}.jpg"),  # Save image anyway
            is_trade_confirmation=False, suggestion="Upload clearer chart or review manually.",
            # Set other fields to defaults/None
            trade_date=None, entry_price=None, exit_price=None, sl_price=None, tp_price=None,
            position_size=None, leverage=None, pnl=None, session="UTC", strategy=None,
            risk_percentage=None, risk_amount=None, reward_amount=None, r_r_ratio=None,
            asset_type=None, margin_required=None, notional_value=None, calculator_note=None,
            auto_calculated=None, calculator_version="1.1"
        )
    
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
    
    try:
        # FIXED: Separate DB fields from calculator fields
        db_fields = {
            'symbol', 'trade_date', 'entry_price', 'exit_price', 'sl_price', 'tp_price', 'direction', 'position_size', 'leverage',
            'pnl', 'notes', 'session', 'strategy', 'risk_percentage', 'risk_amount', 'reward_amount', 'r_r_ratio', 'suggestion', 'chart_url', 'asset_type'
        }
        trade_dict = parsed.dict().copy()
        if 'position_size' not in trade_dict or trade_dict['position_size'] is None:
            trade_dict['position_size'] = trade_dict.get('size')
        if 'risk_percentage' not in trade_dict or trade_dict['risk_percentage'] is None:
            trade_dict['risk_percentage'] = trade_dict.get('risk')
        
        # FIXED: Filter only to DB fields, exclude calculator fields
        trade_dict = {k: v for k, v in trade_dict.items() if k in db_fields}
        
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
                parsed_dt = datetime.fromisoformat(dt_str)
                trade_dict['trade_date'] = parsed_dt.replace(tzinfo=None)
            except ValueError:
                logger.warning("Invalid trade_date format: %s", trade_dict['trade_date'], extra={"corr_id": correlation_id})
                del trade_dict['trade_date']
        
        # Recalc P&L if needed
        if (trade_dict.get('entry_price') is not None and 
            trade_dict.get('exit_price') is not None and
            trade_dict.get('position_size') is not None and
            trade_dict.get('pnl') is None):
            direction_str = str(trade_dict.get('direction')) if trade_dict.get('direction') else None
            delta = (trade_dict['exit_price'] - trade_dict['entry_price']) if direction_str in (None, 'LONG') else (trade_dict['entry_price'] - trade_dict['exit_price'])
            leverage = trade_dict.get('leverage') or 1.0
            base_pnl = delta * trade_dict['position_size'] * leverage
            asset_type_str = str(trade_dict.get('asset_type')) if trade_dict.get('asset_type') else None
            if asset_type_str == 'FOREX':
                pip_factor = 100 if 'JPY' in (trade_dict.get('symbol') or '') else 10000
                base_pnl *= (abs(delta) * pip_factor / abs(delta)) * 10
            fees_val = trade_dict.get('fees') or 0.0
            trade_dict['pnl'] = round(base_pnl - fees_val, 2)
        
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
        
        pnl_delta = new_trade.pnl or 0.0
        if pnl_delta != 0:
            current_user.account_balance += pnl_delta
            logger.info(f"Uploaded trade {new_trade.id} for user {current_user.id}, added {pnl_delta:.2f} to balance (new: {current_user.account_balance:.2f})", extra={"corr_id": correlation_id})
        
        db.add(new_trade)
        await db.commit()
        
        logger.info("Trade saved: id=%s, symbol=%s, conf=%.2f", new_trade.id, new_trade.symbol, conf["overall"], extra={"corr_id": correlation_id})
        
        return schemas.TradeResponse.model_validate(new_trade, from_attributes=True).dict()
    except Exception as e:
        await db.rollback()
        logger.error("DB save failed: %s | Trade data: %s", e, trade_dict, extra={"corr_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Database save failed: {str(e)}")

@router.get("/platform-plan", response_model=Dict[str, str])
async def get_platform_plan(
    current_user: models.User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    # Query active platform subscription (trader_id is None, status=active)
    result = await db.execute(
        select(models.Subscription.plan_type)
        .where(
            models.Subscription.user_id == current_user.id,
            models.Subscription.status == 'active',
            models.Subscription.trader_id.is_(None)
        )
        .order_by(models.Subscription.start_date.desc())
        .limit(1)
    )
    sub_plan_type = result.scalar_one_or_none()
    
    if sub_plan_type:
        # Extract base plan (e.g., 'pro_monthly' -> 'pro')
        platform_plan = sub_plan_type.split('_')[0].lower()
        # Validate it's a platform plan
        if platform_plan in ['starter', 'pro', 'elite']:
            logger.info("Queried platform plan for user %s: %s", current_user.id, platform_plan)
            return {"platform_plan": platform_plan}
    
    # Fallback to user.plan (assuming it's the platform plan)
    user_plan = current_user.plan.lower()
    if user_plan in ['starter', 'pro', 'elite']:
        logger.info("Queried platform plan for user %s: %s (fallback)", current_user.id, user_plan)
        return {"platform_plan": user_plan}
    
    # Ultimate fallback
    logger.info("Queried platform plan for user %s: starter (ultimate fallback)", current_user.id)
    return {"platform_plan": "starter"}
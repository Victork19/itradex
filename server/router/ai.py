# server/router/ai.py
import io
import os
import json
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func  # Added func import
from datetime import datetime, timedelta
import httpx
import logging  # NEW: For logging unrecognized plans

from config import settings
import auth
from database import get_session
from router.insights import compute_insights
from models.models import User, Subscription, Trade, PointTransaction, AiChatLimits

logger = logging.getLogger("iTrade")  # NEW: Logger for warnings

router = APIRouter(prefix="/ai", tags=["AI"])

OPENAI_API_KEY = getattr(settings, "OPENAI_API_KEY", None)
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY required for AI endpoints")

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"  # official TTS endpoint

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}

async def spend_trade_points(db: AsyncSession, user: User, action: str, amount: int = 1) -> None:
    """Deduct Trade Points for an action (e.g., 'chat')."""
    if user.trade_points < amount:
        raise HTTPException(status_code=402, detail=f"Insufficient Trade Points ({user.trade_points} remaining). Refer friends to earn more!")
    
    user.trade_points -= amount
    # Log transaction
    tx = PointTransaction(
        user_id=user.id,
        type=action,
        amount=-amount,
        description=f"Spent {amount} TP on {action}"
    )
    db.add(tx)
    await db.commit()

async def check_monthly_chat_limit(db: AsyncSession, user: User) -> tuple[int, dict]:
    """Check and return monthly chats used and plan limits. Limits fetched from DB or defaults."""
    normalized_plan = user.plan.lower().split('_')[0] if '_' in user.plan.lower() else user.plan.lower()
    
    # NEW: Fallback to 'starter' if normalized to something unrecognized (e.g., 'marketplace')
    valid_plans = {'starter', 'pro', 'elite'}
    if normalized_plan not in valid_plans:
        logger.warning(f"Unrecognized plan '{user.plan}' for user {user.id}, defaulting to 'starter'")
        normalized_plan = 'starter'
    
    # Query DB for limits
    result = await db.execute(
        select(AiChatLimits).where(AiChatLimits.plan == normalized_plan)
    )
    db_limit = result.scalar_one_or_none()
    
    # Defaults if not in DB
    defaults = {
        'starter': {'monthly_chat_limit': 5, 'tp_cost_chat': 1},
        'pro': {'monthly_chat_limit': 25, 'tp_cost_chat': 0},
        'elite': {'monthly_chat_limit': 50, 'tp_cost_chat': 0}
    }
    
    if db_limit:
        limits = {
            'monthly_chat_limit': db_limit.monthly_limit,
            'tp_cost_chat': db_limit.tp_cost
        }
    else:
        limits = defaults.get(normalized_plan, defaults['starter'])

    # Check monthly chat limit
    now = datetime.utcnow()
    start_of_month = now.replace(day=1)
    chat_count_result = await db.execute(
        select(func.count(PointTransaction.id))
        .where(
            PointTransaction.user_id == user.id,
            PointTransaction.type == 'chat',
            PointTransaction.created_at >= start_of_month
        )
    )
    monthly_chats_used = chat_count_result.scalar() or 0
    if monthly_chats_used >= limits['monthly_chat_limit']:
        next_plan = 'pro' if normalized_plan == 'starter' else 'elite'
        next_limit = defaults[next_plan]['monthly_chat_limit']  # Fallback to default for suggestion
        upgrade_msg = f"""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); color: white; border-radius: 18px; margin: 10px 0;">
          <p><strong>Unlock More Conversations</strong></p>
          <p>You've used your {limits['monthly_chat_limit']} chats this month. Upgrade to {next_plan.title()} for {next_limit} chats/mo!</p>
          <button onclick="window.location.href='/plans'" style="display: inline-block; background: white; color: #DC2626; padding: 12px 24px; border-radius: 12px; font-weight: 600; margin-top: 10px; border: none; cursor: pointer;">Upgrade Now</button>
        </div>
        """
        raise HTTPException(status_code=402, detail=upgrade_msg)

    return monthly_chats_used, limits

@router.post("/chat")
async def ai_chat(request: Request, db: AsyncSession = Depends(get_session), current_user = Depends(auth.get_current_user)):
    """
    Proxy text chat to OpenAI. Body: {"message": "hello", "system": "...optional..."}
    Returns assistant text.
    """
    body = await request.json()
    user_message = body.get("message", "")
    system_prompt = body.get("system")

    if not user_message:
        raise HTTPException(status_code=400, detail="message required")

    # Plan-based access control - UPDATED: Use TP for all plans (0 cost for pro/elite via limits)
    await check_monthly_chat_limit(db, current_user)
    monthly_chats_used, limits = await check_monthly_chat_limit(db, current_user)  # This also raises if limit hit

    # Deduct TP if applicable
    tp_cost = limits['tp_cost_chat']
    if tp_cost > 0:
        await spend_trade_points(db, current_user, 'chat', tp_cost)

    # NEW: Log for debugging
    normalized_plan = current_user.plan.lower().split('_')[0] if '_' in current_user.plan.lower() else current_user.plan.lower()
    valid_plans = {'starter', 'pro', 'elite'}
    if normalized_plan not in valid_plans:
        normalized_plan = 'starter'
    logger.info(f"User {current_user.id} plan: '{current_user.plan}' -> normalized: '{normalized_plan}', limits: {limits}")

    # Fetch user insights for context
    insights_dict = await compute_insights(current_user, db)
    insights_summary = json.dumps(insights_dict, default=str, ensure_ascii=False)

    # NEW: Fetch active marketplace subscriptions and summarize subscribed traders' recent trades
    stmt = select(Subscription).where(
        Subscription.user_id == current_user.id,
        Subscription.trader_id.is_not(None),
        Subscription.status == 'active'
    )
    result = await db.execute(stmt)
    subscriptions = result.scalars().all()

    sub_trades_summary = ""
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    for sub in subscriptions:
        trader = await db.get(User, sub.trader_id)
        if trader:
            # Use cached win_rate from trader profile
            cached_win_rate = trader.win_rate
            # Fetch recent trades for additional summary (last 20 trades in last 30 days)
            stmt_trades = select(Trade).where(
                Trade.owner_id == sub.trader_id,
                Trade.trade_date >= thirty_days_ago
            ).order_by(Trade.trade_date.desc()).limit(20)
            result_trades = await db.execute(stmt_trades)
            trades = result_trades.scalars().all()
            if trades:
                wins = sum(1 for t in trades if (t.pnl or 0) > 0)
                recent_win_rate = (wins / len(trades)) * 100
                avg_pnl = sum((t.pnl or 0) for t in trades) / len(trades)
                total_pnl = sum((t.pnl or 0) for t in trades)
                sub_trades_summary += (
                    f"\nTrader {trader.username} (Cached Win Rate: {cached_win_rate:.1f}%): "
                    f"{len(trades)} recent trades, Recent Win Rate: {recent_win_rate:.1f}%, "
                    f"Avg PnL: {avg_pnl:.2f}%, Total Recent PnL: {total_pnl:.2f}%. "
                    f"Strategy: {trader.strategy or 'N/A'}."
                )
            else:
                sub_trades_summary += f"\nTrader {trader.username} (Cached Win Rate: {cached_win_rate:.1f}%): No recent trades. Strategy: {trader.strategy or 'N/A'}."

    # Enhanced system prompt with insights and subscribed traders' info - STRONGER NO-MARKDOWN ENFORCEMENT
    enhanced_system = (
        f"You are a helpful trading assistant for iTrade Journal. Provide concise, actionable advice based on trading psychology, "
        f"risk management, and strategy analysis. Reference the user's profile, own insights, and subscribed traders' performance below when relevant. "
        "IMPORTANT: Respond in PLAIN TEXT ONLY. NEVER use Markdown formatting such as **bold**, *italics*, # headers, - lists (use numbered lists for steps), or any other symbols for emphasis. "
        "For emphasis, use ALL CAPS or simple repetition sparingly. Use numbered lists (1. 2. etc.) for steps if needed. "
        "Keep responses under 200 words. Do not include any code blocks, quotes, or special characters for formatting.\n\n"
        f"User Insights: {insights_summary}\n\n"
        f"Subscribed Traders' Recent Performance: {sub_trades_summary}\n\n"
        f"{system_prompt or ''}"
    )

    messages = []
    if enhanced_system:
        messages.append({"role": "system", "content": enhanced_system})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"),
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 800
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(OPENAI_CHAT_URL, headers=HEADERS, json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"OpenAI chat error: {resp.text}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return JSONResponse({"reply": content, "raw": data})

@router.post("/voice/transcribe")
async def transcribe_audio(file: UploadFile = File(...), current_user = Depends(auth.get_current_user), db: AsyncSession = Depends(get_session)):
    """
    Accept multipart file upload and return transcription text.
    Use small files for now. For long files, chunk or use Realtime.
    UPDATED: Deduct TP for voice chat (treat as chat session)
    """
    # Plan-based access control - UPDATED: Check monthly limit and deduct TP for voice (counts as chat)
    await check_monthly_chat_limit(db, current_user)
    monthly_chats_used, limits = await check_monthly_chat_limit(db, current_user)  # This also raises if limit hit

    # Deduct TP if applicable
    tp_cost = limits['tp_cost_chat']
    if tp_cost > 0:
        await spend_trade_points(db, current_user, 'chat', tp_cost)

    # read bytes
    audio_bytes = await file.read()
    # prepare multipart for OpenAI
    multipart = {
        "file": (file.filename, audio_bytes, file.content_type or "application/octet-stream"),
        "model": (None, "whisper-1")
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(OPENAI_TRANSCRIBE_URL, headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}, files=multipart)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Transcription failed: {resp.text}")
        data = resp.json()
        # OpenAI returns 'text' key
        text = data.get("text") or data.get("transcript") or ""
        return JSONResponse({"text": text, "raw": data})

@router.post("/voice/speak")
async def text_to_speech(payload: dict, current_user = Depends(auth.get_current_user), db: AsyncSession = Depends(get_session)):
    """
    Convert text to speech. Body: {"text":"hello", "voice":"alloy", "format":"mp3"}
    Returns audio stream.
    UPDATED: Deduct TP for TTS (treat as part of chat session; assume called after chat)
    """
    # No additional deduction for TTS (bundled with chat/transcribe)

    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    voice = payload.get("voice", "alloy")
    response_format = payload.get("format", "mp3")  # mp3 or wav

    post_json = {
        "model": getattr(settings, "OPENAI_TTS_MODEL", "tts-1"),
        "input": text,
        "voice": voice,
        "response_format": response_format
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(OPENAI_TTS_URL, headers={**HEADERS, "Content-Type": "application/json"}, json=post_json)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"TTS failed: {resp.text}")

        audio_bytes = resp.content
        media_type = "audio/mpeg" if response_format == "mp3" else "audio/wav"
        return StreamingResponse(io.BytesIO(audio_bytes), media_type=media_type)
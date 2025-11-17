# server/router/ai.py
import io
import os
import json
from typing import Optional, List
from fastapi import APIRouter, Depends, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update, desc
from datetime import datetime, timedelta
import httpx
import logging

import auth
from database import get_session
from router.insights import compute_insights
from models.models import User, Subscription, Trade, PointTransaction, AiChatLimits, AiChatMessage  # <-- NEW
from config import get_settings

settings = get_settings()
logger = logging.getLogger("iTrade")

router = APIRouter(prefix="/ai", tags=["AI"])

OPENAI_API_KEY = getattr(settings, "OPENAI_API_KEY", None)
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY required for AI endpoints")

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"

HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

# === POINT & LIMIT HELPERS (unchanged) ===
async def spend_trade_points(db: AsyncSession, user: User, action: str, amount: int = 1) -> None:
    if user.trade_points < amount:
        raise HTTPException(status_code=402, detail=f"Insufficient Trade Points ({user.trade_points} remaining). Refer friends to earn more!")
    user.trade_points -= amount
    tx = PointTransaction(user_id=user.id, type=action, amount=-amount, description=f"Spent {amount} TP on {action}")
    db.add(tx)
    await db.commit()

async def check_monthly_chat_limit(db: AsyncSession, user: User) -> tuple[int, dict]:
    normalized_plan = user.plan.lower().split('_')[0] if '_' in user.plan.lower() else user.plan.lower()
    valid_plans = {'starter', 'pro', 'elite'}
    if normalized_plan not in valid_plans:
        logger.warning(f"Unrecognized plan '{user.plan}' for user {user.id}, defaulting to 'starter'")
        normalized_plan = 'starter'

    result = await db.execute(select(AiChatLimits).where(AiChatLimits.plan == normalized_plan))
    db_limit = result.scalar_one_or_none()
    defaults = {'starter': {'monthly_chat_limit': 5, 'tp_cost_chat': 1}, 'pro': {'monthly_chat_limit': 25, 'tp_cost_chat': 0}, 'elite': {'monthly_chat_limit': 50, 'tp_cost_chat': 0}}
    limits = {'monthly_chat_limit': db_limit.monthly_limit, 'tp_cost_chat': db_limit.tp_cost} if db_limit else defaults.get(normalized_plan, defaults['starter'])

    now = datetime.utcnow()
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    chat_count = (await db.execute(select(func.count(PointTransaction.id)).where(
        PointTransaction.user_id == user.id,
        PointTransaction.type == 'chat',
        PointTransaction.created_at >= start_of_month
    ))).scalar() or 0

    if chat_count >= limits['monthly_chat_limit']:
        next_plan = 'pro' if normalized_plan == 'starter' else 'elite'
        next_limit = defaults[next_plan]['monthly_chat_limit']
        upgrade_msg = f"""
        <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#EF4444,#DC2626);color:white;border-radius:18px;margin:10px 0;">
          <p><strong>Unlock More Conversations</strong></p>
          <p>You've used your {limits['monthly_chat_limit']} chats this month. Upgrade to {next_plan.title()} for {next_limit} chats/mo!</p>
          <button onclick="window.location.href='/plans'" style="background:white;color:#DC2626;padding:12px 24px;border-radius:12px;font-weight:600;margin-top:10px;border:none;cursor:pointer;">Upgrade Now</button>
        </div>
        """
        raise HTTPException(status_code=402, detail=upgrade_msg)

    return chat_count, limits

# === NEW: Save message to DB ===
async def save_chat_message(db: AsyncSession, user_id: int, role: str, content: str):
    msg = AiChatMessage(user_id=user_id, role=role, content=content)
    db.add(msg)
    await db.commit()

# === NEW: Load chat history (last 20 messages) ===
async def get_chat_history(db: AsyncSession, user_id: int) -> List[dict]:
    result = await db.execute(
        select(AiChatMessage).where(AiChatMessage.user_id == user_id)
        .order_by(AiChatMessage.created_at.desc())
        .limit(20)
    )
    messages = result.scalars().all()
    # Reverse to chronological order
    return [{"role": m.role, "content": m.content} for m in reversed(messages)]

# === MAIN CHAT ENDPOINT (NOW WITH PERSISTENT HISTORY) ===
@router.post("/chat")
async def ai_chat(
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user = Depends(auth.get_current_user)
):
    body = await request.json()
    user_message = body.get("message", "").strip()
    system_prompt = body.get("system")

    if not user_message:
        raise HTTPException(400, "message required")

    # 1. Check limits & deduct TP
    await check_monthly_chat_limit(db, current_user)
    _, limits = await check_monthly_chat_limit(db, current_user)
    if limits['tp_cost_chat'] > 0:
        await spend_trade_points(db, current_user, 'chat', limits['tp_cost_chat'])

    # 2. Save user message
    await save_chat_message(db, current_user.id, "user", user_message)

    # 3. Load recent history
    history = await get_chat_history(db, current_user.id)

    # 4. Build insights & subscriptions context
    insights_dict = await compute_insights(current_user, db)
    insights_summary = json.dumps(insights_dict, default=str, ensure_ascii=False)

    # NEW: Fetch and summarize user's recent trades (last 30 days, top 20)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    stmt_user_trades = select(Trade).where(
        Trade.owner_id == current_user.id,
        Trade.trade_date >= thirty_days_ago
    ).order_by(Trade.trade_date.desc()).limit(20)
    trades_result = await db.execute(stmt_user_trades)
    user_trades = trades_result.scalars().all()
    user_trades_summary = ""
    if user_trades:
        wins = sum(1 for t in user_trades if (t.pnl or 0) > 0)
        recent_win_rate = (wins / len(user_trades)) * 100 if user_trades else 0
        avg_pnl = sum((t.pnl or 0) for t in user_trades) / len(user_trades) if user_trades else 0
        total_pnl = sum((t.pnl or 0) for t in user_trades)
        user_trades_summary = (
            f"Your Recent Trades (last 30 days): {len(user_trades)} total, "
            f"Win Rate: {recent_win_rate:.1f}%, Avg PnL: {avg_pnl:.2f}%, Total PnL: {total_pnl:.2f}%.\n"
        )
        # Append brief details for last 5 trades (symbol, direction, PnL, notes snippet)
        last_five = user_trades[:5]
        user_trades_summary += "Recent 5:\n" + "\n".join([
            f"- {t.symbol} {t.direction.value if t.direction else 'N/A'} | PnL: "
            f"{t.pnl:.2f if t.pnl is not None else 'N/A'}% | "
            f"Notes: {t.notes[:50] + '...' if t.notes and len(t.notes) > 50 else (t.notes or 'No notes')}"
            for t in last_five
        ]) + "\n"
    else:
        user_trades_summary = "No recent trades logged in the last 30 days.\n"

    stmt = select(Subscription).where(Subscription.user_id == current_user.id, Subscription.trader_id.is_not(None), Subscription.status == 'active')
    result = await db.execute(stmt)
    subscriptions = result.scalars().all()
    sub_trades_summary = ""
    for sub in subscriptions:
        trader = await db.get(User, sub.trader_id)
        if trader:
            stmt_trades = select(Trade).where(Trade.owner_id == sub.trader_id, Trade.trade_date >= thirty_days_ago).order_by(Trade.trade_date.desc()).limit(20)
            trades_result = await db.execute(stmt_trades)
            trades = trades_result.scalars().all()
            if trades:
                wins = sum(1 for t in trades if (t.pnl or 0) > 0)
                recent_win_rate = (wins / len(trades)) * 100
                avg_pnl = sum((t.pnl or 0) for t in trades) / len(trades)
                total_pnl = sum((t.pnl or 0) for t in trades)
                sub_trades_summary += f"\nTrader {trader.username} (Win Rate: {trader.win_rate:.1f}%): {len(trades)} trades, Recent WR: {recent_win_rate:.1f}%, Avg PnL: {avg_pnl:.2f}%, Total: {total_pnl:.2f}%."
            else:
                sub_trades_summary += f"\nTrader {trader.username} (Win Rate: {trader.win_rate:.1f}%): No recent trades."

    # 5. Build system prompt: CORE RULES FIRST, then context/custom last
    logger.info(f"Custom system prompt from frontend: {system_prompt}")  # Debug log for custom prompts
    enhanced_system = (
        "You are a helpful trading assistant for iTrade Journal (@_bigvik's app). "
        "CRITICAL RULE #1: Respond in PLAIN TEXT ONLY. NEVER use Markdown, bold, italics, or formatting. "
        "CRITICAL RULE #2: Use normal sentence case with proper capitalization and punctuation—ABSOLUTELY NO ALL CAPS, even for emphasis, excitement, or to match user energy. Stay calm and professional. Examples: 'What's up? How can I help with your trades today?' NOT 'WHAT'S UP?'. 'Not much, just ready to dive into your trading questions.' NOT 'NOT MUCH, JUST READY TO DIVE IN!!!'. "
        "Keep responses concise, actionable, under 200 words. Give trading advice based on context.\n\n"
        f"User Insights: {insights_summary}\n\n"
        f"{user_trades_summary}\n\n"
        f"Subscribed Traders: {sub_trades_summary}\n\n"
        f"{system_prompt or ''}\n\n"  # Custom prompt LAST, after rules
    )

    # 6. Build messages list (unchanged)
    messages = [{"role": "system", "content": enhanced_system}]
    messages.extend(history[-19:])  # last 19 messages (leave room for new response)
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"),
        "messages": messages,
        "temperature": 0.1,  # Drop to 0.1 for even stricter adherence
        "max_tokens": 800
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(OPENAI_CHAT_URL, headers=HEADERS, json=payload)
        if resp.status_code != 200:
            raise HTTPException(502, f"OpenAI error: {resp.text}")
        data = resp.json()
        assistant_reply = data["choices"][0]["message"]["content"]

    logger.info(f"Raw OpenAI reply: {assistant_reply[:200]}...")  # Debug log for reply

    # 7. Save assistant reply
    await save_chat_message(db, current_user.id, "assistant", assistant_reply)

    return JSONResponse({
        "reply": assistant_reply,
        "history_saved": True,
        "raw": data
    })


@router.get("/history")
async def get_ai_history(
    db: AsyncSession = Depends(get_session),
    current_user = Depends(auth.get_current_user)
):
    history = await get_chat_history(db, current_user.id)
    return JSONResponse({
        "messages": history,
        "has_history": len(history) > 0
    })
    

# === VOICE ENDPOINTS (unchanged except TP deduction) ===
@router.post("/voice/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    current_user = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    await check_monthly_chat_limit(db, current_user)
    _, limits = await check_monthly_chat_limit(db, current_user)
    if limits['tp_cost_chat'] > 0:
        await spend_trade_points(db, current_user, 'chat', limits['tp_cost_chat'])

    audio_bytes = await file.read()
    multipart = {
        "file": (file.filename, audio_bytes, file.content_type or "application/octet-stream"),
        "model": (None, "whisper-1")
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(OPENAI_TRANSCRIBE_URL, headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}, files=multipart)
        if resp.status_code != 200:
            raise HTTPException(502, f"Transcription failed: {resp.text}")
        data = resp.json()
        text = data.get("text", "")
        await save_chat_message(db, current_user.id, "user", f"[VOICE] {text}")
        return JSONResponse({"text": text, "raw": data})

@router.post("/voice/speak")
async def text_to_speech(
    payload: dict,
    current_user = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_session)
):
    text = payload.get("text", "")
    if not text:
        raise HTTPException(400, "text required")
    voice = payload.get("voice", "alloy")
    fmt = payload.get("format", "mp3")

    post_json = {
        "model": getattr(settings, "OPENAI_TTS_MODEL", "tts-1"),
        "input": text,
        "voice": voice,
        "response_format": fmt
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(OPENAI_TTS_URL, headers={**HEADERS, "Content-Type": "application/json"}, json=post_json)
        if resp.status_code != 200:
            raise HTTPException(502, f"TTS failed: {resp.text}")
        audio_bytes = resp.content
        media_type = "audio/mpeg" if fmt == "mp3" else "audio/wav"
        return StreamingResponse(io.BytesIO(audio_bytes), media_type=media_type)
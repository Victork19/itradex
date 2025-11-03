# server/router/ai.py
import io
import os
import json
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
import httpx

from config import settings
import auth
from database import get_session
from router.insights import compute_insights
from models.models import User

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

    # Plan-based access control for starters - updated to handle variations like pro_monthly, elite_yearly
    plan_lower = current_user.plan.lower()
    if not any(term in plan_lower for term in ['pro', 'elite']):
        # Starter: Allow only 1 chat
        if current_user.ai_chats_used >= 1:
            upgrade_msg = """
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); color: white; border-radius: 18px; margin: 10px 0;">
              <p><strong>Unlock Unlimited Companionship</strong></p>
              <p>You've used your free chat. Upgrade to Pro for unlimited access to your personal trading coach!</p>
              <a href="/plans" style="display: inline-block; background: white; color: #DC2626; padding: 12px 24px; border-radius: 12px; text-decoration: none; font-weight: 600; margin-top: 10px;">Upgrade Now</a>
            </div>
            """
            return JSONResponse({"reply": upgrade_msg, "is_upgrade": True})
        else:
            # Grant the first chat
            current_user.ai_chats_used = 1
            current_user.updated_at = datetime.utcnow()
            await db.commit()

    # Fetch user insights for context
    insights_dict = await compute_insights(current_user, db)
    insights_summary = json.dumps(insights_dict, default=str, ensure_ascii=False)

    # Enhanced system prompt with insights - STRONGER NO-MARKDOWN ENFORCEMENT
    enhanced_system = (
        f"You are a helpful trading assistant for iTrade Journal. Provide concise, actionable advice based on trading psychology, "
        f"risk management, and strategy analysis. Reference the user's profile and insights below when relevant. "
        "IMPORTANT: Respond in PLAIN TEXT ONLY. NEVER use Markdown formatting such as **bold**, *italics*, # headers, - lists (use numbered lists for steps), or any other symbols for emphasis. "
        "For emphasis, use ALL CAPS or simple repetition sparingly. Use numbered lists (1. 2. etc.) for steps if needed. "
        "Keep responses under 200 words. Do not include any code blocks, quotes, or special characters for formatting.\n\n"
        f"User Insights: {insights_summary}\n\n"
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
async def transcribe_audio(file: UploadFile = File(...), current_user = Depends(auth.get_current_user)):
    """
    Accept multipart file upload and return transcription text.
    Use small files for now. For long files, chunk or use Realtime.
    """
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
async def text_to_speech(payload: dict, current_user = Depends(auth.get_current_user)):
    """
    Convert text to speech. Body: {"text":"hello", "voice":"alloy", "format":"mp3"}
    Returns audio stream.
    """
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
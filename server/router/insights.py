import io
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import asyncio
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel

# Local imports (adjust as needed)
import models.models as models
import auth
from database import get_session
from config import settings

# Templates setup
templates = Jinja2Templates(directory="templates")  # Assume templates dir

# OpenAI setup
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
    logger.warning("openai lib not available; install 'openai' for AI insights")

OPENAI_MODEL = getattr(settings, "OPENAI_MODEL", "gpt-4o-mini")
MAX_TRADES_FOR_ANALYSIS = getattr(settings, "MAX_TRADES_FOR_ANALYSIS", 500)

if not HAS_OPENAI_LIB:
    raise RuntimeError("openai lib required for AI insights")

# Structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights", tags=["Insights"])

class InsightsResponse(BaseModel):
    total_trades: int
    ai_insights: Dict[str, Any]
    insights_based_on: Optional[int] = None
    credits: int = 0

# JSON Schema for structured OpenAI output
INSIGHTS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "strengths": {"type": "array", "items": {"type": "string", "description": "Descriptive strength, e.g., 'Your Sydney session win rate is 65%—leverage this for crypto scalps.'"}},
        "weaknesses": {"type": "array", "items": {"type": "string", "description": "Descriptive weakness, e.g., 'Tokyo trades average 4.2 per session—cap at 3 to align with risk allowance.'"}},
        "actions": {"type": "array", "items": {"type": "string", "description": "Actionable step, e.g., 'Cap trades at 3 per session.'"}},
        "alerts": {"type": "array", "items": {"type": "string", "description": "Alert message, e.g., '3 risk breaches this week.'"}},
        "recommendations": {"type": "string", "description": "Overall recommendation summary."},
        "session_insights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "session": {"type": "string"},
                    "insight": {"type": "string"}
                },
                "required": ["session", "insight"],
                "additionalProperties": False
            },
            "additionalProperties": False
        }
    },
    "required": ["strengths", "weaknesses", "actions", "alerts", "recommendations", "session_insights"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a senior trading analyst. Analyze the provided trade stats for patterns, strengths, weaknesses, alerts, and actionable recommendations. "
    "For each session, generate a short insight based on its stats. "
    "Output descriptive full sentences for strengths, weaknesses, and actions. "
    "Respond ONLY with valid JSON matching this exact schema (no extra fields or text):\n"
    f"{json.dumps(INSIGHTS_JSON_SCHEMA, indent=2)}"
)

async def get_stored_insights(user_id: int, db: AsyncSession, total_trades: Optional[int] = None) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    """Get latest stored insights for user (optionally filtered by total_trades). Returns (insights_dict, trades_count_at_save)."""
    if total_trades is not None and total_trades == 0:
        return None, None
    try:
        stmt = select(models.TradeInsight).where(models.TradeInsight.user_id == user_id)
        if total_trades is not None:
            stmt = stmt.where(models.TradeInsight.total_trades == total_trades)
        stmt = stmt.order_by(desc(models.TradeInsight.created_at)).limit(1)
        result = await db.execute(stmt)
        insight = result.scalar_one_or_none()
        if insight:
            return json.loads(insight.insights_json), insight.total_trades
    except Exception as e:
        logger.error("Stored insights fetch failed: %s", e)
    return None, None

async def has_any_insights(user_id: int, db: AsyncSession) -> bool:
    """Check if user has any stored insights."""
    try:
        stmt = select(func.count()).select_from(models.TradeInsight).where(models.TradeInsight.user_id == user_id)
        result = await db.execute(stmt)
        return result.scalar() > 0
    except Exception as e:
        logger.error("Failed to check for any insights: %s", e)
        return False

async def save_insights_to_db(user_id: int, db: AsyncSession, total_trades: int, ai_part: Dict[str, Any]):
    try:
        new_insight = models.TradeInsight(
            user_id=user_id,
            total_trades=total_trades,
            insights_json=json.dumps(ai_part)
        )
        db.add(new_insight)
        await db.commit()
        await db.refresh(new_insight)
        logger.info("Saved insights to DB for user %d, total_trades=%d", user_id, total_trades)
    except Exception as e:
        logger.error("Failed to save insights to DB: %s", e)
        await db.rollback()

def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    # Expanded formats: add US, with/without time
    formats = [
        '%Y-%m-%d',           # 2025-10-19
        '%d/%m/%Y',           # 19/10/2025
        '%d/%m/%y',           # 19/10/25
        '%m/%d/%Y',           # 10/19/2025 (US)
        '%m/%d/%y',           # 10/19/25 (US)
        '%d-%m-%Y',           # 19-10-2025
        '%Y-%m-%d %H:%M:%S',  # 2025-10-19 14:30:00
        '%d/%m/%Y %H:%M:%S',  # 19/10/2025 14:30:00
        '%m/%d/%Y %H:%M:%S',  # 10/19/2025 14:30:00
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    logger.warning(f"Unparseable date: {date_str}")
    return None

def get_trade_datetime(trade) -> Optional[datetime]:
    """Get parsed datetime from trade_date or fallback to created_at."""
    parsed_trade_date = None
    if trade.trade_date:
        if isinstance(trade.trade_date, str):
            parsed_trade_date = parse_date(trade.trade_date)
        else:
            parsed_trade_date = trade.trade_date  # Already datetime
    if parsed_trade_date:
        return parsed_trade_date
    
    # Always fallback to created_at if available (prioritize over bad trade_date)
    if hasattr(trade, 'created_at') and trade.created_at:
        if isinstance(trade.created_at, str):
            return parse_date(trade.created_at)
        return trade.created_at
    
    logger.warning(f"No valid date for trade {getattr(trade, 'id', 'unknown')}")
    return None

async def fetch_user_trades(db: AsyncSession, user: models.User, limit: int = MAX_TRADES_FOR_ANALYSIS) -> List[models.Trade]:
    """Flexible fetch with owner/user_id support, ordered by date desc."""
    owner_field = None
    stmt = None
    if hasattr(models.Trade, "owner_id"):
        owner_field = models.Trade.owner_id
        stmt = select(models.Trade).where(owner_field == user.id)
    elif hasattr(models.Trade, "user_id"):
        owner_field = models.Trade.user_id
        stmt = select(models.Trade).where(owner_field == user.id)
    elif hasattr(models.Trade, "owner"):
        # Relationship query
        stmt = select(models.Trade).join(models.Trade.owner).where(models.Trade.owner == user)
    elif hasattr(models.Trade, "user"):
        # Relationship query for 'user'
        stmt = select(models.Trade).join(models.Trade.user).where(models.Trade.user == user)
    else:
        raise ValueError("Trade model lacks user linkage")

    # Order by created_at since it's always available and reliable
    order_col = models.Trade.created_at
    stmt = stmt.order_by(desc(order_col)).limit(limit)

    try:
        result = await db.execute(stmt)
        return result.scalars().all()
    except SQLAlchemyError as e:
        logger.error("Trade fetch failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to fetch trades")

async def get_total_trades_count(db: AsyncSession, user: models.User) -> int:
    """Get exact total trades count (no limit)."""
    owner_field = None
    stmt = None
    if hasattr(models.Trade, "owner_id"):
        owner_field = models.Trade.owner_id
        stmt = select(func.count()).where(models.Trade.owner_id == user.id)
    elif hasattr(models.Trade, "user_id"):
        owner_field = models.Trade.user_id
        stmt = select(func.count()).where(models.Trade.user_id == user.id)
    elif hasattr(models.Trade, "owner"):
        stmt = select(func.count(models.Trade.id)).join(models.Trade.owner).where(models.Trade.owner == user)
    elif hasattr(models.Trade, "user"):
        stmt = select(func.count(models.Trade.id)).join(models.Trade.user).where(models.Trade.user == user)
    else:
        raise ValueError("Trade model lacks user linkage")

    try:
        result = await db.execute(stmt)
        return result.scalar() or 0
    except SQLAlchemyError as e:
        logger.error("Total trades count failed: %s", e)
        return 0

def compute_advanced_metrics(pnls: List[float]) -> Dict[str, Any]:
    """Compute Sharpe ratio, max drawdown, symbol allocation."""
    if not pnls:
        return {"sharpe_ratio": None, "max_drawdown": None, "symbol_allocation": {}}
    
    # Simple Sharpe (risk-free rate=0, std dev of PnL)
    mean_pnl = sum(pnls) / len(pnls)
    std_pnl = (sum((x - mean_pnl) ** 2 for x in pnls) / len(pnls)) ** 0.5
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0
    
    # Max drawdown (cumulative)
    cum_pnl = [sum(pnls[:i+1]) for i in range(len(pnls))]
    running_max = 0
    max_dd = 0
    for cum in cum_pnl:
        running_max = max(running_max, cum)
        dd = (running_max - cum) / running_max if running_max > 0 else 0
        max_dd = max(max_dd, dd)
    
    return {"sharpe_ratio": round(sharpe, 2), "max_drawdown": round(max_dd * 100, 2), "symbol_allocation": {}}

def infer_session_from_time(trade_date: datetime) -> Optional[str]:
    """Infer trading session from UTC hour."""
    if not trade_date:
        return None
    hour = trade_date.hour
    if hour >= 22 or hour < 7:
        return 'sydney'
    elif 0 <= hour < 9:
        return 'tokyo'
    elif 8 <= hour < 17:
        return 'london'
    elif 13 <= hour < 22:
        return 'newyork'
    return None

def get_upgrade_insights(session_data: List[Dict], total_trades: int) -> Dict[str, Any]:
    for sd in session_data:
        sd['insight'] = f"Upgrade to Pro for AI insights on your {sd['session']} performance."
    return {
        "strengths": [f"Pro users get personalized strengths from {total_trades} trades."],
        "weaknesses": ["Limited access without Pro—upgrade to see potential risks."],
        "actions": ["Upgrade to Pro for unlimited generations and full analysis."],
        "alerts": ["Insights generations exhausted—consider Pro for full access."],
        "recommendations": "Upgrade to access comprehensive AI recommendations and unlock your trading potential.",
        "session_insights": [],
        "bullets": ["Upgrade now for unlimited insights.", "Get advanced equity projections.", "Unlock detailed strategy breakdowns."]
    }

async def call_openai_structured(messages: List[Dict[str, str]], system_prompt: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    # Switch to json_object for reliability (less strict than json_schema)
    response_format = {"type": "json_object"}
    
    try:
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=full_messages,
            response_format=response_format,
            temperature=0.1,
            max_tokens=800,  # Increased for fuller session_insights
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        # Fill missing keys with defaults to ensure robust output
        for key in ["strengths", "weaknesses", "actions", "alerts", "session_insights"]:
            if key not in parsed:
                parsed[key] = []
        if "recommendations" not in parsed:
            parsed["recommendations"] = ""
        logger.info("OpenAI response parsed successfully")
        return parsed
    except json.JSONDecodeError as e:
        logger.error("OpenAI JSON parse failed: %s", e)
        return None
    except Exception as e:
        logger.error("OpenAI call failed: %s", e)
        return None

@router.get("", response_class=HTMLResponse)
async def get_insights_page(
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user),
):
    now = datetime.utcnow()
    
    # Initialize credits for new starter users ONLY if they've never generated before
    plan = getattr(current_user, 'plan', 'free')
    if plan == 'starter':
        credits = getattr(current_user, 'insights_credits', 0) or 0
        if credits == 0:
            has_generated = await has_any_insights(current_user.id, db)
            if not has_generated:
                current_user.insights_credits = 1
                await db.commit()
                await db.refresh(current_user)
                logger.info("Initialized 1 insight credit for new starter user %s (plan: %s)", current_user.id, plan)

    try:
        insights_dict = await compute_insights(current_user, db)
    except Exception as e:
        logger.error(f"Error computing insights: {e}")
        insights_dict = {"total_trades": 0, "ai_insights": {}, "insights_based_on": None, "credits": getattr(current_user, 'insights_credits', 0)}
    return templates.TemplateResponse("insights.html", {"request": request, "insights": insights_dict, "current_user": current_user, "now": now})

async def compute_insights(current_user: models.User, db: AsyncSession, prompt: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    credits = getattr(current_user, 'insights_credits', 0)
    total_trades = await get_total_trades_count(db, current_user)  # Exact count, no limit
    try:
        trades = await fetch_user_trades(db, current_user)  # Limited for analysis
        analysis_trades_count = len(trades)  # For equity/aggregates (capped)
        if total_trades == 0:
            empty_data = {"total_trades": 0, "ai_insights": {}, "insights_based_on": None, "credits": credits}
            return empty_data

        is_initial = prompt is None and not force
        ai_part = None
        stored = None
        old_total = None
        if is_initial:
            stored, old_total = await get_stored_insights(current_user.id, db)  # Latest, no total filter
            if stored:
                ai_part = stored
                analysis_trades_count = old_total or analysis_trades_count  # Use old if available

        # Only compute full insights if we have stored or are generating/regenerating
        if ai_part is not None or not is_initial:
            # Use account_balance from user model as starting balance
            account_balance = getattr(current_user, 'account_balance', 10000.0)
            initial_balance = account_balance

            # Compute basic metrics always (fresh)
            # Parse dates
            for t in trades:
                t.parsed_date = get_trade_datetime(t)
            parsed_trades = [t for t in trades if t.parsed_date is not None]

            # Equity curve computation (multiplicative for % PnL)
            if not parsed_trades:
                current_date = datetime.now()
                labels = [current_date.strftime('%Y-%m-%d')]
                datasets = {sess: {'data': [initial_balance]} for sess in ['all', 'sydney', 'tokyo', 'london', 'newyork']}
                equity_curve = {'labels': labels, 'datasets': datasets}
            else:
                # Sort trades by date
                parsed_trades.sort(key=lambda t: t.parsed_date)

                # Daily groups for all trades
                daily_groups_all = defaultdict(list)
                for t in parsed_trades:
                    day = t.parsed_date.date()
                    daily_groups_all[day].append(t)

                # Session daily groups
                session_map = {
                    'new york': 'newyork',
                    'newyork': 'newyork',
                    'new_york': 'newyork',
                    'ny': 'newyork'
                }
                session_daily = defaultdict(lambda: defaultdict(list))
                unknown_count = 0
                for t in parsed_trades:
                    day = t.parsed_date.date()
                    if t.session:
                        norm_session = session_map.get(t.session.lower().replace(' ', '_'), t.session.lower().replace(' ', '_'))
                    else:
                        inferred = infer_session_from_time(t.parsed_date)
                        norm_session = inferred or 'unknown'
                        if norm_session == 'unknown':
                            unknown_count += 1
                    session_daily[norm_session][day].append(t)

                # Sessions to include
                sessions = ['sydney', 'tokyo', 'london', 'newyork']
                if unknown_count > 0:
                    sessions.append('unknown')

                # Date range
                min_date = min(t.parsed_date for t in parsed_trades)
                max_date = max(t.parsed_date for t in parsed_trades)
                delta_days = (max_date - min_date).days + 1
                all_days = [min_date + timedelta(days=i) for i in range(delta_days)]
                labels = [d.strftime('%Y-%m-%d') for d in all_days]

                # Datasets
                datasets = {}

                # For 'all'
                current_cum = initial_balance
                equity_data_all = []
                for day_obj in all_days:
                    day = day_obj.date()
                    day_mult = 1.0
                    if day in daily_groups_all:
                        for t in daily_groups_all[day]:
                            pnl_pct = (t.pnl or 0) / 100.0
                            day_mult *= (1 + pnl_pct)
                    current_cum *= day_mult
                    equity_data_all.append(round(current_cum, 2))
                datasets['all'] = {'data': equity_data_all}

                # For each session
                for sess in sessions:
                    current_cum_s = initial_balance
                    equity_data_s = []
                    for day_obj in all_days:
                        day = day_obj.date()
                        day_mult = 1.0
                        if day in session_daily[sess]:
                            for t in session_daily[sess][day]:
                                pnl_pct = (t.pnl or 0) / 100.0
                                day_mult *= (1 + pnl_pct)
                        current_cum_s *= day_mult
                        equity_data_s.append(round(current_cum_s, 2))
                    datasets[sess] = {'data': equity_data_s}

                equity_curve = {'labels': labels, 'datasets': datasets}

            # Session data (aggregates)
            session_groups = defaultdict(list)
            for t in trades:
                if t.session:
                    norm_session = session_map.get(t.session.lower().replace(' ', '_'), t.session.lower().replace(' ', '_'))
                    session_groups[norm_session].append(t)
                else:
                    inferred = infer_session_from_time(t.parsed_date)
                    if inferred:
                        session_groups[inferred].append(t)
                    else:
                        session_groups['unknown'].append(t)

            session_data = []
            display_map = {
                'sydney': 'Sydney',
                'tokyo': 'Tokyo',
                'london': 'London',
                'newyork': 'New York',
                'unknown': 'Unknown'
            }
            for s, s_trades in session_groups.items():
                if len(s_trades) == 0:
                    continue
                s_pnl = [t.pnl or 0 for t in s_trades]
                s_win = len([p for p in s_pnl if p > 0]) / len(s_pnl) * 100 if s_pnl else 0
                s_avg = sum(s_pnl) / len(s_pnl) if s_pnl else 0
                fit = 'High' if s_win > 60 else 'Medium' if s_win > 40 else 'Low'
                display_session = display_map.get(s, s.capitalize())
                session_data.append({
                    'session': display_session,
                    'win_rate': round(s_win),
                    'avg_pnl': round(s_avg, 1),
                    'strategy_fit': fit,
                    'insight': ''  # Filled later
                })

            if not session_data:
                all_pnl = [t.pnl or 0 for t in trades]
                if len(trades) > 0:
                    all_win = len([p for p in all_pnl if p > 0]) / len(all_pnl) * 100
                    all_avg = sum(all_pnl) / len(all_pnl)
                    fit = 'High' if all_win > 60 else 'Medium' if all_win > 40 else 'Low'
                    session_data.append({
                        'session': 'All Trades',
                        'win_rate': round(all_win),
                        'avg_pnl': round(all_avg, 1),
                        'strategy_fit': fit,
                        'insight': ''
                    })

            # Strategy distribution
            strategy_groups = defaultdict(list)
            for t in trades:
                if t.strategy:
                    strategy_groups[t.strategy.lower()].append(t)
                else:
                    strategy_groups['unknown'].append(t)
            strategy_counts = {strat: len(group) for strat, group in strategy_groups.items()}
            total_strat = sum(strategy_counts.values())
            colors = ['#2F6BFF', '#22C55E', '#EF4444', '#FACC15', '#94A3B8']
            strategy_dist = []
            i = 0
            for strat, count in strategy_counts.items():
                strategy_dist.append({'name': strat.capitalize(), 'value': round(count / total_strat * 100) if total_strat else 0, 'color': colors[i % len(colors)]})
                i += 1
            if not strategy_dist:
                strategy_dist = [{'name': 'No Strategies', 'value': 100, 'color': '#94A3B8'}]

            # Symbol allocation
            symbol_pnl = defaultdict(float)
            for t in trades:
                if t.symbol and t.pnl is not None:
                    symbol_pnl[t.symbol] += float(t.pnl)
            total_symbol_pnl = sum(symbol_pnl.values())
            symbol_allocation = {sym: round(pnl / total_symbol_pnl * 100, 2) if total_symbol_pnl else 0 for sym, pnl in symbol_pnl.items()}

            # Advanced metrics
            pnls = [t.pnl or 0 for t in parsed_trades]
            advanced = compute_advanced_metrics(pnls)

            # Decide on AI insights
            plan = getattr(current_user, 'plan', 'free')
            is_unlimited = plan in ['pro', 'elite']
            has_generation_right = is_unlimited or (plan == 'starter' and credits > 0)
            prompt_or_force = bool(prompt or force)

            need_generate = False
            generated_new = False
            if ai_part is None:
                # For regenerate/force or first generate
                need_generate = has_generation_right
                if need_generate:
                    # Prepare summary for AI
                    summary = {
                        "total_trades": analysis_trades_count,
                        "sessions": [{k: sd[k] for k in ['session', 'win_rate', 'avg_pnl', 'strategy_fit']} for sd in session_data],
                        "strategies": strategy_dist,
                        "advanced_metrics": advanced
                    }
                    base_prompt = f"Analyze trading summary: {json.dumps(summary)}. Generate structured insights including per-session insights."
                    user_prompt = f"{prompt or ''} {base_prompt}" if prompt else base_prompt

                    messages = [{"role": "user", "content": user_prompt}]

                    ai_parsed = await call_openai_structured(messages, SYSTEM_PROMPT, INSIGHTS_JSON_SCHEMA)

                    if ai_parsed:
                        ai_part = ai_parsed
                        generated_new = True
                        # Deduct credit if starter (only after successful generation)
                        if plan == 'starter':
                            current_user.insights_credits -= 1
                            await db.commit()
                            await db.refresh(current_user)
                            credits = current_user.insights_credits  # Update local
                            logger.info("Deducted 1 insight credit for user %s. Remaining: %d", current_user.id, credits)
                        # Save to DB the AI part
                        await save_insights_to_db(current_user.id, db, total_trades, ai_part)  # Use exact total_trades
                    else:
                        ai_part = get_upgrade_insights(session_data, total_trades)
                        logger.warning(f"OpenAI generation failed for user {current_user.id}: using upgrade fallback, no credit deducted")

            if ai_part is None:
                ai_part = get_upgrade_insights(session_data, total_trades)

            # Map session insights if available (for stored/generated)
            if ai_part and ai_part.get('session_insights'):
                session_insights_map = {si['session'].lower().replace(' ', ''): si['insight'] for si in ai_part['session_insights']}
                for sd in session_data:
                    norm_key = sd['session'].lower().replace(' ', '')
                    sd['insight'] = session_insights_map.get(norm_key, sd['insight'] or 'No AI insight available')

            # Build full ai_insights
            ai_insights = ai_part.copy() if ai_part else {}
            ai_insights['equity_curve'] = equity_curve
            ai_insights['session_data'] = session_data
            ai_insights['strategy_distribution'] = strategy_dist
            ai_insights['bullets'] = ai_insights.get('actions', [])
            ai_insights['advanced_metrics'] = advanced
            ai_insights['symbol_allocation'] = symbol_allocation
            insights_based_on = old_total if old_total is not None else total_trades
        else:
            # Initial load, no stored: empty ai_insights
            ai_insights = {}
            insights_based_on = None

        response_data = {
            "total_trades": total_trades,
            "ai_insights": ai_insights,
            "insights_based_on": insights_based_on,
            "credits": credits
        }
        
        logger.info("Generated/loaded insights for user %s: total_trades=%d, insights_based_on=%s, sessions=%d", 
                    current_user.id, total_trades, insights_based_on, len(session_data) if 'session_data' in locals() else 0)
        
        return response_data
    except SQLAlchemyError as e:
        logger.error(f"DB error in compute_insights (force={force}): {e}")
        raise HTTPException(status_code=500, detail="Database error—check trade dates/sessions")
    except Exception as e:
        logger.error(f"Unexpected error in compute_insights (force={force}, prompt={prompt}): {e}")
        # Graceful fallback to upgrade/empty
        session_data = locals().get('session_data', [])
        return {"total_trades": total_trades, "ai_insights": get_upgrade_insights(session_data, total_trades), "insights_based_on": None, "credits": credits}

@router.get("/data", response_model=InsightsResponse)
async def get_insights_data(
    prompt: Optional[str] = Query(None),
    force: bool = Query(False),
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user),
):
    insights_dict = await compute_insights(current_user, db, prompt=prompt, force=force)
    return InsightsResponse(**insights_dict)

# Additional endpoint for raw trade data export (premium feature)
@router.get("/export", status_code=status.HTTP_200_OK)
async def export_trades(
    format: str = Query("json", regex="^(json|csv)$"),
    include_insights: bool = Query(False),  # New: Optional insights inclusion
    db: AsyncSession = Depends(get_session),
    current_user: models.User = Depends(auth.get_current_user),
):
    """Export user trades and/or insights in JSON or CSV."""
    trades = await fetch_user_trades(db, current_user, limit=10000)  # Higher limit for export
    insights_dict = await compute_insights(current_user, db) if include_insights else {}
    
    if format == "csv":
        import csv
        output = io.StringIO()
        fieldnames = ["symbol", "trade_date", "entry_price", "exit_price", "pnl", "notes"]
        if include_insights:
            # Append insights summary row (or extend as needed)
            fieldnames += ["total_trades", "win_rate", "avg_pnl"]  # Example
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            row = {
                "symbol": t.symbol or "",
                "trade_date": t.trade_date if t.trade_date else "",
                "entry_price": t.entry_price or "",
                "exit_price": t.exit_price or "",
                "pnl": t.pnl or "",
                "notes": t.notes or ""
            }
            if include_insights:
                # Placeholder: Add insights data here if per-trade
                row.update({
                    "total_trades": insights_dict.get("total_trades", ""),
                    "win_rate": insights_dict["ai_insights"].get("session_data", [{}])[0].get("win_rate", "") if insights_dict.get("ai_insights") else "",
                    "avg_pnl": insights_dict["ai_insights"].get("session_data", [{}])[0].get("avg_pnl", "") if insights_dict.get("ai_insights") else ""
                })
            writer.writerow(row)
        
        # If insights only, add a summary row
        if include_insights and not trades:
            writer.writerow({
                "symbol": "SUMMARY",
                "total_trades": insights_dict.get("total_trades", ""),
                # ... extend as needed
            })
        
        return {"content": output.getvalue(), "type": "text/csv"}
    
    # JSON: Include insights if requested
    export_data = {"trades": [t.__dict__ for t in trades]}
    if include_insights:
        export_data["insights"] = insights_dict
    return {"data": export_data, "type": "application/json"}
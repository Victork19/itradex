from pathlib import Path
import logging
from typing import Optional, Dict
from collections import defaultdict
from datetime import datetime, timedelta, date

from fastapi import FastAPI, Request, Depends, Cookie, HTTPException, status, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_

from database import Base, engine, get_session
from router import users, uploads, insights, journal, profile, admin, ai, payments
from templates_config import templates
from models import models
from models.schemas import TradeResponse, ProfileUpdateRequest
import auth
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("iTrade")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(title="iTrade Journal")

app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    logger.info("Mounted static directory: %s", STATIC_DIR)
else:
    logger.warning("Static directory not found: %s", STATIC_DIR)

async def get_current_user_optional(
    access_token: Optional[str] = Cookie(None),
    db: AsyncSession = Depends(get_session)
) -> Optional[models.User]:
    if not access_token:
        return None
    try:
        payload = auth.decode_access_token(access_token)
        user_id: int = int(payload.get("sub"))
        if user_id is None:
            return None
        result = await db.execute(select(models.User).where(models.User.id == user_id))
        user = result.scalars().first()
        if user is None:
            return None
        return user
    except (HTTPException, ValueError):
        return None

async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database models initialized")

@app.on_event("startup")
async def startup_event():
    await init_models()

@app.middleware("http")
async def auth_redirect_middleware(request: Request, call_next):
    protected_paths = ["/dashboard", "/insights", "/profile", "/plans", "/upload", "/journal", "/onboard", "/chat"]
    is_protected = any(request.url.path.startswith(path) for path in protected_paths)
    if is_protected and not request.cookies.get("access_token"):
        logger.info(f"Unauthenticated access to {request.url.path}, redirecting to /")
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    response = await call_next(request)
    return response

@app.get("/", response_class=HTMLResponse)
async def root(
    request: Request,
    current_user: Optional[models.User] = Depends(get_current_user_optional)
):
    context = {
        "request": request,
        "tab": request.query_params.get("tab", "signup"),
        "now": datetime.utcnow()
    }
    if current_user:
        context["is_logged_in"] = True
        context["current_user"] = current_user
    if "success" in request.query_params:
        context["success"] = True
    ref_code = request.query_params.get("ref")
    context["ref_code"] = ref_code
    return templates.TemplateResponse("index.html", context)

@app.get("/onboard", response_class=HTMLResponse)
async def onboard_page(
    request: Request,
    current_user: Optional[models.User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_session)
):
    if not current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    needs_onboarding = not current_user.trading_style
    if not needs_onboarding:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    initials = ""
    if current_user.full_name:
        names = current_user.full_name.split()
        if len(names) >= 2:
            initials = names[0][0].upper() + names[-1][0].upper()
        elif len(names) == 1:
            initials = names[0][0].upper() * 2
    else:
        initials = "U"
    return templates.TemplateResponse(
        "onboarding.html",
        {
            "request": request,
            "current_user": current_user,
            "initials": initials,
            "now": datetime.utcnow()
        }
    )

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    current_user: Optional[models.User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_session)
):
    if not current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    needs_onboarding = not current_user.trading_style
    if needs_onboarding:
        return RedirectResponse(url="/onboard", status_code=status.HTTP_303_SEE_OTHER)
    initials = ""
    if current_user.full_name:
        names = current_user.full_name.split()
        if len(names) >= 2:
            initials = names[0][0].upper() + names[-1][0].upper()
        elif len(names) == 1:
            initials = names[0][0].upper() * 2
    else:
        initials = "U"
    now = datetime.utcnow()
    result = await db.execute(
        select(models.Trade).where(models.Trade.owner_id == current_user.id).order_by(models.Trade.created_at.desc())
    )
    trades = result.scalars().all()
    first_name = current_user.full_name.split()[0] if current_user.full_name else 'Trader'
    greeting = f"Hey {first_name},"
    assumed_capital = 10000
    day_ago = now - timedelta(days=1)
    daily_trades = [t for t in trades if t.created_at >= day_ago]
    daily_pnl = sum(t.pnl or 0 for t in daily_trades)
    daily_percent = round((daily_pnl / assumed_capital * 100), 1) if assumed_capital else 0
    todays_edge = daily_percent
    week_ago = now - timedelta(weeks=1)
    weekly_trades = [t for t in trades if t.created_at >= week_ago]
    weekly_pnl = sum(t.pnl or 0 for t in weekly_trades)
    weekly_percent = round((weekly_pnl / assumed_capital * 100), 1) if assumed_capital else 0
    month_ago = now - timedelta(days=30)
    monthly_trades = [t for t in trades if t.created_at >= month_ago]
    monthly_pnl = sum(t.pnl or 0 for t in monthly_trades)
    monthly_percent = round((monthly_pnl / assumed_capital * 100), 1) if assumed_capital else 0
    all_pnl = sum(t.pnl or 0 for t in trades)
    pnl = {
        'daily': {'value': daily_pnl, 'percent': daily_percent},
        'weekly': {'value': weekly_pnl, 'percent': weekly_percent},
        'monthly': {'value': monthly_pnl, 'percent': monthly_percent},
    }
    def get_history(period):
        if period == 'daily':
            deltas = [timedelta(days=i) for i in range(5)]
            period_delta = timedelta(days=1)
        elif period == 'weekly':
            deltas = [timedelta(weeks=i) for i in range(5)]
            period_delta = timedelta(weeks=1)
        else:
            deltas = [timedelta(days=30 * i) for i in range(5)]
            period_delta = timedelta(days=30)
        history = []
        for delta in reversed(deltas):
            start = now - delta - period_delta
            end = now - delta
            period_trades = [t for t in trades if start <= t.created_at < end]
            history.append(sum(t.pnl or 0 for t in period_trades))
        return history
    pnl_history = {
        'daily': get_history('daily'),
        'weekly': get_history('weekly'),
        'monthly': get_history('monthly')
    }
    session_config = {
        'sydney': {'display': 'Sydney', 'icon': 'bi-sun'},
        'tokyo': {'display': 'Tokyo', 'icon': 'bi-moon'},
        'london': {'display': 'London', 'icon': 'bi-cloud'},
        'ny': {'display': 'New York', 'icon': 'bi-building'},
    }
    sessions_dict = defaultdict(lambda: {'trades': [], 'pnl': 0})
    for t in trades:
        if not t.session:
            continue
        norm = t.session.lower()
        if 'new york' in norm:
            norm = 'ny'
        key = norm if norm in session_config else 'other'
        if key == 'other':
            if 'other' not in session_config:
                session_config['other'] = {'display': 'Other', 'icon': 'bi-globe2'}
        sessions_dict[key]['trades'].append(t)
        sessions_dict[key]['pnl'] += t.pnl or 0
    sessions = []
    tips = {
        'sydney': 'Early bird gets the worm—focus on AUD pairs.',
        'tokyo': 'Zen mode: patience pays in low vol.',
        'london': 'High energy—watch for breakouts.',
        'ny': 'Power hour: momentum rules.',
        'other': 'Keep grinding!'
    }
    assumed_position = 1000
    for key, data in sessions_dict.items():
        trades_list = data['trades']
        num_trades = len(trades_list)
        if num_trades == 0:
            continue
        total_pnl = data['pnl']
        wins = sum(1 for t in trades_list if t.pnl and t.pnl > 0)
        win_rate = round((wins / num_trades * 100))
        avg_pnl = total_pnl / num_trades
        avg_pnl_str = f"${avg_pnl:+.2f}"
        recent_trades = sorted(trades_list, key=lambda t: t.created_at)[-5:]
        num_recent = len(recent_trades)
        trend = [0] * (5 - num_recent)
        current = 0
        for t in recent_trades:
            change = ((t.pnl or 0) / assumed_position * 100)
            current += change
            trend.append(round(current))
        last_trade_trade = max(trades_list, key=lambda t: t.created_at)
        pnl_str = f"{last_trade_trade.pnl:+.2f}" if last_trade_trade.pnl is not None else "N/A"
        last_trade_str = f"{last_trade_trade.symbol or 'N/A'} ${pnl_str}"
        tip = tips.get(key, 'Keep it up!')
        sessions.append({
            'name': session_config[key]['display'],
            'icon': session_config[key]['icon'],
            'trades': num_trades,
            'winRate': win_rate,
            'avgPnL': avg_pnl_str,
            'trend': trend,
            'lastTrade': last_trade_str,
            'tip': tip
        })
    sample_session = sessions[0] if sessions else {'name': 'your sessions', 'winRate': 0}
    ref_result = await db.execute(
        select(func.count()).where(models.User.referred_by == current_user.id)
    )
    referrals_count = ref_result.scalar() or 0
    earned_analyses = referrals_count
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "current_user": current_user,
            "initials": initials,
            "greeting": greeting,
            "todays_edge": todays_edge,
            "pnl": pnl,
            "pnl_history": pnl_history,
            "sessions": sessions,
            "session": sample_session,
            "referral_code": current_user.referral_code,
            "referrals_count": referrals_count,
            "earned_analyses": earned_analyses,
            "now": datetime.utcnow()
        }
    )

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(
    request: Request,
    current_user: Optional[models.User] = Depends(get_current_user_optional)
):
    if not current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    initials = ""
    if current_user.full_name:
        names = current_user.full_name.split()
        if len(names) >= 2:
            initials = names[0][0].upper() + names[-1][0].upper()
        elif len(names) == 1:
            initials = names[0][0].upper() * 2
    else:
        initials = "U"
    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "current_user": current_user,
            "initials": initials,
            "now": datetime.utcnow()
        }
    )

@app.get("/journal", response_class=HTMLResponse)
async def journal_page(
    request: Request,
    current_user: Optional[models.User] = Depends(get_current_user_optional)
):
    if not current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    initials = ""
    if current_user.full_name:
        names = current_user.full_name.split()
        if len(names) >= 2:
            initials = names[0][0].upper() + names[-1][0].upper()
        elif len(names) == 1:
            initials = names[0][0].upper() * 2
    else:
        initials = "U"
    return templates.TemplateResponse(
        "journal.html",
        {
            "request": request,
            "current_user": current_user,
            "initials": initials,
            "now": datetime.utcnow()
        }
    )

@app.get("/profile", response_class=HTMLResponse)
async def profile_page(
    request: Request,
    current_user: Optional[models.User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_session)
):
    if not current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    initials = ""
    if current_user.full_name:
        names = current_user.full_name.split()
        if len(names) >= 2:
            initials = names[0][0].upper() + names[-1][0].upper()
        elif len(names) == 1:
            initials = names[0][0].upper() * 2
    else:
        initials = "U"
    result = await db.execute(
        select(models.Trade).where(models.Trade.owner_id == current_user.id)
    )
    trades = result.scalars().all()
    lifetime_pnl = sum(trade.pnl for trade in trades if trade.pnl is not None)
    total_trades = len(trades)
    win_trades = len([t for t in trades if t.pnl and t.pnl > 0])
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
    best_trade = max((t for t in trades if t.pnl is not None), key=lambda t: t.pnl, default=None)
    worst_trade = min((t for t in trades if t.pnl is not None), key=lambda t: t.pnl, default=None)
    most_traded = {}
    for trade in trades:
        if trade.symbol:
            most_traded[trade.symbol] = most_traded.get(trade.symbol, 0) + 1
    top_tickers = [ticker for ticker, _ in sorted(most_traded.items(), key=lambda x: x[1], reverse=True)[:4]]
    formatted_joined = current_user.created_at.strftime('%B %d, %Y') if current_user.created_at else ''
    bio = getattr(current_user, 'bio', '')
    trading_style = getattr(current_user, 'trading_style', '')
    goals = getattr(current_user, 'goals', '')
    return templates.TemplateResponse(
        "profile.html",
        {
            "request": request,
            "current_user": current_user,
            "initials": initials,
            "lifetime_pnl": lifetime_pnl,
            "win_rate": win_rate,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "top_tickers": top_tickers,
            "formatted_joined": formatted_joined,
            "bio": bio,
            "trading_style": trading_style,
            "goals": goals,
            "now": datetime.utcnow()
        }
    )

@app.get("/plans", response_class=HTMLResponse)
async def plans_page(
    request: Request,
    current_user: Optional[models.User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_session)
):
    if not current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    initials = ""
    if current_user.full_name:
        names = current_user.full_name.split()
        if len(names) >= 2:
            initials = names[0][0].upper() + names[-1][0].upper()
        elif len(names) == 1:
            initials = names[0][0].upper() * 2
    else:
        initials = "U"
    
    pricing = {
        'pro_monthly': 9.99,
        'pro_yearly': 99.00,
        'elite_monthly': 19.99,
        'elite_yearly': 199.00
    }
    result_pricing = await db.execute(
        select(models.Pricing).where(
            models.Pricing.plan.in_(['pro', 'elite']),
            models.Pricing.interval.in_(['monthly', 'yearly'])
        )
    )
    for p in result_pricing.scalars().all():
        key = f"{p.plan}_{p.interval}"
        pricing[key] = p.amount

    discount = {
        'enabled': False,
        'percentage': 0.0,
        'expiry': None
    }
    result_discount = await db.execute(
        select(models.Discount).where(models.Discount.id == 1)
    )
    db_discount = result_discount.scalar_one_or_none()
    if db_discount:
        discount['enabled'] = db_discount.enabled
        discount['percentage'] = db_discount.percentage
        discount['expiry'] = db_discount.expiry

    effective_discount = 0.0
    if db_discount and db_discount.enabled and (not db_discount.expiry or db_discount.expiry > date.today()):
        effective_discount = db_discount.percentage

    nested_pricing = {
        'pro': {
            'monthly': pricing['pro_monthly'],
            'yearly': pricing['pro_yearly']
        },
        'elite': {
            'monthly': pricing['elite_monthly'],
            'yearly': pricing['elite_yearly']
        }
    }

    return templates.TemplateResponse(
        "plans.html",
        {
            "request": request,
            "current_user": current_user,
            "initials": initials,
            "pricing": pricing,
            "nested_pricing": nested_pricing,
            "discount": discount,
            "effective_discount": effective_discount,
            "now": datetime.utcnow()
        }
    )

app.include_router(users.router)
app.include_router(uploads.router)
app.include_router(insights.router)
app.include_router(journal.router)
app.include_router(profile.router)
app.include_router(admin.router)
app.include_router(ai.router)
app.include_router(payments.router)

@app.get("/{full_path:path}")
async def catch_html_redirect(full_path: str):
    if full_path.endswith('.html'):
        clean_path = full_path[:-5]
        logger.info(f"Redirecting {full_path} to /{clean_path}")
        return RedirectResponse(url=f"/{clean_path}", status_code=status.HTTP_301_MOVED_PERMANENTLY)
    raise HTTPException(status_code=404, detail="Not Found")
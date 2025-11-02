from pathlib import Path
from datetime import timedelta
from typing import Union, Any, Optional
import json
import urllib.parse  # For URI parsing

from fastapi import APIRouter, Depends, Request, Form, HTTPException, status, Cookie
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, exists
from authlib.integrations.starlette_client import OAuth

from templates_config import templates
from models import models, schemas
import auth
from config import settings
from database import get_session
from utils import generate_unique_referral_code

# Initialize OAuth with credentials from settings
oauth = OAuth()
oauth.register(
    name='google',
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

router = APIRouter(prefix="/users", tags=["Users"])

# JSON API signup (existing)
@router.post("/signup-email", response_model=schemas.GenericResponse)
async def create_user_json(payload: schemas.SignupRequest, db: AsyncSession = Depends(get_session)):
    email_check = await db.execute(select(models.User).where(models.User.email == payload.email))
    if email_check.scalars().first():
        return {"success": False, "message": "Email already in use."}

    username_check = await db.execute(select(models.User).where(models.User.username == payload.username))
    if username_check.scalars().first():
        return {"success": False, "message": "Username already taken."}

    hashed_password = auth.hash_password(payload.password)
    admin_exists = await db.execute(select(exists().where(models.User.is_admin == True)))
    is_admin = not admin_exists.scalar()

    new_user = models.User(
        username=payload.username.lower(),
        full_name=payload.full_name,
        email=payload.email,
        password_hash=hashed_password,
        referral_code=await generate_unique_referral_code(db),
        referred_by=None,
        is_admin=is_admin,
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    access_token = auth.create_access_token({"sub": str(new_user.id)}, access_token_expires)
    refresh_token = auth.create_refresh_token({"sub": str(new_user.id)}, refresh_token_expires)

    return {"success": True, "message": "User created successfully", "data": {"access_token": access_token, "refresh_token": refresh_token}}

# JSON API login (existing)
@router.post("/login-email", response_model=schemas.GenericResponse)
async def login_json(payload: schemas.LoginRequest, db: AsyncSession = Depends(get_session)):
    result = await db.execute(select(models.User).where(models.User.username == payload.username))
    user = result.scalars().first()
    if not user or not auth.verify_password(payload.password, user.password_hash):
        return {"success": False, "message": "Invalid username or password."}

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS if payload.remember_me else settings.REFRESH_TOKEN_EXPIRE_DAYS)

    access_token = auth.create_access_token({"sub": str(user.id)}, access_token_expires)
    refresh_token = auth.create_refresh_token({"sub": str(user.id)}, refresh_token_expires)

    return {"success": True, "message": "Login successful", "data": {"access_token": access_token, "refresh_token": refresh_token}}

# JSON API logout (existing)
@router.post("/logout", response_model=schemas.GenericResponse)
async def logout_json(current_user: models.User = Depends(auth.get_current_user)):
    # Add refresh token blacklist logic when needed
    return {"success": True, "message": "Successfully logged out"}

# Form signup handler (UPDATED: Added password_confirm field and validation)
@router.post("/signup", response_class=HTMLResponse)
async def create_user_form(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
    referral_code: str = Form(None),
    db: AsyncSession = Depends(get_session),
):
    if password != password_confirm:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Passwords do not match.", "tab": "signup"}, status_code=400)

    email_check = await db.execute(select(models.User).where(models.User.email == email))
    if email_check.scalars().first():
        return templates.TemplateResponse("index.html", {"request": request, "error": "Email already in use.", "tab": "signup"}, status_code=400)

    username_check = await db.execute(select(models.User).where(models.User.username == username))
    if username_check.scalars().first():
        return templates.TemplateResponse("index.html", {"request": request, "error": "Username already taken.", "tab": "signup"}, status_code=400)

    hashed_password = auth.hash_password(password)
    admin_exists = await db.execute(select(exists().where(models.User.is_admin == True)))
    is_admin = not admin_exists.scalar()

    referred_by = None
    if referral_code:
        referrer_check = await db.execute(select(models.User).where(models.User.referral_code == referral_code))
        referrer = referrer_check.scalars().first()
        if referrer:
            referred_by = referrer.id

    new_user = models.User(
        username=username.lower(),
        full_name=full_name,
        email=email,
        password_hash=hashed_password,
        referral_code=await generate_unique_referral_code(db),
        referred_by=referred_by,
        is_admin=is_admin,
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    redirect = RedirectResponse(url="/?success=true&tab=login", status_code=status.HTTP_303_SEE_OTHER)
    return redirect

# Form login handler (existing)
@router.post("/login", response_class=HTMLResponse)
async def login_form(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    remember_me: bool = Form(False),
    db: AsyncSession = Depends(get_session),
):
    result = await db.execute(select(models.User).where(models.User.username == username))
    user = result.scalars().first()
    if not user or not auth.verify_password(password, user.password_hash):
        return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid username or password.", "tab": "login"}, status_code=401)

    if remember_me:
        access_token_expires = timedelta(days=settings.REMEMBER_ME_REFRESH_TOKEN_EXPIRE_DAYS)
    else:
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    access_token = auth.create_access_token({"sub": str(user.id)}, access_token_expires)

    redirect = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    max_age = int(access_token_expires.total_seconds())
    redirect.set_cookie("access_token", access_token, httponly=True, max_age=max_age, secure=False, samesite="lax")
    return redirect

# Google OAuth login redirect (UPDATED: More robust redirect_uri construction + logging)
@router.get("/google/login")
async def google_login(request: Request):
    referral_code = request.query_params.get('referral_code')
    state = None
    if referral_code:
        state = json.dumps({'ref': referral_code})
    
    # Use url_for for path, but construct full URI explicitly to avoid scheme/host mismatches
    # Adjust BASE_URL in settings.py if needed (e.g., 'http://localhost:8000' for dev)
    base_url = getattr(settings, 'BASE_URL', str(request.url.scheme) + '://' + str(request.url.netloc))
    redirect_path = str(request.url_for('google_callback'))  # FIXED: Convert URL object to str
    redirect_uri = urllib.parse.urljoin(base_url, redirect_path)
    
    # Log for debugging (remove in prod)
    print(f"Generated redirect_uri: {redirect_uri}")
    print(f"Request base: {base_url}")
    
    return await oauth.google.authorize_redirect(request, redirect_uri, state=state)

# Google OAuth callback (UPDATED: Extract referral_code from state and handle for new users)
@router.get("/google/callback", response_class=HTMLResponse)
async def google_callback(request: Request, db: AsyncSession = Depends(get_session)):
    try:
        token = await oauth.google.authorize_access_token(request)
        
        # Try to parse id_token (OpenID Connect preferred)
        try:
            user_info = await oauth.google.parse_id_token(request, token)
        except KeyError as ke:
            if "'id_token'" in str(ke):
                # Fallback: Fetch user info from /userinfo endpoint using access_token
                resp = await oauth.google.get('https://www.googleapis.com/oauth2/v1/userinfo', token=token)
                resp.raise_for_status()
                user_info = resp.json()
            else:
                raise  # Re-raise if not the expected KeyError
        
    except Exception as e:
        # More specific error logging
        error_msg = f"Google authentication failed: {str(e)}"
        print(error_msg)  # Log for debugging
        if "redirect_uri_mismatch" in str(e).lower():
            error_msg = "Redirect URI mismatch. Check Google Console authorized URIs match your app's base URL."
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": error_msg, "tab": "login"},
            status_code=400
        )

    # Extract referral_code from state
    state = request.query_params.get('state')
    ref_data = {}
    if state:
        try:
            ref_data = json.loads(state)
        except json.JSONDecodeError:
            pass
    referral_code = ref_data.get('ref') if isinstance(ref_data, dict) else None

    email = user_info['email']
    full_name = user_info.get('name', '')
    # Generate a unique username based on email or Google name
    username_base = email.split('@')[0].lower().replace('.', '')
    username = username_base
    counter = 1
    # FIXED: Extract await to variable to avoid chaining issues in async loop
    while True:
        result = await db.execute(select(models.User).where(models.User.username == username))
        existing_user = result.scalars().first()
        if not existing_user:
            break
        username = f"{username_base}{counter}"
        counter += 1

    # Check if user exists
    result = await db.execute(select(models.User).where(models.User.email == email))
    user = result.scalars().first()

    if not user:
        # Create new user
        admin_exists = await db.execute(select(exists().where(models.User.is_admin == True)))
        is_admin = not admin_exists.scalar()

        referred_by = None
        if referral_code:
            referrer_check = await db.execute(select(models.User).where(models.User.referral_code == referral_code))
            referrer = referrer_check.scalars().first()
            if referrer:
                referred_by = referrer.id

        new_user = models.User(
            username=username,
            full_name=full_name,
            email=email,
            password_hash=None,  # No password for Google users
            referral_code=await generate_unique_referral_code(db),
            referred_by=referred_by,
            is_admin=is_admin,
        )
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        user = new_user

    # Generate tokens
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token({"sub": str(user.id)}, access_token_expires)

    redirect = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    max_age = int(access_token_expires.total_seconds())
    redirect.set_cookie("access_token", access_token, httponly=True, max_age=max_age, secure=False, samesite="lax")
    return redirect

# Logout handler (existing)
@router.get("/logout")
async def logout(
    access_token: Optional[str] = Cookie(None)
):
    redirect = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    if access_token:
        redirect.delete_cookie("access_token")
    return redirect


#     INFO:     127.0.0.1:58726 - "GET /users/google/login HTTP/1.1" 500 Internal Server Error
# ERROR:    Exception in ASGI application
# Traceback (most recent call last):
#   File "/home/ukov/.local/lib/python3.11/site-packages/httpx/_transports/default.py", line 101, in map_httpcore_exceptions
#     yield
#   File "/home/ukov/.local/lib/python3.11/site-packages/httpx/_transports/default.py", line 394, in handle_async_request
#     resp = await self._pool.handle_async_request(req)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/httpcore/_async/connection_pool.py", line 256, in handle_async_request
#     raise exc from None
#   File "/home/ukov/.local/lib/python3.11/site-packages/httpcore/_async/connection_pool.py", line 236, in handle_async_request
#     response = await connection.handle_async_request(
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/httpcore/_async/connection.py", line 101, in handle_async_request
#     raise exc
#   File "/home/ukov/.local/lib/python3.11/site-packages/httpcore/_async/connection.py", line 78, in handle_async_request
#     stream = await self._connect(request)
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/httpcore/_async/connection.py", line 124, in _connect
#     stream = await self._network_backend.connect_tcp(**kwargs)
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/httpcore/_backends/auto.py", line 31, in connect_tcp
#     return await self._backend.connect_tcp(
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/httpcore/_backends/anyio.py", line 113, in connect_tcp
#     with map_exceptions(exc_map):
#   File "/usr/lib/python3.11/contextlib.py", line 155, in __exit__
#     self.gen.throw(typ, value, traceback)
#   File "/home/ukov/.local/lib/python3.11/site-packages/httpcore/_exceptions.py", line 14, in map_exceptions
#     raise to_exc(exc) from exc
# httpcore.ConnectTimeout

# The above exception was the direct cause of the following exception:

# Traceback (most recent call last):
#   File "/home/ukov/.local/lib/python3.11/site-packages/uvicorn/protocols/http/httptools_impl.py", line 401, in run_asgi
#     result = await app(  # type: ignore[func-returns-value]
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/uvicorn/middleware/proxy_headers.py", line 60, in __call__
#     return await self.app(scope, receive, send)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/fastapi/applications.py", line 1054, in __call__
#     await super().__call__(scope, receive, send)
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/applications.py", line 113, in __call__
#     await self.middleware_stack(scope, receive, send)
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/middleware/errors.py", line 187, in __call__
#     raise exc
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/middleware/errors.py", line 165, in __call__
#     await self.app(scope, receive, _send)
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/middleware/base.py", line 185, in __call__
#     with collapse_excgroups():
#   File "/usr/lib/python3.11/contextlib.py", line 155, in __exit__
#     self.gen.throw(typ, value, traceback)
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/_utils.py", line 82, in collapse_excgroups
#     raise exc
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/middleware/base.py", line 187, in __call__
#     response = await self.dispatch_func(request, call_next)
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/itrade/server/main.py", line 123, in auth_redirect_middleware
#     response = await call_next(request)
#                ^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/middleware/base.py", line 163, in call_next
#     raise app_exc
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/middleware/base.py", line 149, in coro
#     await self.app(scope, receive_or_disconnect, send_no_error)
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/middleware/sessions.py", line 85, in __call__
#     await self.app(scope, receive, send_wrapper)
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/middleware/cors.py", line 85, in __call__
#     await self.app(scope, receive, send)
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/middleware/exceptions.py", line 62, in __call__
#     await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
#     raise exc
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
#     await app(scope, receive, sender)
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/routing.py", line 715, in __call__
#     await self.middleware_stack(scope, receive, send)
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/routing.py", line 735, in app
#     await route.handle(scope, receive, send)
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/routing.py", line 288, in handle
#     await self.app(scope, receive, send)
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/routing.py", line 76, in app
#     await wrap_app_handling_exceptions(app, request)(scope, receive, send)
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
#     raise exc
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
#     await app(scope, receive, sender)
#   File "/home/ukov/.local/lib/python3.11/site-packages/starlette/routing.py", line 73, in app
#     response = await f(request)
#                ^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/fastapi/routing.py", line 301, in app
#     raw_response = await run_endpoint_function(
#                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/fastapi/routing.py", line 212, in run_endpoint_function
#     return await dependant.call(**values)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/itrade/server/router/users.py", line 176, in google_login
#     return await oauth.google.authorize_redirect(request, redirect_uri, state=state)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/authlib/integrations/starlette_client/apps.py", line 36, in authorize_redirect
#     rv = await self.create_authorization_url(redirect_uri, **kwargs)
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/authlib/integrations/base_client/async_app.py", line 100, in create_authorization_url
#     metadata = await self.load_server_metadata()
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/authlib/integrations/base_client/async_app.py", line 79, in load_server_metadata
#     resp = await client.request(
#            ^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/authlib/integrations/httpx_client/oauth2_client.py", line 119, in request
#     return await super().request(method, url, auth=auth, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/httpx/_client.py", line 1540, in request
#     return await self.send(request, auth=auth, follow_redirects=follow_redirects)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/httpx/_client.py", line 1629, in send
#     response = await self._send_handling_auth(
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/httpx/_client.py", line 1657, in _send_handling_auth
#     response = await self._send_handling_redirects(
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/httpx/_client.py", line 1694, in _send_handling_redirects
#     response = await self._send_single_request(request)
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/httpx/_client.py", line 1730, in _send_single_request
#     response = await transport.handle_async_request(request)
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ukov/.local/lib/python3.11/site-packages/httpx/_transports/default.py", line 393, in handle_async_request
#     with map_httpcore_exceptions():
#   File "/usr/lib/python3.11/contextlib.py", line 155, in __exit__
#     self.gen.throw(typ, value, traceback)
#   File "/home/ukov/.local/lib/python3.11/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
#     raise mapped_exc(message) from exc
# httpx.ConnectTimeout

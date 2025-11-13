import jwt
import bcrypt  # NEW: Direct import instead of passlib
from datetime import datetime, timedelta
from jose import JWTError
from fastapi import Depends, HTTPException, status, Cookie
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import logging

from database import get_session
from models import models
from config import get_settings

settings = get_settings()

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

logger = logging.getLogger(__name__)

def hash_password(password: str) -> str:
    if not password:
        raise ValueError("Password cannot be empty")
    
    # NEW: Explicitly truncate to 72 bytes (bcrypt limit; UTF-8 safe)
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        logger.warning(f"Password truncated from {len(password_bytes)} to 72 bytes for bcrypt compatibility")
        password_bytes = password_bytes[:72]
        # Note: We don't decode back—hash the bytes directly
    
    salt = bcrypt.gensalt()  # Uses default rounds (12); adjust if needed via rounds=14
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')  # Return as str for storage

def verify_password(plain_password: str, hashed_password: str) -> bool:
    plain_bytes = plain_password.encode('utf-8')
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(plain_bytes, hashed_bytes)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict, expires_delta: timedelta) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("exp") < int(datetime.utcnow().timestamp()):
            raise HTTPException(status_code=401, detail="Token expired")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(
    access_token: Optional[str] = Cookie(None),
    db: AsyncSession = Depends(get_session)
) -> models.User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not access_token:
        raise credentials_exception
    try:
        payload = decode_access_token(access_token)
        user_id: int = int(payload.get("sub"))
        if user_id is None:
            raise credentials_exception
    except HTTPException:
        raise credentials_exception
    result = await db.execute(select(models.User).where(models.User.id == user_id))
    user = result.scalars().first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_user_optional(
    access_token: Optional[str] = Cookie(None),
    db: AsyncSession = Depends(get_session)
) -> Optional[models.User]:
    if not access_token:
        return None
    try:
        payload = decode_access_token(access_token)
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
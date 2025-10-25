import secrets
import string
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from models.models import User 

# Simple random referral code generator
def generate_referral_code(length: int = 10) -> str:
    """Generate a random alphanumeric referral code."""
    return ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(length))

# Ensure the generated code is unique in the database
async def generate_unique_referral_code(db: AsyncSession, length: int = 10) -> str:
    """Generate a unique referral code, retrying if it already exists."""
    while True:
        code = generate_referral_code(length)
        result = await db.execute(select(User).where(User.referral_code == code))
        if not result.scalars().first():
            return code

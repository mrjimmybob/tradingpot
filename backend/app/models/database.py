"""Database configuration and session management."""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

DATABASE_URL = "sqlite+aiosqlite:///./tradingbot.db"

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


async def get_session() -> AsyncSession:
    """Dependency for getting database sessions."""
    async with async_session_maker() as session:
        yield session


async def init_db():
    """Initialize the database, creating all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

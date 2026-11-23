from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from video_ranker.config import get_settings
from video_ranker.db.models import Base

engine = create_engine(get_settings().database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def init_db() -> None:
    """Enable pgvector and create tables if they don't exist."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)

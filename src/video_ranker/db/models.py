from datetime import datetime
from enum import StrEnum

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from video_ranker.config import get_settings

EMBEDDING_DIM = get_settings().embedding_dim


class Base(DeclarativeBase):
    pass


class VideoStatus(StrEnum):
    PENDING = "pending"
    DOWNLOADED = "downloaded"
    PROCESSED = "processed"
    FAILED = "failed"


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_url: Mapped[str] = mapped_column(String(2048), unique=True)
    platform: Mapped[str | None] = mapped_column(String(64))
    external_id: Mapped[str | None] = mapped_column(String(128))
    title: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    duration_s: Mapped[float | None]
    status: Mapped[str] = mapped_column(String(16), default=VideoStatus.PENDING)
    error: Mapped[str | None] = mapped_column(Text)
    video_key: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    keyframes: Mapped[list["Keyframe"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )
    transcript: Mapped["Transcript | None"] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )


class Keyframe(Base):
    __tablename__ = "keyframes"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    timestamp_s: Mapped[float]
    image_key: Mapped[str] = mapped_column(String(512))
    caption: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))

    video: Mapped[Video] = relationship(back_populates="keyframes")


class Transcript(Base):
    __tablename__ = "transcripts"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), unique=True)
    language: Mapped[str | None] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))

    video: Mapped[Video] = relationship(back_populates="transcript")

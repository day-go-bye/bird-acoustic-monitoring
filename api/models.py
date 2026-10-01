from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from api.database import Base


class Recording(Base):
    __tablename__ = "recordings"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    filename: Mapped[str] = mapped_column(String, nullable=False)

    audio_path: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    recorded_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    latitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    longitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    duration_seconds: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    source: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="pending",
    )

class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    recording_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("recordings.id"),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="queued",
    )

    error_message: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )


class Observation(Base):
    __tablename__ = "observations"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    recording_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("recordings.id"),
        nullable=False,
    )

    species: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    start_seconds: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    end_seconds: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    review_status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="unreviewed",
    )
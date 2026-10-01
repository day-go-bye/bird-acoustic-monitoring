from datetime import datetime

from pydantic import BaseModel, Field


class RecordingCreate(BaseModel):
    """Fields a client can provide when creating a recording."""

    filename: str = Field(min_length=1)
    recorded_at: datetime | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    duration_seconds: float | None = Field(default=None, gt=0)
    source: str | None = None


class RecordingResponse(BaseModel):
    """Fields returned to API clients for a recording."""

    id: str
    filename: str
    audio_path: str | None
    recorded_at: datetime | None
    latitude: float | None
    longitude: float | None
    duration_seconds: float | None
    source: str | None
    status: str

    model_config = {"from_attributes": True}


class ProcessingJobResponse(BaseModel):
    """Fields returned to API clients for a processing job."""

    id: str
    recording_id: str
    status: str
    error_message: str | None

    model_config = {"from_attributes": True}


class ObservationResponse(BaseModel):
    """Fields returned to API clients for a model observation."""

    id: str
    recording_id: str
    species: str
    confidence: float
    start_seconds: float
    end_seconds: float
    model_version: str
    review_status: str

    model_config = {"from_attributes": True}


class ObservationReview(BaseModel):
    """Review decision submitted by a human reviewer."""

    review_status: str


class ObservationDetailResponse(BaseModel):
    """Observation plus recording metadata needed for human review."""

    id: str
    recording_id: str
    species: str
    confidence: float
    start_seconds: float
    end_seconds: float
    model_version: str
    review_status: str

    filename: str
    audio_path: str | None
    recorded_at: datetime | None
    latitude: float | None
    longitude: float | None

    model_config = {"from_attributes": True}
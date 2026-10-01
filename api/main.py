from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from api import models  # noqa: F401
from api.database import Base, engine, get_db
from api.models import Observation, ProcessingJob, Recording
from api.schemas import (
    ObservationDetailResponse,
    ObservationResponse,
    ObservationReview,
    ProcessingJobResponse,
    RecordingCreate,
    RecordingResponse,
)
from api.storage import save_audio
from datetime import datetime
from pathlib import Path
import io
import wave
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bird_monitoring.audio import compute_spectrogram, load_audio


Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="Bird Acoustic Monitoring API",
    description="Platform API for passive acoustic biodiversity monitoring",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recordings", response_model=RecordingResponse)
async def create_recording(
    filename: str = Form(...),
    recorded_at: datetime | None = Form(None),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    duration_seconds: float | None = Form(None),
    source: str | None = Form(None),
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    file_bytes = await audio_file.read()

    audio_path = save_audio(
        file_bytes,
        audio_file.filename or filename,
    )

    recording = RecordingCreate(
        filename=filename,
        recorded_at=recorded_at,
        latitude=latitude,
        longitude=longitude,
        duration_seconds=duration_seconds,
        source=source,
    )

    db_recording = Recording(
        **recording.model_dump(),
        audio_path=audio_path,
    )

    db.add(db_recording)
    db.commit()
    db.refresh(db_recording)

    return db_recording


@app.post(
    "/recordings/{recording_id}/process",
    response_model=ProcessingJobResponse,
)
def process_recording(
    recording_id: str,
    db: Session = Depends(get_db),
):
    recording = db.get(Recording, recording_id)

    if recording is None:
        raise HTTPException(
            status_code=404,
            detail="Recording not found",
        )

    existing_job = (
        db.query(ProcessingJob)
        .filter(
            ProcessingJob.recording_id == recording_id,
            ProcessingJob.status.in_(["queued", "processing"]),
        )
        .first()
    )

    if existing_job is not None:
        raise HTTPException(
            status_code=409,
            detail="Recording already has a processing job in progress",
        )

    job = ProcessingJob(
        recording_id=recording.id,
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    return job


@app.get("/recordings/{recording_id}", response_model=RecordingResponse)
def get_recording(
    recording_id: str,
    db: Session = Depends(get_db),
):
    recording = db.get(Recording, recording_id)

    if recording is None:
        raise HTTPException(
            status_code=404,
            detail="Recording not found",
        )

    return recording


@app.get("/recordings/{recording_id}/audio")
def get_recording_audio(
    recording_id: str,
    db: Session = Depends(get_db),
):
    recording = db.get(Recording, recording_id)

    if recording is None:
        raise HTTPException(
            status_code=404,
            detail="Recording not found",
        )

    if recording.audio_path is None:
        raise HTTPException(
            status_code=404,
            detail="Recording has no audio file",
        )

    audio_path = Path(recording.audio_path)

    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Audio file not found",
        )

    return FileResponse(
        path=audio_path,
        media_type="audio/wav",
        filename=recording.filename,
    )


@app.get(
    "/recordings/{recording_id}/observations",
    response_model=list[ObservationResponse],
)
def get_recording_observations(
    recording_id: str,
    db: Session = Depends(get_db),
):
    recording = db.get(Recording, recording_id)

    if recording is None:
        raise HTTPException(
            status_code=404,
            detail="Recording not found",
        )

    return (
        db.query(Observation)
        .filter(Observation.recording_id == recording_id)
        .order_by(Observation.start_seconds)
        .all()
    )


@app.get(
    "/observations",
    response_model=list[ObservationResponse],
)
def get_observations(
    review_status: str | None = None,
    db: Session = Depends(get_db),
):
    query = db.query(Observation)

    if review_status is not None:
        query = query.filter(
            Observation.review_status == review_status
        )

    return (
        query
        .order_by(Observation.recording_id, Observation.start_seconds)
        .all()
    )


@app.get(
    "/observations/{observation_id}",
    response_model=ObservationDetailResponse,
)
def get_observation(
    observation_id: str,
    db: Session = Depends(get_db),
):
    observation = db.get(Observation, observation_id)

    if observation is None:
        raise HTTPException(
            status_code=404,
            detail="Observation not found",
        )

    recording = db.get(Recording, observation.recording_id)

    if recording is None:
        raise HTTPException(
            status_code=404,
            detail="Recording not found",
        )

    return {
        **{
            "id": observation.id,
            "recording_id": observation.recording_id,
            "species": observation.species,
            "confidence": observation.confidence,
            "start_seconds": observation.start_seconds,
            "end_seconds": observation.end_seconds,
            "model_version": observation.model_version,
            "review_status": observation.review_status,
        },
        "filename": recording.filename,
        "audio_path": recording.audio_path,
        "recorded_at": recording.recorded_at,
        "latitude": recording.latitude,
        "longitude": recording.longitude,
    }


@app.post(
    "/observations/{observation_id}/review",
    response_model=ObservationResponse,
)
def review_observation(
    observation_id: str,
    review: ObservationReview,
    db: Session = Depends(get_db),
):
    if review.review_status not in {
        "confirmed",
        "rejected",
        "uncertain",
    }:
        raise HTTPException(
            status_code=400,
            detail="Invalid review status",
        )

    observation = db.get(Observation, observation_id)

    if observation is None:
        raise HTTPException(
            status_code=404,
            detail="Observation not found",
        )

    observation.review_status = review.review_status

    db.commit()
    db.refresh(observation)

    return observation


@app.get("/observations/{observation_id}/audio")
def get_observation_audio(
    observation_id: str,
    db: Session = Depends(get_db),
):
    observation = db.get(Observation, observation_id)

    if observation is None:
        raise HTTPException(
            status_code=404,
            detail="Observation not found",
        )

    recording = db.get(Recording, observation.recording_id)

    if recording is None:
        raise HTTPException(
            status_code=404,
            detail="Recording not found",
        )

    if recording.audio_path is None:
        raise HTTPException(
            status_code=404,
            detail="Recording has no audio file",
        )

    audio_path = Path(recording.audio_path)

    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Audio file not found",
        )

    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            start_frame = int(
                observation.start_seconds * frame_rate
            )
            end_frame = int(
                observation.end_seconds * frame_rate
            )

            wav_file.setpos(start_frame)

            frames = wav_file.readframes(
                end_frame - start_frame
            )

            output = io.BytesIO()

            with wave.open(output, "wb") as output_wav:
                output_wav.setnchannels(
                    wav_file.getnchannels()
                )
                output_wav.setsampwidth(
                    wav_file.getsampwidth()
                )
                output_wav.setframerate(frame_rate)
                output_wav.writeframes(frames)

            output.seek(0)

    except (wave.Error, EOFError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to read audio recording",
        ) from exc

    return StreamingResponse(
        output,
        media_type="audio/wav",
        headers={
            "Content-Disposition": (
                f'inline; filename="{observation.id}.wav"'
            )
        },
    )


@app.get("/observations/{observation_id}/spectrogram")
def get_observation_spectrogram(
    observation_id: str,
    db: Session = Depends(get_db),
):
    observation = db.get(Observation, observation_id)

    if observation is None:
        raise HTTPException(
            status_code=404,
            detail="Observation not found",
        )

    recording = db.get(Recording, observation.recording_id)

    if recording is None:
        raise HTTPException(
            status_code=404,
            detail="Recording not found",
        )

    if recording.audio_path is None:
        raise HTTPException(
            status_code=404,
            detail="Recording has no audio file",
        )

    audio_path = Path(recording.audio_path)

    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Audio file not found",
        )

    try:
        audio, sample_rate = load_audio(audio_path)

        start_sample = int(
            observation.start_seconds * sample_rate
        )
        end_sample = int(
            observation.end_seconds * sample_rate
        )

        audio_segment = audio[start_sample:end_sample]

        frequencies, times, spectrogram = compute_spectrogram(
            audio_segment,
            sample_rate,
        )

        frequency_mask = frequencies <= 12000

        frequencies = frequencies[frequency_mask]
        spectrogram = spectrogram[frequency_mask, :]

        fig, ax = plt.subplots(
            figsize=(10, 4),
        )

        image = ax.pcolormesh(
            times,
            frequencies / 1000,
            spectrogram,
            shading="auto",
        )

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Frequency (kHz)")
        ax.set_title(
            f"{observation.species} "
            f"({observation.start_seconds:.1f}–"
            f"{observation.end_seconds:.1f}s)"
        )

        fig.colorbar(
            image,
            ax=ax,
            label="Power (dB)",
        )

        fig.tight_layout()

        output = io.BytesIO()

        fig.savefig(
            output,
            format="png",
            dpi=150,
        )

        plt.close(fig)

        output.seek(0)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate spectrogram",
        ) from exc

    return StreamingResponse(
        output,
        media_type="image/png",
    )


@app.get(
    "/jobs/{job_id}",
    response_model=ProcessingJobResponse,
)
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
):
    job = db.get(ProcessingJob, job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Processing job not found",
        )

    return job


@app.get("/review")
def review_page():
    return FileResponse("api/static/review.html")
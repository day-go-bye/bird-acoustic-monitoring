from sqlalchemy import delete

from api.database import SessionLocal
from api.models import Observation, ProcessingJob, Recording

from bird_monitoring.inference import BirdInference


def process_next_job(
    inference: BirdInference,
    model_version: str,
):
    db = SessionLocal()

    try:
        job = (
            db.query(ProcessingJob)
            .filter(ProcessingJob.status == "queued")
            .first()
        )

        if job is None:
            return None

        recording = db.get(Recording, job.recording_id)

        if recording is None:
            job.status = "failed"
            job.error_message = "Recording not found"
            db.commit()
            return job.id

        if recording.audio_path is None:
            job.status = "failed"
            job.error_message = "Recording has no audio file"
            db.commit()
            return job.id

        try:
            job.status = "processing"
            recording.status = "processing"
            db.commit()

            predictions = inference.predict_recording(
                recording.audio_path
            )

            db.execute(
                delete(Observation).where(
                    Observation.recording_id == recording.id
                )
            )

            for prediction in predictions:
                observation = Observation(
                    recording_id=recording.id,
                    species=prediction["species"],
                    confidence=prediction["confidence"],
                    start_seconds=prediction["start_seconds"],
                    end_seconds=prediction["end_seconds"],
                    model_version=model_version,
                )

                db.add(observation)

            job.status = "completed"
            recording.status = "processed"

            db.commit()

            return job.id

        except Exception as exc:
            db.rollback()

            job.status = "failed"
            job.error_message = str(exc)
            recording.status = "failed"
            db.commit()

            return job.id

    finally:
        db.close()
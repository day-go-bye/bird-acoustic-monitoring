from pathlib import Path
from uuid import uuid4


STORAGE_ROOT = Path("data/recordings")


def save_audio(file_bytes: bytes, original_filename: str) -> str:
    """Save an audio file and return its server-controlled storage path."""

    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

    extension = Path(original_filename).suffix.lower()
    stored_filename = f"{uuid4()}{extension}"

    destination = STORAGE_ROOT / stored_filename
    destination.write_bytes(file_bytes)

    return str(destination)
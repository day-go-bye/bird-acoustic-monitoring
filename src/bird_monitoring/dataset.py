from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from bird_monitoring.audio import load_audio


class BirdClefWindowDataset(Dataset):
    """Fixed-length audio windows from a recording-level BirdCLEF split."""

    def __init__(
        self,
        metadata_path: str | Path,
        audio_dir: str | Path,
        split: str,
        window_seconds: float = 5.0,
        hop_seconds: float = 2.5,
        max_windows_per_recording: int = 8,
        random_crop: bool = False,
        random_gain: bool = False,
        random_time_shift: bool = False,
    ):
        self.metadata_path = Path(metadata_path)
        self.audio_dir = Path(audio_dir)
        self.split = split
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.max_windows_per_recording = max_windows_per_recording
        self.random_crop = random_crop
        self.random_gain = random_gain
        self.random_time_shift = random_time_shift

        metadata = pd.read_csv(self.metadata_path)

        # Only use recordings that belong to this split and are long
        # enough to contain at least one complete window.
        metadata = metadata[
            (metadata["split"] == split)
            & (metadata["duration_seconds"] >= window_seconds)
        ].copy()

        if metadata.empty:
            raise ValueError(f"No usable recordings found for split={split!r}")

        # Stable class mapping shared across splits.
        all_metadata = pd.read_csv(self.metadata_path)
        classes = sorted(all_metadata["primary_label"].dropna().unique())
        self.class_to_idx = {label: idx for idx, label in enumerate(classes)}
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}

        self.recordings = metadata.reset_index(drop=True)
        self.windows = self._build_window_index()

    def _build_window_index(self) -> list[dict]:
        """Build a deterministic index of windows without loading audio."""
        windows = []

        for recording_idx, row in self.recordings.iterrows():
            duration = float(row["duration_seconds"])

            max_start = duration - self.window_seconds

            # Candidate starts at 0, hop, 2*hop, ... while a full
            # window still fits inside the recording.
            starts = np.arange(
                0.0,
                max_start + 1e-9,
                self.hop_seconds,
            )

            # Keep at most max_windows_per_recording candidates.
            # Evenly sample across the recording so long recordings
            # don't dominate the training set.
            if len(starts) > self.max_windows_per_recording:
                indices = np.linspace(
                    0,
                    len(starts) - 1,
                    self.max_windows_per_recording,
                    dtype=int,
                )
                starts = starts[indices]

            for window_idx, start_seconds in enumerate(starts):
                windows.append(
                    {
                        "recording_idx": recording_idx,
                        "window_idx": window_idx,
                        "start_seconds": float(start_seconds),
                    }
                )

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        window = self.windows[index]
        row = self.recordings.iloc[window["recording_idx"]]

        audio_path = self.audio_dir / Path(row["filename"]).name

        audio, sample_rate = load_audio(audio_path)

        if self.random_crop:
            max_start_seconds = max(
                0.0,
                float(row["duration_seconds"]) - self.window_seconds,
            )
            start_seconds = np.random.uniform(
                0.0,
                max_start_seconds,
            )
        else:
            start_seconds = window["start_seconds"]

        start_sample = int(round(start_seconds * sample_rate))
        window_samples = int(round(self.window_seconds * sample_rate))
        end_sample = start_sample + window_samples

        audio_window = audio[start_sample:end_sample]

        if self.random_gain:
            gain_db = np.random.uniform(-6.0, 6.0)
            gain = 10 ** (gain_db / 20.0)
            audio_window = audio_window * gain

        if self.random_time_shift:
            max_shift_samples = int(0.25 * sample_rate)
            shift_samples = np.random.randint(
                -max_shift_samples,
                max_shift_samples + 1,
            )

            shifted = np.zeros_like(audio_window)

            if shift_samples > 0:
                shifted[shift_samples:] = audio_window[:-shift_samples]
            elif shift_samples < 0:
                shifted[:shift_samples] = audio_window[-shift_samples:]
            else:
                shifted[:] = audio_window

            audio_window = shifted

        # This should never happen because the index only contains
        # complete windows, but fail loudly if the source changes.
        if len(audio_window) != window_samples:
            raise ValueError(
                f"Window is shorter than expected for {audio_path}: "
                f"{len(audio_window)} samples, expected {window_samples}"
            )

        return {
            "audio": torch.from_numpy(audio_window.copy()).float(),
            "sample_rate": sample_rate,
            "label": self.class_to_idx[row["primary_label"]],
            "species": row["primary_label"],
            "recording_id": row["filename"],
            "window_idx": window["window_idx"],
            "start_seconds": start_seconds,
        }


class BirdClefSpectrogramDataset(BirdClefWindowDataset):
    """BirdCLEF dataset returning log-power spectrograms."""

    def __getitem__(self, index: int) -> dict:
        from bird_monitoring.audio import compute_spectrogram

        item = super().__getitem__(index)

        frequencies, _, spectrogram = compute_spectrogram(
            item["audio"].numpy(),
            item["sample_rate"],
        )

        # Keep frequencies from 0 through 12 kHz.
        frequency_mask = frequencies <= 12000

        spectrogram = spectrogram[frequency_mask, :]

        # Normalize each spectrogram independently.
        mean = spectrogram.mean()
        std = spectrogram.std()

        if std > 0:
            spectrogram = (spectrogram - mean) / std

        # CNN input: [channel, frequency, time]
        spectrogram_tensor = torch.from_numpy(
            spectrogram.copy()
        ).float().unsqueeze(0)

        item["spectrogram"] = spectrogram_tensor

        return item

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import stft


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load audio and return mono floating-point samples plus sample rate."""

    audio, sample_rate = sf.read(path, dtype="float32")

    # Convert stereo/multichannel audio to mono.
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio, sample_rate


def compute_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    window_seconds: float = 0.032,
    hop_seconds: float = 0.016,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a log-power STFT spectrogram."""

    nperseg = int(sample_rate * window_seconds)
    noverlap = nperseg - int(sample_rate * hop_seconds)

    frequencies, times, spectrum = stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
    )

    power = np.abs(spectrum) ** 2
    spectrogram_db = 10 * np.log10(power + 1e-10)

    return frequencies, times, spectrogram_db


def save_spectrogram(
    audio_path: str | Path,
    output_path: str | Path,
    max_frequency: float = 12000,
) -> None:
    """Generate and save a bird-call spectrogram."""

    audio, sample_rate = load_audio(audio_path)

    frequencies, times, spectrogram_db = compute_spectrogram(
        audio,
        sample_rate,
    )

    frequency_mask = frequencies <= max_frequency

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.pcolormesh(
        times,
        frequencies[frequency_mask],
        spectrogram_db[frequency_mask],
        shading="auto",
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title(Path(audio_path).stem)

    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_waveform_and_spectrogram(
    audio_path: str | Path,
    output_path: str | Path,
    max_frequency: float = 12000,
) -> None:
    """Save waveform and spectrogram together for manual inspection."""

    audio, sample_rate = load_audio(audio_path)

    frequencies, times, spectrogram_db = compute_spectrogram(
        audio,
        sample_rate,
    )

    frequency_mask = frequencies <= max_frequency

    waveform_times = np.arange(len(audio)) / sample_rate

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
    )

    # Waveform
    axes[0].plot(waveform_times, audio)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(Path(audio_path).stem)

    # Spectrogram
    mesh = axes[1].pcolormesh(
        times,
        frequencies[frequency_mask],
        spectrogram_db[frequency_mask],
        shading="auto",
    )

    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh, ax=axes[1], label="Power (dB)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

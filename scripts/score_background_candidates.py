from pathlib import Path

import numpy as np
import pandas as pd

from bird_monitoring.audio import load_audio


METADATA_PATH = Path("data/birdclef/selected_metadata.csv")
AUDIO_DIR = Path("data/birdclef/audio")
OUTPUT_PATH = Path("data/birdclef/background_scores.csv")

WINDOW_SECONDS = 5.0
HOP_SECONDS = 2.5

N_FFT = 1024
HOP_LENGTH = 512


def spectral_features(audio: np.ndarray, sample_rate: int) -> dict:
    """Calculate simple spectral features for one audio window."""

    # Frame the signal.
    frame_length = N_FFT
    hop_length = HOP_LENGTH

    if len(audio) < frame_length:
        return {
            "spectral_flatness": 0.0,
            "spectral_entropy": 0.0,
            "spectral_centroid": 0.0,
        }

    frames = np.lib.stride_tricks.sliding_window_view(
        audio,
        frame_length,
    )[::hop_length]

    window = np.hanning(frame_length)
    frames = frames * window

    spectrum = np.abs(np.fft.rfft(frames, axis=1)) ** 2

    # Avoid numerical problems.
    spectrum += 1e-12

    # Normalize each frame into a probability distribution.
    probabilities = spectrum / spectrum.sum(axis=1, keepdims=True)

    # Spectral flatness:
    # geometric mean / arithmetic mean.
    log_spectrum = np.log(spectrum)
    geometric_mean = np.exp(log_spectrum.mean(axis=1))
    arithmetic_mean = spectrum.mean(axis=1)

    flatness = geometric_mean / arithmetic_mean

    # Spectral entropy.
    entropy = -np.sum(
        probabilities * np.log(probabilities),
        axis=1,
    )

    # Normalize entropy to [0, 1].
    entropy /= np.log(spectrum.shape[1])

    # Spectral centroid.
    frequencies = np.fft.rfftfreq(
        frame_length,
        d=1.0 / sample_rate,
    )

    centroid = (
        probabilities * frequencies[None, :]
    ).sum(axis=1)

    return {
        "spectral_flatness": float(np.mean(flatness)),
        "spectral_entropy": float(np.mean(entropy)),
        "spectral_centroid": float(np.mean(centroid)),
    }


def main():
    metadata = pd.read_csv(METADATA_PATH)

    metadata = metadata[
        (metadata["split"] == "train")
        & (metadata["duration_seconds"] >= WINDOW_SECONDS)
    ].copy()

    candidates = []

    print(f"Training recordings: {len(metadata)}")

    for row in metadata.itertuples(index=False):
        audio_path = AUDIO_DIR / Path(row.filename).name
        audio, sample_rate = load_audio(audio_path)

        window_samples = int(
            round(WINDOW_SECONDS * sample_rate)
        )
        hop_samples = int(
            round(HOP_SECONDS * sample_rate)
        )

        for start_sample in range(
            0,
            len(audio) - window_samples + 1,
            hop_samples,
        ):
            end_sample = start_sample + window_samples
            audio_window = audio[start_sample:end_sample]

            rms = float(
                np.sqrt(np.mean(audio_window ** 2))
            )

            features = spectral_features(
                audio_window,
                sample_rate,
            )

            candidates.append(
                {
                    "filename": row.filename,
                    "primary_label": row.primary_label,
                    "split": row.split,
                    "start_seconds": start_sample / sample_rate,
                    "rms": rms,
                    **features,
                }
            )

    df = pd.DataFrame(candidates)

    # Rank each feature independently.
    #
    # Lower RMS = quieter.
    # Higher flatness = more noise-like.
    # Higher entropy = less concentrated spectrum.
    #
    # These are NOT "background probabilities."
    # They are simply useful ranking signals.

    df["rms_rank"] = df["rms"].rank(pct=True)
    df["flatness_rank"] = df["spectral_flatness"].rank(pct=True)
    df["entropy_rank"] = df["spectral_entropy"].rank(pct=True)

    # Candidate score:
    # quiet + noise-like + spectrally distributed.
    df["background_score"] = (
        (1.0 - df["rms_rank"])
        + df["flatness_rank"]
        + df["entropy_rank"]
    ) / 3.0

    df = df.sort_values(
        "background_score",
        ascending=False,
    ).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Total windows: {len(df)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nTop 30 background candidates:")

    columns = [
        "filename",
        "primary_label",
        "start_seconds",
        "rms",
        "spectral_flatness",
        "spectral_entropy",
        "spectral_centroid",
        "background_score",
    ]

    print(
        df[columns].head(30).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


if __name__ == "__main__":
    main()

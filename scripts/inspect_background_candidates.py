from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bird_monitoring.audio import load_audio, compute_spectrogram


METADATA_PATH = Path("data/birdclef/background_candidates.csv")
AUDIO_DIR = Path("data/birdclef/audio")
OUTPUT_PATH = Path("outputs/background_candidates.png")

NUM_CANDIDATES = 20


def main():
    candidates = pd.read_csv(
        "data/birdclef/background_scores.csv"
    )

    # Avoid filling the contact sheet with many windows from one recording.
    candidates = (
        candidates.sort_values(
            "background_score",
            ascending=False,
        )
        .drop_duplicates(subset=["filename"])
        .head(NUM_CANDIDATES)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        5,
        4,
        figsize=(16, 18),
    )

    for ax, row in zip(axes.flat, candidates.itertuples(index=False)):
        audio_path = AUDIO_DIR / Path(row.filename).name
        audio, sample_rate = load_audio(audio_path)

        start_sample = int(round(row.start_seconds * sample_rate))
        window_samples = int(round(5.0 * sample_rate))

        audio_window = audio[
            start_sample:start_sample + window_samples
        ]

        frequencies, times, spectrogram = compute_spectrogram(
            audio_window,
            sample_rate,
        )

        mask = frequencies <= 12000

        spectrogram = spectrogram[mask]
        frequencies = frequencies[mask]

        ax.imshow(
            spectrogram,
            aspect="auto",
            origin="lower",
            extent=[
                times[0],
                times[-1],
                frequencies[0],
                frequencies[-1],
            ],
        )

        ax.set_title(
            f"{row.primary_label} | "
            f"{row.filename.split('/')[-1]}\n"
            f"start={row.start_seconds:.1f}s | "
            f"RMS={row.rms:.5f}",
            fontsize=8,
        )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hz")

    # Hide unused axes.
    for ax in axes.flat[len(candidates):]:
        ax.axis("off")

    fig.suptitle(
        "Acoustic Background Candidates",
        fontsize=16,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)

    print(f"Saved contact sheet to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

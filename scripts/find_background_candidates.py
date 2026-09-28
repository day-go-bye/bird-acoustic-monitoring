from pathlib import Path

import numpy as np
import pandas as pd

from bird_monitoring.audio import load_audio


METADATA_PATH = Path("data/birdclef/selected_metadata.csv")
AUDIO_DIR = Path("data/birdclef/audio")
OUTPUT_PATH = Path("data/birdclef/background_candidates.csv")

WINDOW_SECONDS = 5.0
HOP_SECONDS = 2.5


def main():
    metadata = pd.read_csv(METADATA_PATH)

    # Only use training recordings. Validation/test recordings must remain
    # completely untouched when we eventually create training background data.
    metadata = metadata[
        (metadata["split"] == "train")
        & (metadata["duration_seconds"] >= WINDOW_SECONDS)
    ].copy()

    print(f"Training recordings: {len(metadata)}")

    candidates = []

    for row in metadata.itertuples(index=False):
        audio_path = AUDIO_DIR / Path(row.filename).name
        audio, sample_rate = load_audio(audio_path)

        window_samples = int(round(WINDOW_SECONDS * sample_rate))
        hop_samples = int(round(HOP_SECONDS * sample_rate))

        for start_sample in range(
            0,
            len(audio) - window_samples + 1,
            hop_samples,
        ):
            end_sample = start_sample + window_samples
            audio_window = audio[start_sample:end_sample]

            rms = float(np.sqrt(np.mean(audio_window ** 2)))

            candidates.append(
                {
                    "filename": row.filename,
                    "primary_label": row.primary_label,
                    "split": row.split,
                    "start_seconds": start_sample / sample_rate,
                    "rms": rms,
                }
            )

    candidates_df = pd.DataFrame(candidates)
    candidates_df = candidates_df.sort_values("rms").reset_index(drop=True)

    candidates_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Total candidate windows: {len(candidates_df)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nQuietest 30 windows:")
    print(
        candidates_df.head(30).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nQuietest 5 windows per species:")
    print(
        candidates_df.groupby("primary_label", group_keys=False)
        .head(5)
        .sort_values(["primary_label", "rms"])
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


if __name__ == "__main__":
    main()

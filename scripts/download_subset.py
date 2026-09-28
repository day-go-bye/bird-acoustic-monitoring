import subprocess
from pathlib import Path

import pandas as pd


SPECIES = [
    "fepowl",
    "osprey",
    "houspa",
    "banana",
    "soulap1",
]

RECORDINGS_PER_SPECIES = 50
RANDOM_SEED = 42

DATA_DIR = Path("data/birdclef")
AUDIO_DIR = DATA_DIR / "audio"
TRAIN_CSV = DATA_DIR / "train.csv"
OUTPUT_METADATA = DATA_DIR / "selected_metadata.csv"


def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_CSV)

    selected_parts = []

    for species in SPECIES:
        rows = train[train["primary_label"] == species].copy()

        if len(rows) < RECORDINGS_PER_SPECIES:
            raise ValueError(
                f"{species} only has {len(rows)} recordings, "
                f"but {RECORDINGS_PER_SPECIES} were requested."
            )

        selected = rows.sample(
            n=RECORDINGS_PER_SPECIES,
            random_state=RANDOM_SEED,
        )

        selected_parts.append(selected)

    selected = pd.concat(selected_parts, ignore_index=True)

    print(f"Selected {len(selected)} recordings:")
    print(selected["primary_label"].value_counts().to_string())

    # Save metadata before downloading so the selection itself is reproducible.
    selected.to_csv(OUTPUT_METADATA, index=False)

    failures = []

    for _, row in selected.iterrows():
        source_path = row["filename"]
        filename = Path(source_path).name
        local_path = AUDIO_DIR / filename

        if local_path.exists():
            print(f"SKIP  {filename}")
            continue

        kaggle_path = f"train_audio/{source_path}"

        print(f"GET   {kaggle_path}")

        result = subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                "birdclef-2026",
                "-f",
                kaggle_path,
                "-p",
                str(AUDIO_DIR),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"FAIL  {filename}")
            print(result.stderr.strip())
            failures.append(source_path)

    print()
    print("Download complete.")
    print(f"Metadata: {OUTPUT_METADATA}")
    print(f"Audio directory: {AUDIO_DIR}")
    print(f"Successful files: {len(selected) - len(failures)}/{len(selected)}")

    if failures:
        print("\nFailed downloads:")
        for path in failures:
            print(f"  {path}")


if __name__ == "__main__":
    main()

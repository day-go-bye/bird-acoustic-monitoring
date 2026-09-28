from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


METADATA_PATH = Path("data/birdclef/selected_metadata.csv")
RANDOM_SEED = 42
MIN_DURATION_SECONDS = 5.0


def main():
    metadata = pd.read_csv(METADATA_PATH)

    usable = metadata[
        metadata["duration_seconds"] >= MIN_DURATION_SECONDS
    ].copy()

    print(f"Total recordings: {len(metadata)}")
    print(f"Usable recordings (>= {MIN_DURATION_SECONDS}s): {len(usable)}")

    # First split: 70% train, 30% temporary.
    train, temp = train_test_split(
        usable,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=usable["primary_label"],
    )

    # Split the remaining 30% evenly into validation and test.
    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp["primary_label"],
    )

    train = train.copy()
    val = val.copy()
    test = test.copy()

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    split_metadata = pd.concat(
        [train, val, test],
        ignore_index=True,
    )

    split_metadata = split_metadata.sort_values(
        ["split", "primary_label", "filename"]
    ).reset_index(drop=True)

    split_metadata.to_csv(METADATA_PATH, index=False)

    print("\nSaved split assignments to:")
    print(METADATA_PATH)

    print("\nSplit counts:")
    print(split_metadata["split"].value_counts())

    print("\nSpecies by split:")
    print(
        pd.crosstab(
            split_metadata["primary_label"],
            split_metadata["split"],
        )
    )


if __name__ == "__main__":
    main()

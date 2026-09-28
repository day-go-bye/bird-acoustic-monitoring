from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from bird_monitoring.dataset import BirdClefSpectrogramDataset
from bird_monitoring.model import BirdCNN


METADATA_PATH = "data/birdclef/selected_metadata.csv"
AUDIO_DIR = "data/birdclef/audio"
CHECKPOINT_PATH = "outputs/training/best_model.pt"

BATCH_SIZE = 16


def aggregate_recording_predictions(predictions):
    """Average window probabilities into one prediction per recording."""
    grouped = defaultdict(list)

    for prediction in predictions:
        grouped[prediction["recording_id"]].append(prediction)

    recording_results = []

    for recording_id, windows in grouped.items():
        probabilities = np.stack(
            [window["probabilities"] for window in windows]
        )
        mean_probabilities = probabilities.mean(axis=0)

        recording_results.append(
            {
                "recording_id": recording_id,
                "true_label": windows[0]["true_label"],
                "predicted_label": int(mean_probabilities.argmax()),
            }
        )

    return recording_results


def main():
    device = torch.device("cpu")

    dataset = BirdClefSpectrogramDataset(
        metadata_path=METADATA_PATH,
        audio_dir=AUDIO_DIR,
        split="test",
    )

    class_names = [
        dataset.idx_to_class[i]
        for i in range(len(dataset.idx_to_class))
    ]

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = BirdCNN(num_classes=len(class_names))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    window_predictions = []

    with torch.no_grad():
        for batch in loader:
            spectrograms = batch["spectrogram"].to(device)

            logits = model(spectrograms)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

            for i in range(len(probabilities)):
                window_predictions.append(
                    {
                        "recording_id": batch["recording_id"][i],
                        "true_label": int(batch["label"][i]),
                        "probabilities": probabilities[i],
                    }
                )

    recording_predictions = aggregate_recording_predictions(
        window_predictions
    )

    y_true = [
        result["true_label"]
        for result in recording_predictions
    ]

    y_pred = [
        result["predicted_label"]
        for result in recording_predictions
    ]

    print()
    print("=" * 60)
    print("RECORDING-LEVEL TEST EVALUATION")
    print("=" * 60)
    print()

    print(f"Test recordings: {len(recording_predictions)}")
    print(f"Test windows:    {len(window_predictions)}")
    print()

    print("PER-SPECIES METRICS")
    print("-" * 60)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=3,
        zero_division=0,
    )

    print(report)

    print("CONFUSION MATRIX")
    print("-" * 60)
    print("Rows = true species")
    print("Cols = predicted species")
    print()

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    header = " " * 16 + " ".join(f"{name:>10}" for name in class_names)
    print(header)

    for name, row in zip(class_names, matrix):
        values = " ".join(f"{value:10d}" for value in row)
        print(f"{name:>16} {values}")

    print()
    print("CLASS NAMES")
    print("-" * 60)

    for index, name in enumerate(class_names):
        print(f"{index}: {name}")


if __name__ == "__main__":
    main()

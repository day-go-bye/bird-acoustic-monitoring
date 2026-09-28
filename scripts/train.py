from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from bird_monitoring.dataset import BirdClefSpectrogramDataset
from bird_monitoring.model import BirdCNN


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

METADATA_PATH = "data/birdclef/selected_metadata.csv"
AUDIO_DIR = "data/birdclef/audio"

OUTPUT_DIR = Path("outputs/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
MAX_EPOCHS = 20
PATIENCE = 5

RANDOM_SEED = 42


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

device = torch.device("cpu")

print(f"Using device: {device}")


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------

train_dataset = BirdClefSpectrogramDataset(
    metadata_path=METADATA_PATH,
    audio_dir=AUDIO_DIR,
    split="train",
    random_crop=True,
)

val_dataset = BirdClefSpectrogramDataset(
    metadata_path=METADATA_PATH,
    audio_dir=AUDIO_DIR,
    split="val",
)

test_dataset = BirdClefSpectrogramDataset(
    metadata_path=METADATA_PATH,
    audio_dir=AUDIO_DIR,
    split="test",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

num_classes = len(train_dataset.class_to_idx)

model = BirdCNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate(model, loader, dataset, device):
    """Evaluate both window-level and recording-level performance."""

    model.eval()

    all_predictions = []
    all_targets = []

    # recording_id -> list of probability vectors
    recording_probabilities = {}

    # recording_id -> true class
    recording_targets = {}

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            spectrograms = batch["spectrogram"].to(device)
            targets = batch["label"].to(device)

            logits = model(spectrograms)
            loss = criterion(logits, targets)

            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            batch_size = targets.size(0)

            total_loss += loss.item() * batch_size
            total_examples += batch_size

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )
            all_targets.extend(
                targets.cpu().numpy().tolist()
            )

            for i, recording_id in enumerate(batch["recording_id"]):
                recording_id = str(recording_id)

                if recording_id not in recording_probabilities:
                    recording_probabilities[recording_id] = []

                recording_probabilities[recording_id].append(
                    probabilities[i].cpu().numpy()
                )

                recording_targets[recording_id] = int(
                    targets[i].item()
                )

    # Window-level metrics.
    window_accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    window_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    # Recording-level metrics.
    recording_predictions = []
    recording_true = []

    for recording_id, probabilities in recording_probabilities.items():
        mean_probability = np.mean(
            probabilities,
            axis=0,
        )

        prediction = int(np.argmax(mean_probability))
        target = recording_targets[recording_id]

        recording_predictions.append(prediction)
        recording_true.append(target)

    recording_accuracy = accuracy_score(
        recording_true,
        recording_predictions,
    )

    recording_f1 = f1_score(
        recording_true,
        recording_predictions,
        average="macro",
        zero_division=0,
    )

    average_loss = total_loss / total_examples

    return {
        "loss": average_loss,
        "window_accuracy": window_accuracy,
        "window_f1": window_f1,
        "recording_accuracy": recording_accuracy,
        "recording_f1": recording_f1,
    }


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

print()
print("Training configuration:")
print(f"  Training recordings:   {len(train_dataset.recordings)}")
print(f"  Training windows:      {len(train_dataset)}")
print(f"  Validation recordings: {len(val_dataset.recordings)}")
print(f"  Validation windows:    {len(val_dataset)}")
print(f"  Test recordings:       {len(test_dataset.recordings)}")
print(f"  Test windows:          {len(test_dataset)}")
print(f"  Classes:               {train_dataset.class_to_idx}")
print(f"  Batch size:            {BATCH_SIZE}")
print(f"  Learning rate:         {LEARNING_RATE}")
print(f"  Max epochs:            {MAX_EPOCHS}")
print()

best_val_f1 = -float("inf")
epochs_without_improvement = 0

history = []

best_checkpoint_path = OUTPUT_DIR / "best_model.pt"

for epoch in range(1, MAX_EPOCHS + 1):

    model.train()

    running_loss = 0.0
    training_examples = 0

    for batch in train_loader:
        spectrograms = batch["spectrogram"].to(device)
        targets = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(spectrograms)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)

        running_loss += loss.item() * batch_size
        training_examples += batch_size

    train_loss = running_loss / training_examples

    validation = evaluate(
        model,
        val_loader,
        val_dataset,
        device,
    )

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": validation["loss"],
        "val_window_accuracy": validation["window_accuracy"],
        "val_window_f1": validation["window_f1"],
        "val_recording_accuracy": validation["recording_accuracy"],
        "val_recording_f1": validation["recording_f1"],
    }

    history.append(row)

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"train loss {train_loss:.4f} | "
        f"val loss {validation['loss']:.4f} | "
        f"window F1 {validation['window_f1']:.3f} | "
        f"recording F1 {validation['recording_f1']:.3f}"
    )

    # Select the model using recording-level validation F1.
    if validation["recording_f1"] > best_val_f1:
        best_val_f1 = validation["recording_f1"]
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_dataset.class_to_idx,
                "config": {
                    "window_seconds": train_dataset.window_seconds,
                    "hop_seconds": train_dataset.hop_seconds,
                    "max_windows_per_recording": (
                        train_dataset.max_windows_per_recording
                    ),
                    "learning_rate": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                },
                "epoch": epoch,
                "val_recording_f1": best_val_f1,
            },
            best_checkpoint_path,
        )

        print(
            f"  Saved best model "
            f"(recording F1={best_val_f1:.3f})"
        )

    else:
        epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(
                f"\nEarly stopping after {epoch} epochs."
            )
            break


# ---------------------------------------------------------------------
# Save training history
# ---------------------------------------------------------------------

history_path = OUTPUT_DIR / "training_history.csv"

pd.DataFrame(history).to_csv(
    history_path,
    index=False,
)


# ---------------------------------------------------------------------
# Load best model
# ---------------------------------------------------------------------

checkpoint = torch.load(
    best_checkpoint_path,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ---------------------------------------------------------------------
# Final test evaluation
# ---------------------------------------------------------------------

test_results = evaluate(
    model,
    test_loader,
    test_dataset,
    device,
)

print()
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Window accuracy:    {test_results['window_accuracy']:.3f}")
print(f"Window macro F1:    {test_results['window_f1']:.3f}")
print(f"Recording accuracy: {test_results['recording_accuracy']:.3f}")
print(f"Recording macro F1: {test_results['recording_f1']:.3f}")

print()
print(f"Best model: {best_checkpoint_path}")
print(f"History:    {history_path}")

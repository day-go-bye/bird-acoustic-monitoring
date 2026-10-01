from pathlib import Path

import numpy as np
import torch

from bird_monitoring.audio import compute_spectrogram, load_audio
from bird_monitoring.model import BirdCNN


class BirdInference:
    """Load a trained bird classifier and run inference on recordings."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        self.class_to_idx = checkpoint["class_to_idx"]
        self.idx_to_class = {
            index: label
            for label, index in self.class_to_idx.items()
        }

        config = checkpoint["config"]
        self.window_seconds = config["window_seconds"]
        self.hop_seconds = config["hop_seconds"]

        self.model = BirdCNN(
            num_classes=len(self.class_to_idx),
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

    def predict_recording(
        self,
        audio_path: str | Path,
    ) -> list[dict]:
        """Run window-level predictions across a recording."""

        audio, sample_rate = load_audio(audio_path)

        window_samples = int(
            round(self.window_seconds * sample_rate)
        )
        hop_samples = int(
            round(self.hop_seconds * sample_rate)
        )

        predictions = []

        for start_sample in range(
            0,
            len(audio) - window_samples + 1,
            hop_samples,
        ):
            end_sample = start_sample + window_samples
            audio_window = audio[start_sample:end_sample]

            frequencies, _, spectrogram = compute_spectrogram(
                audio_window,
                sample_rate,
            )

            frequency_mask = frequencies <= 12000
            spectrogram = spectrogram[frequency_mask, :]

            mean = spectrogram.mean()
            std = spectrogram.std()

            if std > 0:
                spectrogram = (spectrogram - mean) / std

            tensor = torch.from_numpy(
                spectrogram.copy()
            ).float().unsqueeze(0).unsqueeze(0)

            tensor = tensor.to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probabilities = torch.softmax(logits, dim=1)

            predicted_index = int(
                probabilities.argmax(dim=1).item()
            )
            confidence = float(
                probabilities[0, predicted_index].item()
            )

            predictions.append(
                {
                    "species": self.idx_to_class[predicted_index],
                    "confidence": confidence,
                    "start_seconds": start_sample / sample_rate,
                    "end_seconds": end_sample / sample_rate,
                }
            )

        return predictions
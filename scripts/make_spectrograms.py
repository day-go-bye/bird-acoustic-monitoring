from pathlib import Path

from bird_monitoring.audio import save_spectrogram


for audio_path in Path("data/wav").glob("*.wav"):
    output_path = Path("outputs/spectrograms") / f"{audio_path.stem}.png"
    save_spectrogram(audio_path, output_path)
    print(f"Created {output_path}")

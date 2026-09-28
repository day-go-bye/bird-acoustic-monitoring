# Bird Acoustic Monitoring

An end-to-end prototype for passive acoustic bird monitoring, exploring how recorded environmental audio can be transformed into species-level bird detections.

The project uses real BirdCLEF recordings to investigate the pipeline from raw audio through spectrogram generation, temporal windowing, convolutional classification, and recording-level evaluation.

## Project Goals

This project is designed as a practical exploration of problems encountered in passive acoustic biodiversity monitoring:

* ingesting and preprocessing field recordings
* converting audio into ML-ready spectrograms
* detecting species from short acoustic windows
* aggregating window-level predictions into recording-level detections
* evaluating models without leaking information between recordings
* investigating augmentation and environmental noise
* understanding the challenges of ambiguous acoustic background

The eventual goal is to explore architectures and data pipelines that could support scalable biodiversity monitoring from autonomous acoustic recordings.

## Current Experiment

The current prototype uses five bird species from the BirdCLEF 2026 dataset:

| Label     | Species               |
| --------- | --------------------- |
| `banana`  | Bananaquit            |
| `fepowl`  | Ferruginous Pygmy Owl |
| `houspa`  | House Sparrow         |
| `osprey`  | Osprey                |
| `soulap1` | Southern Lapwing      |

The initial dataset contains 250 selected recordings. After excluding recordings shorter than the 5-second analysis window, 230 recordings remain.

The data is split **at the recording level before generating training windows**:

* 161 training recordings
* 34 validation recordings
* 35 test recordings

This prevents multiple windows from the same recording from appearing across different splits, which could otherwise allow the model to learn recording-specific background, microphone, or environmental characteristics.

## Audio Pipeline

Each usable recording is divided into 5-second windows with a 2.5-second hop.

The current spectrogram configuration uses:

* 32 kHz mono audio
* 32 ms Hann analysis window
* 16 ms hop
* log-power spectrogram
* frequency range through 12 kHz for model input

The resulting spectrogram is normalized per example before being passed to the classifier.

```text
BirdCLEF recording
       │
       ▼
5-second audio windows
       │
       ▼
STFT / log-power spectrogram
       │
       ▼
per-example normalization
       │
       ▼
CNN classifier
       │
       ▼
window-level species probabilities
       │
       ▼
recording-level aggregation
```

## Model

The current baseline is a small convolutional neural network implemented in PyTorch.

It contains:

* three convolutional blocks
* batch normalization
* ReLU activations
* 2×2 max pooling
* adaptive global pooling
* dropout
* a five-class linear classifier

The model has approximately 94,000 trainable parameters.

The emphasis at this stage is on establishing a reproducible end-to-end pipeline rather than maximizing model size.

## Results

The best experiment so far uses **random temporal cropping** during training.

| Experiment               | Window Accuracy | Window Macro F1 | Recording Accuracy | Recording Macro F1 |
| ------------------------ | --------------: | --------------: | -----------------: | -----------------: |
| Fixed windows            |           0.436 |           0.443 |              0.457 |              0.443 |
| **Random temporal crop** |       **0.508** |       **0.523** |          **0.543** |          **0.553** |
| Random crop + gain       |           0.475 |           0.470 |              0.514 |              0.474 |
| Random crop + time shift |           0.470 |           0.471 |              0.457 |              0.456 |

The improvement from random cropping suggests that exposing the model to different temporal portions of the same recordings provides useful augmentation.

Macro F1 is reported because the goal is to measure performance across species rather than allowing the most frequent or easiest classes to dominate the metric.

## What I've Learned So Far

One of the more interesting challenges has been defining a reliable acoustic "background" class.

An initial experiment ranked low-energy windows as potential background examples. However, listening to these recordings showed that many of the quietest windows still contained identifiable bird vocalizations.

A second experiment incorporated spectral flatness and spectral entropy to find more noise-like environmental segments. These candidates also frequently contained birds.

This highlights an important property of passive acoustic recordings:

> Low-energy or acoustically complex audio is not necessarily a negative example.

Environmental recordings can contain multiple simultaneously vocalizing species, insects, wind, and other acoustic activity. Treating these ambiguous windows as a clean `background` class could introduce label noise.

The current direction is therefore to investigate environmental audio primarily as **realistic acoustic augmentation**, while keeping the labeled bird recordings as positive examples.

## Repository Structure

```text
bird-acoustic-monitoring/
├── src/
│   └── bird_monitoring/
│       ├── audio.py
│       ├── dataset.py
│       └── model.py
├── scripts/
│   ├── train.py
│   ├── split_dataset.py
│   ├── download_subset.py
│   ├── find_background_candidates.py
│   ├── score_background_candidates.py
│   └── inspect_background_candidates.py
├── data/
│   └── metadata and local datasets
├── outputs/
│   └── generated diagnostics and model results
└── README.md
```

Raw datasets, downloaded audio, generated outputs, and local environment files are excluded from version control.

## Reproducibility

The experiments use:

* Python 3.12
* PyTorch
* NumPy
* SciPy
* pandas
* soundfile
* matplotlib
* scikit-learn

The training experiments use a fixed random seed for reproducibility.

The current implementation uses CPU inference/training on the development machine.

## Next Steps

Planned experiments include:

1. realistic environmental-noise augmentation
2. stronger audio representations and CNN architectures
3. improved recording-level aggregation
4. confidence calibration
5. multi-species / overlapping-vocalization handling
6. explicit bird-event detection
7. human-review workflows for uncertain detections
8. scaling the pipeline toward larger biodiversity monitoring datasets

The long-term goal is to move from a five-species classification experiment toward a more realistic passive acoustic monitoring system capable of producing interpretable, reviewable biodiversity observations.

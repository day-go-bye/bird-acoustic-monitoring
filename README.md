# Bird Acoustic Monitoring

An end-to-end prototype for passive acoustic biodiversity monitoring, combining wildlife audio analysis with a small production-style data platform.

The project explores how machine-learning models can be integrated into a workflow where field recordings are ingested, processed asynchronously, converted into model observations, and reviewed by a human.

## Why this project

Passive acoustic monitoring can produce large volumes of wildlife recordings that are difficult to analyze manually. A useful system therefore needs more than a classifier: it needs infrastructure for ingesting recordings, running analysis, storing observations and model provenance, and allowing domain experts to review automated results.

This project is a small-scale implementation of that workflow.

## Architecture

```text
                    Field Recording
                          |
                          v
                  +---------------+
                  |  Recording API |
                  +-------+-------+
                          |
                          v
                  +---------------+
                  | Local Storage |
                  +-------+-------+
                          |
                          v
                  +---------------+
                  | Processing Job|
                  |     Queue     |
                  +-------+-------+
                          |
                          v
                  +---------------+
                  | ML Inference  |
                  +-------+-------+
                          |
                          v
                  +---------------+
                  | Observations  |
                  | + provenance  |
                  +-------+-------+
                          |
                          v
                  +---------------+
                  | Human Review  |
                  |  UI / API     |
                  +---------------+
```

The current implementation uses SQLite and local filesystem storage so that the entire system can run locally. The interfaces are designed so that these components could later be replaced with managed database and object-storage services.

## Current capabilities

### Audio ingestion

The FastAPI service accepts audio recordings together with metadata including:

* recording timestamp
* latitude / longitude
* duration
* source
* filename

Audio is stored using server-generated filenames rather than trusting client-provided paths.

### Asynchronous processing

Recordings create processing jobs rather than performing ML inference directly inside the API request.

A separate worker:

1. claims a queued processing job
2. loads the recording
3. runs model inference
4. creates observation records
5. stores model version information
6. marks the job and recording as completed

Failures are persisted on the processing job so they can be inspected rather than disappearing with a request.

### Machine-learning inference

The current model is a small convolutional neural network trained on a subset of the BirdCLEF 2026 dataset.

The prototype currently recognizes five species:

* Bananaquit (`banana`)
* Ferruginous Pygmy Owl (`fepowl`)
* House Sparrow (`houspa`)
* Osprey (`osprey`)
* Southern Lapwing (`soulap1`)

The model operates on 5-second audio windows and produces an observation containing:

* predicted species
* confidence
* start/end timestamp
* model version
* review status

The inference layer is separated from the API and worker so that the model can be replaced without redesigning the surrounding platform.

## Human-in-the-loop review

Automated predictions are explicitly stored as `unreviewed` observations.

The review interface allows a reviewer to inspect:

* predicted species
* model confidence
* model version
* recording metadata
* geographic coordinates
* the exact audio segment associated with the prediction
* a generated spectrogram

The reviewer can mark an observation:

* `confirmed`
* `rejected`
* `uncertain`

The review queue is driven by observation state, so reviewed observations automatically disappear from the active queue.

This creates a simple human-in-the-loop workflow rather than treating model predictions as ground truth.

## Data and model experiments

The initial ML experiments use a deterministic subset of BirdCLEF 2026.

### Dataset subset

* 5 species
* 50 recordings selected per species
* 250 recordings initially selected
* 230 recordings at least 5 seconds long
* recording-level train/validation/test split
* 161 training recordings
* 34 validation recordings
* 35 test recordings

Audio is converted to 32 kHz mono.

### Spectrogram representation

The current representation uses:

* 32 ms Hann window
* 16 ms hop
* log-power spectrogram
* frequencies up to 12 kHz
* per-example normalization

### Model

The current CNN is intentionally small at approximately 94,000 parameters.

The experiment compared several temporal and audio augmentations. Random temporal cropping produced the strongest result in the initial experiments.

Current best experiment:

| Metric             | Result |
| ------------------ | -----: |
| Window accuracy    |  50.8% |
| Window macro F1    |  52.3% |
| Recording accuracy |  54.3% |
| Recording macro F1 |  55.3% |

These results are an early baseline rather than a production-quality species identification system.

## Important limitation

The current classifier is a closed-set five-species model. It must choose one of those five classes even when a recording contains a species outside the training set.

For example, the repository includes recordings used during platform testing that are not members of the five training classes. Low-confidence predictions on those recordings demonstrate an important limitation of closed-set classification rather than evidence of a correct identification.

The review workflow therefore treats predictions as hypotheses requiring human validation.

A production biodiversity system would need mechanisms such as:

* unknown / out-of-distribution detection
* confidence calibration
* event detection
* overlapping-species handling
* improved representations
* larger and more geographically diverse datasets
* domain-expert review and feedback

## Project structure

```text
bird-acoustic-monitoring/
├── api/
│   ├── database.py
│   ├── main.py
│   ├── models.py
│   ├── run_worker.py
│   ├── schemas.py
│   ├── storage.py
│   ├── worker.py
│   └── static/
│       └── review.html
├── data/
│   ├── metadata.csv
│   └── ...
├── scripts/
│   ├── download_subset.py
│   ├── evaluate.py
│   ├── find_background_candidates.py
│   ├── inspect_background_candidates.py
│   ├── make_spectrograms.py
│   ├── score_background_candidates.py
│   ├── split_dataset.py
│   └── train.py
├── src/
│   └── bird_monitoring/
│       ├── audio.py
│       ├── dataset.py
│       ├── inference.py
│       ├── model.py
│       └── ...
├── requirements.txt
└── README.md
```

## Running locally

Create and activate a Python 3.12 virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
PYTHONPATH=src uvicorn api.main:app --reload
```

The API is available at:

```text
http://127.0.0.1:8000
```

The interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

The human review interface is available at:

```text
http://127.0.0.1:8000/review
```

Start the processing worker separately:

```bash
PYTHONPATH=src python -m api.run_worker
```

## Future platform work

The current prototype deliberately keeps the infrastructure small. Potential next steps include:

### Event detection

Move from fixed-window classification toward:

```text
long recording
      |
      v
candidate vocalization detection
      |
      v
species classification
      |
      v
observation
```

This would allow long field recordings to produce observations only when candidate biological events are detected.

### Production storage

Replace local filesystem storage and SQLite with:

* object storage for recordings and derived artifacts
* PostgreSQL for metadata and observations

The storage abstraction is intentionally kept separate from API logic to make this transition straightforward.

### Scalable processing

The current worker polls a local database. A production deployment could replace this with a managed queue while preserving the same job lifecycle:

```text
queued → processing → completed
                  ↘ failed
```

### Model provenance and monitoring

Future observations could include additional provenance such as:

* model hash
* training dataset version
* preprocessing version
* inference configuration
* calibration information

This would make model outputs reproducible and auditable as models evolve.

### Geospatial analysis

Recording coordinates could support:

* spatial filtering
* habitat/ecological context
* geographic observation aggregation
* integration with GIS workflows

### Review feedback

Human decisions could eventually feed back into model evaluation and retraining, creating a continuous improvement loop between automated analysis and domain expertise.

## Technology

* Python
* FastAPI
* SQLAlchemy
* SQLite
* PyTorch
* SciPy
* NumPy
* SoundFile
* Matplotlib
* scikit-learn
* BirdCLEF 2026

## Status

This is an actively developed prototype exploring the intersection of:

* machine learning
* passive acoustic monitoring
* biodiversity data platforms
* human-in-the-loop scientific workflows
* cloud-oriented backend architecture

The emphasis is on building the smallest useful system around an ML model rather than optimizing the classifier in isolation.

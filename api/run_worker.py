import time

from bird_monitoring.inference import BirdInference
from api.worker import process_next_job


POLL_INTERVAL_SECONDS = 2
MODEL_VERSION = "bird-cnn-0.1.0"


def main():
    print("Bird monitoring worker started.")

    inference = BirdInference(
        "outputs/training/best_model.pt"
    )

    print("Model loaded.")

    while True:
        job_id = process_next_job(
            inference,
            MODEL_VERSION,
        )

        if job_id is not None:
            print(f"Processed job: {job_id}")
        else:
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
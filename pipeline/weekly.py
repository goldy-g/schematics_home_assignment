"""The weekly job. The scheduler starts it every Sunday at 05:00, Israel time.

    python -m pipeline.weekly

1. Test mode: train on the older data, score the newest week, log the accuracy to MLflow.
2. Production mode: train, register a new version, smoke-check it, make it the champion.

The production run is tagged with the test run's id, so every version points to its own accuracy report.
(Both modes train on the same rows, quirk T1, so that report describes exactly the model being shipped.)
"""
from pipeline import train


def main():
    evaluation_run_id = train.run("test")
    train.run("production", evaluation_run_id=evaluation_run_id)


if __name__ == "__main__":
    main()

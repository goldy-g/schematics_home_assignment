"""Time the served model the way a server uses it: load a version once, then rank one visitor per call.

Usage: .venv/bin/python scripts/ranker_speed.py [calls] [version]     (default: 10 calls, @champion)
Prints the load time, the per-call times and the process's peak memory (to compare two versions).
"""
import resource
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import settings  # noqa: E402

settings.load_env()
import mlflow  # noqa: E402


def main():
    calls = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    settings.setup_mlflow()
    version = sys.argv[2] if len(sys.argv) > 2 else (
        mlflow.MlflowClient().get_model_version_by_alias(settings.MODEL_NAME, settings.CHAMPION).version)
    start = time.perf_counter()
    model = mlflow.pyfunc.load_model(f"models:/{settings.MODEL_NAME}/{version}")
    print(f"load v{version}: {time.perf_counter() - start:.1f} s")
    ranker = model.unwrap_python_model()
    seconds, sources, ranking = [], [], None
    for _ in range(calls):
        start = time.perf_counter()
        ranking, source = ranker.rank(dict(settings.EXAMPLE_VISITOR))
        seconds.append(time.perf_counter() - start)
        sources.append(source)
    ordered = sorted(seconds)
    print(f"{calls} calls: median {statistics.median(seconds):.2f} s, p90 {ordered[int(0.9 * (calls - 1))]:.2f} s, "
          f"max {ordered[-1]:.2f} s, fallbacks {sources.count('fallback')}")
    print(f"peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB")  # bytes on macOS
    print("ranking:", {brand: (int(v["rank"]), round(v["expected_payout"], 2)) for brand, v in ranking.items()})


if __name__ == "__main__":
    main()

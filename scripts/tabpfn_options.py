"""Task 1.8: measure ways to call TabPFN for one ranking request, and pick the most efficient.

Every option uses the production context (1,000 rows) and the same 10 request rows: their predictor's features for
the made-up visitor, one row per brand.
  A  as given             fit + predict on every request (their predictor)
  B  fit once             fit at start-up, predict per request
  C  fit once, cached     fit_mode="fit_with_cache": Prior Labs keeps the fitted context, predict per request
  D  saved fit            C's fit saved with save_model; a fresh estimator loads it and only predicts
C runs for three model versions. Then each version's payout accuracy on the test week (their test-mode split).
Writes outputs/tabpfn_options/results.json.  Usage: .venv/bin/python scripts/tabpfn_options.py
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import settings  # noqa: E402

settings.load_env()
from tabpfn_client import TabPFNRegressor, get_api_usage, set_access_token  # noqa: E402

import bl_exp_payout_predictor as their_predictor  # noqa: E402
import bl_models_train as their_trainer  # noqa: E402

OUT = ROOT / "outputs" / "tabpfn_options"
VERSIONS = ["v3.5_default", "v3_default", "v3.5-fast_default"]
REPEATS = 5
results = {}


def server_seconds(model):
    """Seconds Prior Labs' server spent on the last prediction (queues + work). The rest of the wait is network."""
    timings = model.get_timings().get("predict") or {}
    return sum(v for v in timings.values() if isinstance(v, (int, float)))


def measure(name, model, rows, reference, repeats=REPEATS):
    walls, servers = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        prediction = model.predict(rows)
        walls.append(time.perf_counter() - start)
        servers.append(server_seconds(model))
    diff = None if reference is None else float(np.max(np.abs(prediction - reference)))
    results[name] = {"seconds_each": [round(w, 2) for w in walls], "median_s": round(statistics.median(walls), 2),
                     "server_median_s": round(statistics.median(servers), 2), "max_dollar_diff_vs_A": diff}
    print(f"{name:<34} median {results[name]['median_s']:>6.2f} s  server {results[name]['server_median_s']:>5.2f} s"
          f"  calls {results[name]['seconds_each']}  diff vs A {diff}", flush=True)
    return prediction


def main():
    set_access_token(os.environ["TABPFN_TOKEN"])
    OUT.mkdir(parents=True, exist_ok=True)
    print("usage before:", get_api_usage(), flush=True)

    files = max((ROOT / "outputs" / "runs").glob("*/files"), key=lambda p: p.stat().st_mtime)  # newest production files
    context = joblib.load(files / "payout_tfm_context.joblib")
    predictor = their_predictor.BLPayoutModelsPredict(f"{files}/", None, f"{OUT}/")
    their_predictor.user_data = settings.EXAMPLE_VISITOR
    rows = predictor.bl_preprocessing()[context["columns"]]  # 10 rows: the visitor x each brand
    x, y = context["x"], context["y"]

    # A: as given, a new fit on every request
    walls, reference = [], None
    for _ in range(2):
        start = time.perf_counter()
        model = TabPFNRegressor(ignore_pretraining_limits=True)
        model.fit(x, y)
        reference = model.predict(rows)
        walls.append(time.perf_counter() - start)
    results["A as given (fit + predict)"] = {"seconds_each": [round(w, 2) for w in walls], "median_s": round(statistics.median(walls), 2)}
    print(f"A as given (fit + predict)          calls {results['A as given (fit + predict)']['seconds_each']}", flush=True)

    # B: fit once, predict per request (what our ranker does today)
    model = TabPFNRegressor(ignore_pretraining_limits=True)
    model.fit(x, y)
    measure("B fit once (auto model)", model, rows, reference)

    # C: fit once with the server-side cache, per model version
    cached = {}
    for version in VERSIONS:
        try:
            model = TabPFNRegressor(ignore_pretraining_limits=True, model_path=version, fit_mode="fit_with_cache")
            start = time.perf_counter()
            model.fit(x, y)
            results[f"fit seconds, {version} cached"] = round(time.perf_counter() - start, 2)
            measure(f"C cached, {version}", model, rows, reference)
            cached[version] = model
        except Exception as error:  # an alpha model can refuse; record it and go on
            results[f"C cached, {version}"] = {"error": f"{type(error).__name__}: {error}"[:300]}
            print(f"C cached, {version}: {results[f'C cached, {version}']['error']}", flush=True)

    # D: save the cached v3.5 fit, load it into a fresh estimator, predict without fitting
    if "v3.5_default" in cached:
        record = OUT / "fitted_v3.5.json"
        cached["v3.5_default"].save_model(record)
        reloaded = TabPFNRegressor.load_model(record)
        measure("D saved fit, v3.5 reloaded", reloaded, rows, reference, repeats=3)

    # Accuracy of each version on the test week, with their own test-mode split and context
    trainer = their_trainer.BLPayoutModelsFit(f"{settings.DATA_FILE.parent}/", settings.DATA_FILE.name, f"{OUT}/eval/", train_test=True)
    x_train, y_train, x_test, y_test = trainer.bl_preprocessing()
    x_pay, y_pay = trainer.prepare_for_cont_payout_prediction_train(x_train, y_train)
    x_test_pay, y_test_pay = trainer.prepare_for_cont_payout_prediction_test(x_test, y_test)
    _, test_context = trainer.tabpfn_regression_payout(x_pay, y_pay)  # their context: the newest 1,000 paid rows
    for version in ["auto"] + VERSIONS:
        try:
            model = TabPFNRegressor(ignore_pretraining_limits=True, model_path=version)
            model.fit(test_context["x"], test_context["y"])
            predicted = model.predict(x_test_pay)
            results[f"accuracy {version}"] = {
                "payout_mape": round(float(mean_absolute_percentage_error(y_test_pay["payout"], predicted)), 4),
                "payout_mae": round(float(mean_absolute_error(y_test_pay["payout"], predicted)), 3),
                "test_rows": int(len(predicted))}
        except Exception as error:
            results[f"accuracy {version}"] = {"error": f"{type(error).__name__}: {error}"[:300]}
        print(f"accuracy {version:<20} {results[f'accuracy {version}']}", flush=True)

    print("usage after:", get_api_usage(), flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"saved {OUT / 'results.json'}")


if __name__ == "__main__":
    main()

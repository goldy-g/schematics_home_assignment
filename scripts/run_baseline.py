"""Task 1.1 baseline: run the researcher's two scripts exactly as given, and time each step.

Their files (bl_models_train.py, bl_exp_payout_predictor.py) are imported, never edited.
Everything this script writes goes to outputs/baseline/ (git-ignored).

Usage:  .venv/bin/python scripts/run_baseline.py
Needs:  data/bl_full_data.csv, and TABPFN_TOKEN in .env
"""
import ast
import json
import os
import sys
import time
from pathlib import Path

import joblib
import tabpfn_client

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "baseline"
sys.path.insert(0, str(ROOT))  # their two files sit at the repo root

timings = {}  # step name -> seconds


def timed(step, fn, *args, **kwargs):
    """Run fn, remember how long it took, print one line."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    timings[step] = round(time.perf_counter() - start, 3)
    print(f"{step:<48} {timings[step]:>9.3f} s", flush=True)
    return result


def load_env(path=ROOT / ".env"):
    """Copy KEY=VALUE lines from .env into the environment. The token lives there, never in code."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def their_example_user():
    """The example visitor hard-coded in their predictor's __main__ block, read from their file (not copied here)."""
    tree = ast.parse((ROOT / "bl_exp_payout_predictor.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "user_data":
            return ast.literal_eval(node.value)
    raise RuntimeError("user_data not found in bl_exp_payout_predictor.py")


def main():
    load_env()
    if not os.environ.get("TABPFN_TOKEN"):
        sys.exit("TABPFN_TOKEN is missing: create .env first (see README)")

    tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
    print(f"TabPFN usage before the run:\n{tabpfn_client.get_api_usage()}\n")

    data_dir = f"{ROOT}/data/"                  # their code joins paths with +, so folders end with /
    test_dir = f"{OUT}/test_mode/"
    prod_dir = f"{OUT}/production/"

    # 1. Training, both modes, exactly as their class defines them
    trainer = timed("import trainer (loads the names database)", __import__, "bl_models_train")
    timed("train, test mode (train_test=True)",
          lambda: trainer.BLPayoutModelsFit(data_dir, "bl_full_data.csv", test_dir, train_test=True).fit_())
    timed("train, production mode (train_test=False)",
          lambda: trainer.BLPayoutModelsFit(data_dir, "bl_full_data.csv", prod_dir, train_test=False).fit_())

    # 2. Proof of the 7-day holdout: the production context should end on Feb 4-5, not Feb 10-11
    context = joblib.load(prod_dir + "payout_tfm_context.joblib")
    days = context["x"]["session_day"].value_counts().sort_index().to_dict()
    print(f"production context: {context['x'].shape[0]} rows x {context['x'].shape[1]} columns; "
          f"session_day counts {days}")

    # 3. The predictor, as given: one full call per request, the way their __main__ runs it
    predictor = timed("import predictor (loads the names database)", __import__, "bl_exp_payout_predictor")
    user = their_example_user()
    predictor.user_data = user  # their import_preprocess reads this module global (L58)
    log_dir = f"{OUT}/predictor_logs/"
    result = None
    for i in range(1, 4):
        result = timed(f"predict_() as given, call {i}",
                       lambda: predictor.BLPayoutModelsPredict(prod_dir, user, log_dir).predict_())

    # 4. Where the time goes inside one call: their methods, in the order predict_() calls them
    p = predictor.BLPayoutModelsPredict(prod_dir, user, log_dir)
    rows = timed("  features: user x brands", p.bl_preprocessing)
    model_tfm, columns, cb_model = timed("  load_models (CatBoost file + TabPFN fit)", p.load_models)
    timed("  CatBoost predict_proba, 10 rows", cb_model.predict_proba, rows[columns])
    timed("  TabPFN predict, 10 rows (internet)", model_tfm.predict, rows[columns])
    server = model_tfm.get_timings()  # seconds Prior Labs' server spent; the rest of the wait is network
    print(f"  TabPFN server-side timings: {server}")

    # 5. What one prediction costs, without spending quota, and what this run spent
    for operation in ("predict", "cache_predict"):
        print(f"  estimated cost, {operation:<13}: {tabpfn_client.estimate_cost(context['x'], rows[columns], operation=operation)}")
    print(f"\nTabPFN usage after the run:\n{tabpfn_client.get_api_usage()}")

    print("\nranking for their example user:")
    print(json.dumps(result, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(
        {"timings_s": timings, "context_session_day_counts": days, "example_ranking": result}, indent=2, default=str))
    print(f"\nsaved {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()

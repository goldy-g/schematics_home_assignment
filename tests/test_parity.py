"""Parity: our served model must rank exactly like their predictor, run as given, on the same model files.

Slow and online (their predictor re-fits TabPFN on every call; every call spends TabPFN tokens), so it runs only on
request:    RUN_PARITY=1 .venv/bin/python -m pytest -s tests/test_parity.py
Needs the local MLflow records with a @champion, the data file and TABPFN_TOKEN in .env.

Both sides use the same TabPFN model: their predictor leaves it to the server's default, which drifts (v3 when the brief
was written, v3.5 today), so the test pins their predictor to the model the served version uses. It compares logic, not
model versions. Their predictor otherwise runs exactly as given, with the full names database. Our side gets a long
TabPFN deadline: the 2 s deadline and its fallback are a serving policy, measured by the load test, not logic.

What must match: the brands shown and their order, exactly; that is what the visitor sees. The expected payouts behind
the order may differ a little: we call TabPFN with its server-side cache, which rounds differently from a fresh
fit (first measured at $0.31 on one visitor, then up to $0.57 across ten). So payouts only get a sanity bound, 5% or $1,
which still catches real bugs (wrong columns, wrong model, wrong files) that would move them by far more.
"""
import math
import os

import pytest

from pipeline import settings
from pipeline.visitors import newest_week_visitors

pytestmark = pytest.mark.skipif(os.environ.get("RUN_PARITY") != "1", reason="slow and online: set RUN_PARITY=1")

VISITORS = 10


def test_served_model_ranks_like_their_predictor():
    settings.load_env()
    from functools import partial

    import mlflow
    from tabpfn_client import TabPFNRegressor

    import bl_exp_payout_predictor as theirs  # their import loads the full names database

    settings.setup_mlflow()
    uri = f"models:/{settings.MODEL_NAME}@{settings.CHAMPION}"
    full_names = theirs.nd
    os.environ["PAYOUT_DEADLINE_S"] = "60"  # read when the version loads: wait for TabPFN, never fall back here
    served = mlflow.pyfunc.load_model(uri)  # our loader gives their module the small first-name table
    del os.environ["PAYOUT_DEADLINE_S"]
    ours, small_names = served.unwrap_python_model(), theirs.nd
    files = mlflow.artifacts.download_artifacts(f"{uri}") + "/artifacts/files/"  # the very files this version serves
    log_dir = str(settings.ROOT / "outputs" / "parity_logs") + "/"
    given_tabpfn = theirs.TabPFNRegressor
    theirs.TabPFNRegressor = partial(TabPFNRegressor, model_path=served.model_config["tabpfn_model"])

    mismatches = []
    try:
        for i, visitor in enumerate(newest_week_visitors(VISITORS), start=1):
            theirs.user_data, theirs.nd = visitor, full_names  # as their __main__ does, with their own names database
            expected = theirs.BLPayoutModelsPredict(files, visitor, log_dir).predict_()  # as given: fit + predict
            theirs.nd = small_names
            got, source = ours.rank(dict(visitor))
            same_brands = set(expected) == set(got)
            same_ranks = same_brands and all(expected[b]["rank"] == got[b]["rank"] for b in expected
                                             if isinstance(expected[b], dict))
            pairs = [(expected[b]["expected_payout"], got[b]["expected_payout"]) for b in expected
                     if same_brands and isinstance(expected[b], dict)]
            gap = max((abs(a - b) for a, b in pairs), default=0.0) if same_brands else math.inf
            relative = max((abs(a - b) / a for a, b in pairs if a > 0), default=0.0)
            within = all(abs(a - b) <= max(1.0, 0.05 * a) for a, b in pairs)
            print(f"visitor {i:>2}: {len(expected)} brands, ranks {'same' if same_ranks else 'DIFFERENT'}, "
                  f"largest payout gap ${gap:.3f} ({relative:.1%}), our payout source {source}")
            if not (same_ranks and within):
                mismatches.append((i, expected, got))
    finally:
        theirs.TabPFNRegressor, theirs.nd = given_tabpfn, full_names
    assert not mismatches, mismatches

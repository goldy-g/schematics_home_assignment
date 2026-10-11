"""Parity: our served model must rank exactly like their predictor, run as given, on the same model files.

Slow and online (their predictor re-fits TabPFN on every call; every call spends TabPFN tokens), so it runs only on
request:    RUN_PARITY=1 .venv/bin/python -m pytest -s tests/test_parity.py
Needs the local MLflow records with a @champion, the data file and TABPFN_TOKEN in .env.

What must match: the brands shown and their order, exactly; that is what the visitor sees. The expected payouts behind
the order may differ a little: we call TabPFN with its server-side cache (task 1.8), which rounds differently from a fresh
fit (first measured at $0.31 on one visitor, then up to $0.57 across ten). So payouts only get a sanity bound, 5% or $1,
which still catches real bugs (wrong columns, wrong model, wrong files) that would move them by far more.
"""
import math
import os

import pandas as pd
import pytest

from pipeline import settings

pytestmark = pytest.mark.skipif(os.environ.get("RUN_PARITY") != "1", reason="slow and online: set RUN_PARITY=1")

VISITORS = 10
FIELDS = list(settings.EXAMPLE_VISITOR)
SURVEY = ["credit_score", "industry", "loan_amount", "loan_reason", "monthly_revenue", "time_in_business", "device_type",
          "business_type"]


def newest_week_visitors(count):
    """Real registered visitors from the newest week, as request dicts (they stay in memory, never written)."""
    data = pd.read_csv(settings.DATA_FILE, low_memory=False)
    data = data[data["register_date"].notna() & (data[SURVEY].notna().sum(axis=1) >= 5)]
    data["session_dt"] = pd.to_datetime(data["session_dt"])
    week = data[data["session_dt"] > data["session_dt"].max() - pd.Timedelta(days=6)]
    rows = week.sample(count, random_state=0)
    rows["session_dt"] = rows["session_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    rows["cellphone"] = rows["cellphone"].astype("int64")
    # a missing answer arrives as JSON null, i.e. None (their code expects text or None, not a float NaN)
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
            for row in rows[FIELDS].to_dict("records")]


def test_served_model_ranks_like_their_predictor():
    settings.load_env()
    import mlflow
    import bl_exp_payout_predictor as theirs

    settings.setup_mlflow()
    uri = f"models:/{settings.MODEL_NAME}@{settings.CHAMPION}"
    ours = mlflow.pyfunc.load_model(uri).unwrap_python_model()
    files = mlflow.artifacts.download_artifacts(f"{uri}") + "/artifacts/files/"  # the very files this version serves
    log_dir = str(settings.ROOT / "outputs" / "parity_logs") + "/"

    mismatches = []
    for i, visitor in enumerate(newest_week_visitors(VISITORS), start=1):
        theirs.user_data = visitor  # how their __main__ passes the visitor
        expected = theirs.BLPayoutModelsPredict(files, visitor, log_dir).predict_()  # as given: fit + predict
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
    assert not mismatches, mismatches

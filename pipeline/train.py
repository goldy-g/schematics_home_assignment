"""Run their training in one mode, recorded as one MLflow run.

    python -m pipeline.train --mode test         train on older data, score the newest week, log the accuracy
    python -m pipeline.train --mode production   train, register a new model version, check it, make it the champion

Their code (bl_models_train.py) runs unchanged. This file records what happens around it.
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from functools import partial
from importlib.metadata import version
from platform import python_version

from pipeline import settings

settings.load_env()  # before importing mlflow: the TabPFN token, and MLflow's usage statistics off

import joblib  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from tabpfn_client import TabPFNRegressor  # noqa: E402

import bl_models_train as theirs  # noqa: E402  (their file; importing it loads the names database, about 3 s)
from pipeline import promote, researcher_log  # noqa: E402

# Pin TabPFN's model inside their trainer: the same class with one argument preset; their file stays unchanged.
# Without it, Prior Labs' default ("auto") answers, and that can change between this run and serving.
theirs.TabPFNRegressor = partial(TabPFNRegressor, model_path=settings.TABPFN_MODEL)

PACKAGES = ("pandas", "numpy", "scikit-learn", "joblib", "catboost", "names-dataset", "tabpfn-client", "mlflow")
TABPFN_SETTINGS = ("model_path", "n_estimators", "fit_mode", "random_state", "ignore_pretraining_limits")


class ObservedFit(theirs.BLPayoutModelsFit):
    """Their class, plus notes taken around three of its steps. Each override calls theirs and changes nothing."""

    def __init__(self, *args, **kwargs):
        self.facts = {}
        super().__init__(*args, **kwargs)

    def split_by_time(self, bl_data, days_for_test):
        train, test = super().split_by_time(bl_data, days_for_test)
        self.facts.update(rows_train=len(train), rows_test=len(test), days_for_test=days_for_test,
                          train_until=str(train["session_dt"].max()), test_from=str(test["session_dt"].min()))
        return train, test

    def catbosot_model_sold_to_client(self, x_train, y_train):
        start = time.perf_counter()
        model = super().catbosot_model_sold_to_client(x_train, y_train)
        self.facts["seconds_catboost_fit"] = time.perf_counter() - start
        self.facts["n_features"] = x_train.shape[1]
        self.facts["catboost"] = model.get_params()
        return model

    def tabpfn_regression_payout(self, x_train_payout, y_train_payout):
        start = time.perf_counter()
        model, context = super().tabpfn_regression_payout(x_train_payout, y_train_payout)
        self.facts["seconds_tabpfn_fit"] = time.perf_counter() - start
        self.facts["tabpfn_context_rows"] = len(context["x"])
        given = model.get_params()
        self.facts["tabpfn"] = {k: "server default" if given.get(k) is None else given[k] for k in TABPFN_SETTINGS}
        return model, context


def data_facts(path):
    """Which data this run used: a fingerprint of the file and the sessions' date range."""
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    dates = pd.to_datetime(pd.read_csv(path, usecols=["session_dt"])["session_dt"])
    return {"data_file": path.name, "data_sha256": sha256[:16],
            "data_first_session": str(dates.min()), "data_last_session": str(dates.max())}


def log_accuracy(text):
    """Test mode: the researcher's accuracy numbers as MLflow metrics. Per-day numbers use step = test day (1 to 7)."""
    reports = researcher_log.catboost_reports(text)
    if reports:
        mlflow.log_metrics({f"week_{name}": value for name, value in reports[0].items()})
        for day, report in enumerate(reports[1:], start=1):
            for name, value in report.items():
                mlflow.log_metric(f"day_{name}", value, step=day)
    overall, by_threshold = researcher_log.payout_errors(text)
    if overall:
        mlflow.log_metrics({"payout_mape": overall[0], "payout_mae": overall[1]})
        for threshold, (mape, mae) in by_threshold.items():
            mlflow.log_metric("payout_mape_at_threshold", mape, step=threshold)
            mlflow.log_metric("payout_mae_at_threshold", mae, step=threshold)


def save_tabpfn_fit(files):
    """Fit TabPFN once, here, with the server-side cache, and save the fit's id: servers then start without fitting."""
    context = joblib.load(files / "payout_tfm_context.joblib")
    tabpfn = TabPFNRegressor(ignore_pretraining_limits=True, model_path=settings.TABPFN_MODEL,
                             fit_mode=settings.TABPFN_FIT_MODE)
    tabpfn.fit(context["x"], context["y"])
    tabpfn.save_model(files / settings.TABPFN_FIT_FILE)


def save_first_name_gender(path):
    """Task 1.12: write the part of the names database their gender feature reads (FirstNameGender in ranker.py reads it).
    Built from the full database their trainer already loaded, with the same keys and the same shares."""
    names = {"M": "Male", "F": "Female"}
    table = {name: {names[code]: share for code, share in info["gender"].items()}
             for name, info in theirs.nd.first_names.items() if info["gender"]}
    with gzip.open(path, "wt") as f:
        json.dump(table, f, separators=(",", ":"))


def register(out_dir):
    """Package the three files, the saved TabPFN fit, their predictor and our wrapper as one MLflow model,
    registered as a new version."""
    files = out_dir / "files"
    files.mkdir()
    for name in settings.MODEL_FILES:
        shutil.copy2(out_dir / name, files / name)
    save_tabpfn_fit(files)
    save_first_name_gender(files / settings.FIRST_NAMES_FILE)
    example = pd.DataFrame([settings.EXAMPLE_VISITOR])
    info = mlflow.pyfunc.log_model(
        name="ranker",
        python_model=str(settings.ROOT / "pipeline" / "ranker.py"),     # models from code: the file is stored, not pickled
        artifacts={"files": str(files)},                                 # the three files are copied into the package
        code_paths=[str(settings.ROOT / "bl_exp_payout_predictor.py")],  # their predictor travels with every version
        signature=infer_signature(example),                              # the 22 request fields and their types
        input_example=example,
        model_config={"tabpfn_model": settings.TABPFN_MODEL, "tabpfn_fit_mode": settings.TABPFN_FIT_MODE,
                      "tabpfn_fit_file": settings.TABPFN_FIT_FILE, "payout_deadline_s": settings.PAYOUT_DEADLINE_S,
                      "first_names_file": settings.FIRST_NAMES_FILE},
        pip_requirements=[f"{p}=={version(p)}" for p in PACKAGES],
        registered_model_name=settings.MODEL_NAME,
    )
    return info.registered_model_version


def smoke_check(model_version):
    """Load the new version the way a server would, rank the made-up visitor, and check the answer's shape."""
    model = mlflow.pyfunc.load_model(f"models:/{settings.MODEL_NAME}/{model_version}")
    answer = model.predict(pd.DataFrame([settings.EXAMPLE_VISITOR]))
    rows = [v for v in answer.values() if isinstance(v, dict)] if isinstance(answer, dict) else []
    ranks = sorted(v["rank"] for v in rows)
    ok = bool(rows) and ranks == list(range(1, len(rows) + 1)) and all(v["expected_payout"] > 0.01 for v in rows)
    return ok, answer


def run(mode, evaluation_run_id=None):
    """Train in one mode as one MLflow run, and return its id. Production also registers, checks and promotes."""
    production = mode == "production"
    settings.setup_mlflow(settings.PROD_EXPERIMENT if production else settings.EVAL_EXPERIMENT)
    with mlflow.start_run(run_name=f"{mode}-{datetime.now():%Y%m%d-%H%M}") as run:
        mlflow.set_tag("trigger", os.environ.get("BL_TRIGGER", "manual"))  # "scheduler" when the scheduler started it
        if evaluation_run_id:
            mlflow.set_tag("evaluation_run_id", evaluation_run_id)  # the test run that scored this same data and code
        out_dir = settings.RUNS_DIR / run.info.run_id
        mlflow.log_params({"mode": mode, "train_test": not production, **data_facts(settings.DATA_FILE),
                           "python": python_version(), **{f"pkg_{p}": version(p) for p in PACKAGES}})

        fit = ObservedFit(f"{settings.DATA_FILE.parent}/", settings.DATA_FILE.name, f"{out_dir}/",
                          train_test=not production)
        start = time.perf_counter()
        fit.fit_()  # their whole pipeline, unchanged
        mlflow.log_metric("seconds_total", time.perf_counter() - start)
        mlflow.log_metrics({k: fit.facts.pop(k) for k in ("seconds_catboost_fit", "seconds_tabpfn_fit")})
        mlflow.log_params({f"catboost_{k}": v for k, v in fit.facts.pop("catboost").items()})
        mlflow.log_params({f"tabpfn_{k}": v for k, v in fit.facts.pop("tabpfn").items()})
        mlflow.log_params(fit.facts)  # row counts of the split, its boundary, the number of features, the context size

        log_file = max(out_dir.glob("log_file_bl_train_*.log"))  # the researcher's log, as their code wrote it
        mlflow.log_artifact(str(log_file), artifact_path="researcher_log")
        text = log_file.read_text()
        mlflow.log_params(researcher_log.row_counts(text))
        if not production:
            log_accuracy(text)
            print(f"test run {run.info.run_id}: accuracy logged")
            return run.info.run_id

        model_version = register(out_dir)
        ok, answer = smoke_check(model_version)
        client = mlflow.MlflowClient()
        client.set_model_version_tag(settings.MODEL_NAME, model_version, "smoke_check", "passed" if ok else "failed")
        if not ok:  # the run is marked FAILED, and @champion stays on the version that works
            sys.exit(f"version {model_version} failed the smoke check, so @champion stays where it is. Answer: {answer}")
        promote.promote(client, model_version)
        print(f"production run {run.info.run_id}: {settings.MODEL_NAME} v{model_version} is now @{settings.CHAMPION}")
        return run.info.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["test", "production"], required=True)
    run(parser.parse_args().mode)


if __name__ == "__main__":
    main()

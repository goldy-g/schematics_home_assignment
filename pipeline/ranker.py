"""The served model: their feature and scoring code with the models loaded once, ranking the brands for one visitor.

MLflow stores this file inside every registered version ("models from code"), together with their predictor file, the
three model files and the saved TabPFN fit. A version is self-contained: the files, the code that reads them, the
package list and its settings (model_config).
"""
import gzip
import json
import os
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import mlflow
import pandas as pd


class PayoutWithDeadline:
    """Takes TabPFN's place inside their scoring function: the same predict(rows), but it never waits past the deadline.
    If TabPFN is late or fails (network, quota), each row gets its brand's median payout from the context instead."""

    def __init__(self, tabpfn, context, deadline_s):
        self.tabpfn, self.deadline_s = tabpfn, deadline_s
        paid = pd.DataFrame({"brand": context["x"]["client_name"].to_numpy(), "payout": context["y"].to_numpy()})
        self.median_by_brand = paid.groupby("brand")["payout"].median()
        self.median_all = float(paid["payout"].median())  # for brands the context never saw
        self.pool = ThreadPoolExecutor(max_workers=8)  # TabPFN calls run here, so a late one can be left behind
        self.source = threading.local()  # per request: "tabpfn" or "fallback"

    def predict(self, rows):
        call = self.pool.submit(self.tabpfn.predict, rows)
        try:
            payout = call.result(timeout=self.deadline_s)
            self.source.value = "tabpfn"
            return payout
        except Exception:  # too late, quota (HTTP 429) or network: rank with the fallback rather than fail the page
            self.source.value = "fallback"
            return rows["client_name"].map(self.median_by_brand).fillna(self.median_all).to_numpy()


class FirstNameGender:
    """Task 1.12: the names database, cut to the one thing their code reads from it.

    Their predictor runs `nd = NameDataset()` when imported (names-dataset: 2.1 GB of memory, 2.7 s) and uses it only in
    detect_gender_with_confidence(), which reads search(first_name)["first_name"]["gender"]. This keeps exactly that:
    every first name that has a gender, with the same shares, behind the same search(). Training writes the table
    from the full database; scripts/names_table_check.py proves the answers are identical."""

    def __init__(self, path):
        with gzip.open(path, "rt") as f:
            self.table = json.load(f)

    def search(self, name):
        gender = self.table.get(name.strip().title())  # the same key normalisation as names_dataset
        return {"first_name": None if gender is None else {"gender": dict(gender)}, "last_name": None}

    def import_predictor(self):
        """Import their predictor with this table standing in for the names_dataset package, then put it back."""
        saved = sys.modules.get("names_dataset")
        stand_in = types.ModuleType("names_dataset")
        stand_in.NameDataset = lambda *args, **kwargs: self
        sys.modules["names_dataset"] = stand_in
        try:
            import bl_exp_payout_predictor as theirs
        finally:
            if saved is None:
                sys.modules.pop("names_dataset", None)
            else:
                sys.modules["names_dataset"] = saved
        theirs.nd = self  # in case this process imported it earlier with the full database
        return theirs


class BrandRanker(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Runs ONCE, when a server (or the smoke check) loads this version.
        from catboost import CatBoostClassifier

        config = context.model_config
        files = Path(context.artifacts["files"])
        if "first_names_file" in config:  # versions from 1.12 on carry the small table
            theirs = FirstNameGender(files / config["first_names_file"]).import_predictor()
        else:  # older versions: their import, which loads the full names database
            import bl_exp_payout_predictor as theirs
        log_dir = os.environ.get("BL_LOG_DIR", str(Path.cwd() / "outputs" / "serving_logs"))
        self.theirs = theirs
        self.predictor = theirs.BLPayoutModelsPredict(f"{files}/", None, log_dir)  # their constructor: one log file per load
        # Our loader replaces their load_models(), which re-fitted TabPFN on every request. Same files, same models.
        tfm_context = joblib.load(files / "payout_tfm_context.joblib")
        self.columns = tfm_context["columns"]
        self.catboost = CatBoostClassifier(allow_writing_files=False).load_model(str(files / "CB_bl_lead.cbm"), format="cbm")
        tabpfn = self._tabpfn(files, config, tfm_context)
        deadline_s = float(os.environ.get("PAYOUT_DEADLINE_S", config["payout_deadline_s"]))
        self.payout = PayoutWithDeadline(tabpfn, tfm_context, deadline_s)
        self.lock = threading.Lock()

    @staticmethod
    def _tabpfn(files, config, tfm_context):
        """The TabPFN fit saved during training (no upload, no fit here); fit again only if Prior Labs no longer has it."""
        from tabpfn_client import FittedModelNotFoundError, TabPFNRegressor, set_access_token

        set_access_token(os.environ["TABPFN_TOKEN"])
        fit_file = files / config["tabpfn_fit_file"]
        if fit_file.exists():
            tabpfn = TabPFNRegressor.load_model(fit_file)
            try:
                tabpfn.predict(tfm_context["x"].iloc[:1])  # warm-up: opens the connection and proves the fit still exists
                return tabpfn
            except FittedModelNotFoundError:
                pass
        tabpfn = TabPFNRegressor(ignore_pretraining_limits=True, model_path=config["tabpfn_model"],
                                 fit_mode=config["tabpfn_fit_mode"])
        tabpfn.fit(tfm_context["x"], tfm_context["y"])
        return tabpfn

    def rank(self, visitor):
        """One visitor's answers in; their ranked dict and the payout source ("tabpfn" or "fallback") out."""
        with self.lock:  # their feature code reads the visitor from a module-level variable (their L58): one at a time
            self.theirs.user_data = visitor
            rows = self.predictor.bl_preprocessing()
        if rows.shape[0] == 0:
            return {"expected_payout": 0, "prob_lead": 0}, "none"  # their answer for a too-incomplete survey (L239)
        ranking = self.predictor.prediction_expected_payout(rows, self.payout, self.columns, self.catboost)
        return ranking, self.payout.source.value

    def predict(self, context, model_input: pd.DataFrame, params=None) -> dict:
        # MLflow's entry point, called for every request: one visitor per call.
        return self.rank(model_input.iloc[0].to_dict())[0]


mlflow.models.set_model(BrandRanker())

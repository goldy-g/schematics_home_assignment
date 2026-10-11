"""The served model: their predictor, loaded once, ranking the brands for one visitor per call.

MLflow stores this file inside every registered version ("models from code"), together with their predictor file and
the three model files. A version is therefore self-contained: the files, the code that reads them, the package list.
"""
import os
import threading
from pathlib import Path

import mlflow
import pandas as pd


class BrandRanker(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Runs ONCE, when a server (or the smoke check) loads this version.
        import bl_exp_payout_predictor as theirs  # their file, unchanged; importing it loads the names database
        self.theirs = theirs
        files = context.artifacts["files"] + "/"  # their code joins folder and file names with +
        log_dir = os.environ.get("BL_LOG_DIR", str(Path.cwd() / "outputs" / "serving_logs"))
        self.predictor = theirs.BLPayoutModelsPredict(files, None, log_dir)  # their constructor: one log file per load
        # Their load_models(): read the TabPFN context and fit TabPFN on it, load the CatBoost file. Once, not per request.
        self.tabpfn, self.columns, self.catboost = self.predictor.load_models()
        self.lock = threading.Lock()

    def predict(self, context, model_input: pd.DataFrame, params=None) -> dict:
        # Runs on EVERY request: their feature code and their scoring, unchanged.
        visitor = model_input.iloc[0].to_dict()
        with self.lock:  # their feature code reads the visitor from a module-level variable (their L58): one at a time
            self.theirs.user_data = visitor
            rows = self.predictor.bl_preprocessing()
        if rows.shape[0] == 0:
            return {"expected_payout": 0, "prob_lead": 0}  # their own answer when the survey is too incomplete (L239)
        return self.predictor.prediction_expected_payout(rows, self.tabpfn, self.columns, self.catboost)


mlflow.models.set_model(BrandRanker())

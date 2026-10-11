"""Names, paths and environment for the whole pipeline. One name per concept."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # the repo; their two scripts sit here
DATA_FILE = ROOT / "data" / "bl_full_data.csv"       # git-ignored copy of their CSV
RUNS_DIR = ROOT / "outputs" / "runs"                 # where their trainer writes its files during a run (git-ignored)

# MLflow: a database for the records, a folder for the files (both git-ignored)
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
FILES_ROOT = ROOT / "mlruns"
EVAL_EXPERIMENT = "bl_ranking_evaluation"            # test-mode runs: accuracy on the newest week, no model
PROD_EXPERIMENT = "bl_ranking_production"            # production runs: each one registers a new model version
MODEL_NAME = "bl_brand_ranker"                       # registered model: what the endpoint serves
CHAMPION = "champion"                                # alias: the version production uses
PREVIOUS = "previous"                                # alias: the version before it, the rollback target

# The three files their trainer writes in production mode, which the predictor reads
MODEL_FILES = ("CB_bl_lead.cbm", "payout_tfm_context.joblib", "all_clients.csv")

# A made-up visitor in the request format (their predictor's 22 fields); no real person. MLflow stores it as the
# model's input example, and the smoke check scores it. Types follow the CSV: sub1-sub3 as text, ids and phone as numbers.
EXAMPLE_VISITOR = {
    "session_dt": "2026-02-10 14:02:11", "conversion_dt": "2026-02-10 14:04:40", "register_date": "2026-02-10 14:04:38",
    "campaign_id": 120236009619230306, "page": "top10us.com/app/business-loans-v6",
    "auto_city": "Houston", "auto_country": "United States", "auto_state": "Texas", "device_type": "mobile",
    "sub1": "01122005", "sub2": "01122005 Ad set", "sub3": "151644740",
    "business_type": "LLC", "credit_score": "Good (650-719)", "industry": "construction",
    "loan_amount": "$50,000 - $74,999", "loan_reason": "Equipment purchase",
    "monthly_revenue": "$20,000 - $49,999", "time_in_business": "2+ years",
    "fname": "Alex", "lname": "Morgan", "cellphone": 7135550123,
}


def load_env():
    """Copy KEY=VALUE lines from .env into the environment (the TabPFN token lives there, never in code).
    Call it before importing mlflow, so MLflow's usage statistics are switched off from the start."""
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def setup_mlflow(experiment=None):
    """Point MLflow at our records. With an experiment name: create it if missing, then record new runs in it."""
    import mlflow
    mlflow.set_tracking_uri(TRACKING_URI)
    if experiment is not None:
        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(experiment, artifact_location=(FILES_ROOT / experiment).as_uri())
        mlflow.set_experiment(experiment)

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

# The weekly job (briefing: "Schedule the production run weekly, Sunday 05:00"), read in Israel time.
# Cron fields: minute 0, hour 5, any day of the month, any month, day of week 0 = Sunday.
WEEKLY_CRON = os.environ.get("BL_WEEKLY_CRON", "0 5 * * 0")  # override only to test the scheduler
TIMEZONE = "Asia/Jerusalem"

# TabPFN (task 1.8, measured in scripts/tabpfn_options.py). Their code leaves the model to the server's default, "auto",
# which is TabPFN-3.5 today and can change without notice. We pin it, fit once with the server-side cache, and give
# every request a deadline: past it, the ranking falls back to CatBoost's P(lead) x the brand's median payout.
TABPFN_MODEL = "v3.5_default"
TABPFN_FIT_MODE = "fit_with_cache"
TABPFN_FIT_FILE = "tabpfn_fit.json"   # the saved fit, stored inside each model version
PAYOUT_DEADLINE_S = float(os.environ.get("PAYOUT_DEADLINE_S", "2.0"))

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
    """Point MLflow at our records. With an experiment name: create it if missing, then record new runs in it.
    Locally the files go to the repo's mlruns/ folder; behind an MLflow server (docker compose) the server keeps them."""
    import mlflow
    mlflow.set_tracking_uri(TRACKING_URI)
    if experiment is not None:
        if mlflow.get_experiment_by_name(experiment) is None:
            local = not TRACKING_URI.startswith("http")
            mlflow.create_experiment(experiment, artifact_location=(FILES_ROOT / experiment).as_uri() if local else None,
                                     tags={"mlflow.experimentKind": "custom_model_development"})  # UI opens on "Model training"
        mlflow.set_experiment(experiment)

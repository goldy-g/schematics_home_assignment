"""The ranking endpoint: one visitor's survey answers in, their ranked brand dictionary out.

    uvicorn serving.app:app --port 8000

    POST /rank      the visitor (their predictor's 22 fields)  ->  {brand: {"rank": 1.0, "expected_payout": 42.31}, ...}
    GET  /health    200 once a model is loaded, 503 before that
    GET  /version   which registered version is being served
    POST /reload    load the version @champion points at now (the server also checks every minute)

The body keeps their exact format. Two headers say how the answer was made: X-Model-Version, and X-Payout-Source
("tabpfn", or "fallback" when TabPFN missed its deadline).
"""
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from pipeline import settings

settings.load_env()
os.environ.setdefault("BL_LOG_DIR", str(settings.ROOT / "outputs" / "serving_logs"))  # their predictor's log file

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

CHAMPION_CHECK_S = 60
state = {"model": None, "ranker": None, "version": None, "run_id": None, "loaded_at": None}
reload_lock = threading.Lock()


class Visitor(BaseModel):
    """Their predictor's 22 fields. Numbers are accepted where their own example sends numbers (campaign_id, sub1,
    sub3, cellphone); their code turns sub1-sub3 into text. Note: in the data sub1 is text with a leading zero
    ("01121993"), so sending it as a number gives a value training never saw (quirk P7)."""
    session_dt: str
    conversion_dt: str | None = None
    register_date: str | None = None
    campaign_id: int | str
    page: str
    auto_city: str | None = None
    auto_country: str | None = None
    auto_state: str | None = None
    device_type: str | None = None
    sub1: str | int
    sub2: str | int
    sub3: str | int
    business_type: str | None = None
    credit_score: str | None = None
    industry: str | None = None
    loan_amount: str | None = None
    loan_reason: str | None = None
    monthly_revenue: str | None = None
    time_in_business: str | None = None
    fname: str | None = None
    lname: str | None = None
    cellphone: int | str


def champion_version():
    return str(mlflow.MlflowClient().get_model_version_by_alias(settings.MODEL_NAME, settings.CHAMPION).version)


def load(version):
    """Load one registered version, then swap it in. Requests keep using the old one until the new one is ready."""
    model = mlflow.pyfunc.load_model(f"models:/{settings.MODEL_NAME}/{version}")
    state.update(model=model, ranker=model.unwrap_python_model(), version=version, run_id=model.metadata.run_id,
                 loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    print(f"serving {settings.MODEL_NAME} v{version}", flush=True)


def reload_if_moved():
    with reload_lock:
        version = champion_version()
        if version != state["version"]:
            load(version)
        return version


def watch_champion():
    """Load @champion now, then check every minute whether it moved (a weekly run, a promotion, a rollback)."""
    while True:
        try:
            reload_if_moved()
        except Exception as error:  # MLflow unreachable or no champion yet: keep serving what is loaded, retry later
            print(f"champion check failed: {error}", flush=True)
        time.sleep(CHAMPION_CHECK_S)


def answer(visitor):
    """Versions from v3 on have rank(), which also says where the payout came from. Older versions (a rollback can
    bring them back) only have MLflow's standard predict(): use it, with the types their stored signature expects."""
    if hasattr(state["ranker"], "rank"):
        return state["ranker"].rank(visitor)
    row = dict(visitor, campaign_id=int(visitor["campaign_id"]), cellphone=int(visitor["cellphone"]),
               sub1=str(visitor["sub1"]), sub2=str(visitor["sub2"]), sub3=str(visitor["sub3"]))
    return state["model"].predict(pd.DataFrame([row])), "unknown"


@asynccontextmanager
async def lifespan(app):
    settings.setup_mlflow()
    threading.Thread(target=watch_champion, daemon=True).start()
    yield


app = FastAPI(title="Business Loans brand ranking", lifespan=lifespan)


@app.post("/rank")
def rank(visitor: Visitor, response: Response):
    ranker = state["ranker"]
    if ranker is None:
        raise HTTPException(503, "the model is still loading")
    if visitor.register_date is None:  # their code raises a bare Exception here (their L66-67): answer it as a 422
        raise HTTPException(422, "user cannot be a lead - register_date is absent")
    ranking, source = answer(visitor.model_dump())
    response.headers["X-Model-Version"] = state["version"]
    response.headers["X-Payout-Source"] = source
    return ranking


@app.get("/health")
def health():
    if state["ranker"] is None:
        raise HTTPException(503, "the model is still loading")
    return {"status": "ok", "version": state["version"]}


@app.get("/version")
def version():
    return {"model": settings.MODEL_NAME, "alias": settings.CHAMPION, "version": state["version"],
            "run_id": state["run_id"], "loaded_at": state["loaded_at"]}


@app.post("/reload")
def reload():
    return {"version": reload_if_moved()}

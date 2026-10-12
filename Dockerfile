# One image for every service: the MLflow server, training, the scheduler and the endpoint (see docker-compose.yml).
FROM python:3.12-slim

# CatBoost needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Their two files, unchanged, and our code. The data and the TabPFN token are never baked in: they arrive at run time.
COPY bl_models_train.py bl_exp_payout_predictor.py ./
COPY pipeline/ pipeline/
COPY serving/ serving/

ENV PYTHONUNBUFFERED=1 MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1

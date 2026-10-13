# Business Loans Brand Ranking: MLOps

The researcher's two scripts, `bl_models_train.py` and `bl_exp_payout_predictor.py`, run here as a production system:
weekly training in both modes, every run recorded in MLflow, versioned models with rollback, a synchronous ranking
endpoint, and a load simulation. **The two original files are unchanged** (byte-identical to the assignment commit
`ae1fce1`); everything else wraps them. The assignment text is in [ASSIGNMENT.md](ASSIGNMENT.md).

## Contents

1. [Quick start with Docker](#1-quick-start-with-docker)
2. [Run without Docker](#2-run-without-docker)
3. [The brief, item by item](#3-the-brief-item-by-item)
4. [How it works](#4-how-it-works)
5. [Results](#5-results)
6. [Reference](#6-reference)
7. [Found in the original code and data](#7-found-in-the-original-code-and-data)
8. [For production](#8-for-production)

---

## 1. Quick start with Docker

### Before you start

| You need | Notes |
|---|---|
| Docker with **at least 6 GB of memory** | Docker Desktop: Settings → Resources. colima: `colima start --memory 8 --cpu 4` |
| The data file `bl_full_data.csv` | From the assignment's Drive link. It holds personal details, so it never enters git, MLflow or the image |
| A Prior Labs TabPFN token | From your Prior Labs account |

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/goldy-g/schematics_home_assignment.git && cd schematics_home_assignment
```

With colima on a Mac, clone under your home folder: colima shares only that folder with Docker.

**2. Add the data**

```bash
mkdir -p data && cp /path/to/bl_full_data.csv data/
```

**3. Add the token**

```bash
echo 'TABPFN_TOKEN=<your token>' > .env
```

**4. Start everything**

```bash
docker compose up --build -d
```

(`docker-compose up --build -d` with the older standalone tool.) The first build downloads the Python image and the
packages, about 4 minutes. Four services start:

| Service | What it does | Where |
|---|---|---|
| `mlflow` | Tracking server and model registry | http://localhost:5001 |
| `train` | Runs the weekly job once at start-up, so there is a model to serve, then exits | `docker compose logs train` |
| `scheduler` | Runs the weekly job every Sunday 05:00, Israel time | `docker compose logs scheduler` |
| `api` | The ranking endpoint | http://localhost:8100 |

**5. Wait for the first model** (about 2 minutes after the build)

```bash
curl localhost:8100/health
```

It answers `503` while the first model trains, then `{"status":"ok","version":"1"}`.

**6. Rank a visitor**

```bash
curl -i -X POST localhost:8100/rank -H 'content-type: application/json' -d @examples/visitor.json
```

`examples/visitor.json` is a made-up visitor in the format of the original predictor's example. You should see:

```text
HTTP/1.1 200 OK
x-model-version: 1
x-payout-source: tabpfn

{"xlt": {"rank": 1.0, "expected_payout": 202.81}, "businessloans.com": {"rank": 2.0, "expected_payout": 145.07},
 "fundera / nerdwallet": {"rank": 3.0, "expected_payout": 19.82}, ...}
```

**7. Look at the records** at http://localhost:5001 (choose **Model training** at the top left if it opens on GenAI)

- **Experiments → `bl_ranking_evaluation`:** test runs with their parameters, accuracy metrics and the researcher's log
  under Artifacts → `researcher_log/`.
- **Experiments → `bl_ranking_production`:** production runs, each linked to the version it registered.
- **Model registry → `bl_brand_ranker`:** the versions, with the aliases `@champion` (served) and `@previous`.

**8. Stop**

```bash
docker compose down
```

The MLflow records stay in a Docker volume; `docker compose down -v` deletes them too.

### If something goes wrong

| What you see | Cause and fix |
|---|---|
| `/health` answers `503` for more than 5 minutes | Training failed: `docker compose logs train` |
| `train` exits with code 137 | Out of memory: give Docker at least 6 GB |
| `train` says the data file is missing | The file is not at `data/bl_full_data.csv`, or (colima) the repository is outside your home folder |
| `env file .env not found`, or `KeyError: 'TABPFN_TOKEN'` | `.env` is missing, or its line is not `TABPFN_TOKEN=...` (step 3) |
| `port is already allocated` | Something else uses port 5001 or 8100; stop it, or change the left side of `ports:` in `docker-compose.yml` |

---

## 2. Run without Docker

Needs Python 3.12 or newer, plus the data file and `.env` from steps 2 and 3 above.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m pipeline.weekly
```

Runs the weekly job once: a test run, then a production run that registers a version and promotes it.

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050
```

The records, at http://localhost:5050 (port 5050, because macOS keeps 5000 for AirPlay).

```bash
.venv/bin/uvicorn serving.app:app --port 8100
```

The endpoint, used exactly as in step 6 above. Local records live in `mlflow.db` and `mlruns/`, both git-ignored.

---

## 3. The brief, item by item

| The brief asks | Where it is done |
|---|---|
| Productize with MLflow, runnable locally, Docker allowed; do not change functions and logic | The two original files are unchanged; [Quick start](#1-quick-start-with-docker) |
| Both modes: train/test and production (registers artifacts) | `pipeline.train --mode test` / `--mode production`; [Training](#41-training) |
| Schedule the production run weekly, Sunday 05:00 | `pipeline/scheduler.py`, Israel time; [Training](#41-training) |
| Log technical parameters so runs are comparable | Parameters and metrics on every run; [Training](#41-training) |
| Save the researcher's log as an artifact | `researcher_log/` on every run; [Training](#41-training) |
| Version artifacts; identify and roll back a serving version | Registry versions, `@champion` / `@previous`, header `X-Model-Version`; [Versions](#42-versions-and-rollback) |
| Productize TabPFN in the most efficient way | Fit once a week, saved in the version, pinned to v3, 2 s deadline; [TabPFN](#44-tabpfn-made-efficient) |
| Endpoint: a user's dict in, the ranked brand dict out, synchronous | `POST /rank`; [Serving](#43-serving) |
| Simulate production traffic; median and tail latency; minimize latency | `scripts/load_test.py`; [Results](#5-results) |
| Runnable code in main | This branch |

One difference from the brief's description is kept on purpose: the original production mode does not train on all
data (it skips the newest 6 days, like test mode). Changing that would change the researcher's logic, so it is kept and
explained in [section 7](#7-found-in-the-original-code-and-data).

---

## 4. How it works

```text
WEEKLY (Sunday 05:00, Israel)                    MLflow                              PER VISITOR
scheduler ──► weekly job                    ──►  runs: parameters, metrics, log  ──►  POST /rank
                1. test run: train, grade        registry: bl_brand_ranker            loads @champion, swaps it live
                2. production run: train,        v1, v2, … with @champion             CatBoost P(lead) × TabPFN $payout
                   register, smoke-check,        and @previous                        ──► ranked brands
                   promote
```

### 4.1 Training

| Command | What it does |
|---|---|
| `python -m pipeline.train --mode test` | The original trainer with `train_test=True`: trains, grades the newest week, logs the accuracy |
| `python -m pipeline.train --mode production` | `train_test=False`: trains, registers a version, smoke-checks it, promotes it |
| `python -m pipeline.weekly` | Both, in that order; the production run is tagged with the test run's id |
| `python -m pipeline.scheduler` | Runs the weekly job every Sunday 05:00 Asia/Jerusalem (`--next 3` prints the next runs) |

Run the commands from the repository root with the project's Python (`.venv/bin/python` locally; in Docker, prefix
them with `docker compose exec api`). The original trainer runs through a subclass (`ObservedFit` in
`pipeline/train.py`) that only takes notes around three of its steps. Each run records:

| Kind | What |
|---|---|
| Parameters | Data fingerprint (`data_sha256`) and date range, row counts after each filter, the train/test boundary, CatBoost and TabPFN settings, package versions, Python |
| Tags | `trigger` (`scheduler` or `manual`), `evaluation_run_id` on production runs; MLflow adds the git commit for runs from a git checkout (local runs; the Docker image has no git history) |
| Metrics | Step timings; in test mode the researcher's accuracy read from the log: `week_f1_lead`, `week_precision_lead`, …, `day_*` (one point per test day), `payout_mape`, `payout_mae`, and both per threshold |
| Artifact | The researcher's log file, as written, under `researcher_log/` |

Experiments: `bl_ranking_evaluation` (test runs) and `bl_ranking_production` (production runs).

### 4.2 Versions and rollback

Each production run registers a version of `bl_brand_ranker` that is complete on its own: the three model files, the
saved TabPFN fit, the first-name table, the original predictor file, the wrapper (`pipeline/ranker.py`), its settings
and the package list. A version is promoted only if it passes a smoke check: it is loaded the way a server loads it, and
it ranks one visitor.

```bash
python -m pipeline.promote                  # show @champion and @previous
python -m pipeline.promote --rollback       # @champion goes back to @previous
python -m pipeline.promote --version 3      # make any version the champion
```

In Docker, run them inside a container: `docker compose exec api python -m pipeline.promote --rollback`. The endpoint
checks `@champion` every minute and swaps versions without a restart; `POST /reload` does it at once.

### 4.3 Serving

| Route | What it does |
|---|---|
| `POST /rank` | One visitor's answers in; the original predictor's ranked dictionary out |
| `GET /health` | `503` until a model is loaded, then `200` |
| `GET /version` | The served version, its run and when it was loaded |
| `POST /reload` | Check `@champion` now instead of within the minute |

Every answer carries two headers: `X-Model-Version` (which version answered) and `X-Payout-Source` (`tabpfn`, or
`fallback` when TabPFN missed its deadline). A request without `register_date` gets `422`. A visitor with fewer than 5
of 8 survey answers gets the original predictor's own answer, `{"expected_payout": 0, "prob_lead": 0}`.

### 4.4 TabPFN, made efficient

TabPFN runs on Prior Labs' servers. Its `fit` trains nothing: it uploads the 1,000 example rows that the model reads at
every prediction. The original predictor fits on every request. Here the weekly job fits once, with the server-side
cache, and saves the fit inside the version, so a server only predicts. Every call has a **2 s deadline**: if TabPFN is
late, fails or is out of quota, each brand's median payout stands in, and the header says `fallback`.

**The model is pinned to TabPFN v3** (`v3_default`), the version the brief names ("tabpfn-client (v3.0)"; the client
package itself is 0.6.x). The original code never names a model, so the server's default decides, and that default has
already moved to v3.5 without notice: on the same rows the two differ by up to $88 a payout. v3.5 measured more
accurate on the test week, so it is a candidate for the researcher, one setting away (`TABPFN_MODEL=v3.5_default`).

---

## 5. Results

All measured on a laptop over the internet to Prior Labs. The scripts that produced each number are in `scripts/`.

### 5.1 TabPFN: how to call it (`scripts/tabpfn_options.py`)

| How it is called | Per request |
|---|---:|
| As given: `fit` + `predict` on every request | 8.5 s, once 89 s |
| `fit` once, then `predict` | 1.0 s |
| + server-side cache (`fit_mode="fit_with_cache"`) | 0.89 s |
| + the fit saved in the version (`save_model`) | 0.74 s (a repeated visitor) |

| Model | Payout MAPE | Payout MAE | Per call (served, repeated visitor) |
|---|---:|---:|---:|
| v3, pinned | 6.84% | $2.35 | 0.79 s |
| v3.5, today's "auto" | 5.43% | $2.34 | 0.74 s |

### 5.2 Load simulation (`scripts/load_test.py`)

Real registered visitors from the newest week, sent at random (Poisson) arrival times, each request on its own so a
slow answer never holds back the next. The realistic rate comes from the data: at most 31 registrations in one minute,
about 0.5 a second. Measured against the Docker stack, 60 requests per run, so p99 rests on about one request.

| Run | Rate | p50 | p90 | p95 | p99 | max | Errors | Fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Realistic | 0.53/s | 1.36 s | 1.63 s | 1.75 s | 1.79 s | 2.02 s | 0 | 1.7% |
| Stress, ~3× | 1.69/s | 1.58 s | 2.03 s | 2.04 s | 2.04 s | 2.04 s | 0 | 23% |
| Before: the original predictor, one request at a time | — | 8.8–14.9 s | | | | 89 s seen | | |

The deadline caps the tail at about 2 s. The stress run passes Prior Labs' limit of 60 predictions a minute: a quarter
of the rankings used the fallback, and still no request failed or waited longer than 2.04 s. About 0.6 s of every
request is network round trips; a new visitor costs about 0.4 s more for server-side row preparation.

```bash
python scripts/load_test.py --name realistic --rate 0.5 --requests 60 --seed 1
python scripts/load_test.py --name stress    --rate 2   --requests 60 --seed 11
```

Give each run its own `--seed` (Prior Labs reuses rows it has just prepared, so repeated visitors look faster) and
leave a minute between runs, so the rate limit starts empty.

### 5.3 Memory and start-up (`scripts/names_table_check.py`, `scripts/ranker_speed.py`)

The original predictor loads the names-dataset package (2.1 GB) for one feature, the gender of the first name. Each
version now carries a 4.1 MB table of exactly that (714,220 first names) behind the same `search()`. The original
gender function, run both ways on 3,535,486 names: **0 differences**.

| | Before (full database) | After (table) |
|---|---:|---:|
| Version load | 6.6 s | 2.5 s |
| Peak memory per process | 2.27 GB | 0.76 GB |
| Request (repeated visitor, median; both on v3.5) | 0.78 s | 0.74 s |

### 5.4 Correctness

| Claim | Evidence |
|---|---|
| The researcher's logic is unchanged | `tests/test_parity.py`: the original predictor as given vs the endpoint, 10 real visitors, same TabPFN model: identical brands and ranks; payouts within $0.26 (the cache rounds slightly differently) |
| Results are reproducible | Every test run with the same data and model scores the same: F1 0.47; payout MAPE 6.84% with v3 |
| The schedule works | Runs started by the scheduler carry `trigger = scheduler`; a test covers the October clock change |

---

## 6. Reference

### Tests

```bash
.venv/bin/python -m pytest -q
```

Fast and offline: the schedule, the log parser, the HTTP layer, the first-name table.

```bash
RUN_PARITY=1 .venv/bin/python -m pytest -s tests/test_parity.py
```

Slow and online (spends TabPFN tokens): the original predictor against the served model.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `TABPFN_TOKEN` | required, in `.env` | Prior Labs API token |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` (Docker: the `mlflow` service) | Where runs and versions are recorded |
| `TABPFN_MODEL` | `v3_default` | The TabPFN model (`v3.5_default` is the measured candidate) |
| `PAYOUT_DEADLINE_S` | `2.0` | TabPFN deadline per request, in seconds |
| `BL_WEEKLY_CRON` | `0 5 * * 0` | The schedule, in Israel time; override only to test |
| `BL_LOG_DIR` | `outputs/serving_logs` | Where the original predictor writes its log |

### Repository

```text
bl_models_train.py            the original trainer, unchanged
bl_exp_payout_predictor.py    the original predictor, unchanged
pipeline/                     settings, train (both modes), researcher_log, promote, weekly, scheduler,
                              ranker (the served model), visitors
serving/app.py                the endpoint
scripts/                      measurements: run_baseline, tabpfn_options, ranker_speed, names_table_check, load_test
tests/                        automated checks
examples/visitor.json         a made-up visitor for the endpoint
Dockerfile, docker-compose.yml
```

---

## 7. Found in the original code and data

Kept as given and flagged for the researcher:

- **Production mode trains on the same rows as test mode.** The newest 6 days never reach the live model. Part of that
  holdout is useful: brands' verdicts arrive 1–2 days late (the lead share falls from ~33% to 15% and 0% on the two
  newest days), so training on the newest days would teach "new visitors don't convert". The fix to propose is a
  ~2-day holdout, not none.
- **The two models learn different definitions of success.** CatBoost learns a verified "Lead" verdict; TabPFN learns
  from every paid row, and about half of the paid rows have no verified verdict. Brands that pay without reporting back
  are ranked down.
- **Verdicts are copied across a visit's rows** when one visitor went to two brands; the original brand check undoes
  it.
- **TabPFN's model is left to the server's default.** The brief names v3; the default now means v3.5. Pinned to v3 here.
- **TabPFN's 1,000 examples cover about 28 hours and 4 of the 10 brands.**
- **A 2.1 GB package for one feature, and that feature guesses gender from a name:** a fairness question for a lending
  funnel.

---

## 8. For production

- **Quota:** TabPFN-3 charges at least 10,000 tokens a call and allows 60 calls a minute; the default 20M tokens a month
  cover about 2,000 rankings, while the data implies about 30,000. Precomputing is not an option, because the inputs
  include the session's hour and the seconds it took to register. Options: a paid agreement with higher limits, or
  TabPFN in the company's own cloud (Prior Labs offers it through AWS and Azure), which also removes the ~0.6 s of
  internet round trips.
- **Monitoring:** log every request (version, payout source, latency, later the outcome).
- **Databricks:** the pipeline points at a workspace through `MLFLOW_TRACKING_URI=databricks`; the data becomes a Delta
  table; the schedule becomes a Job (`0 0 5 ? * SUN`, Asia/Jerusalem); the endpoint becomes Model Serving. Registering
  in Unity Catalog also needs an output signature on the model (not tried here).

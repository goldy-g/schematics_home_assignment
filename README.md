# Business Loans Brand Ranking, productized

The researcher's two scripts (`bl_models_train.py`, `bl_exp_payout_predictor.py`) run here as a system: weekly
training in both modes, every run recorded in MLflow, versioned models with rollback, a synchronous ranking endpoint,
and a load test. **The two original files are unchanged** (byte-identical to the assignment commit `ae1fce1`); everything
else wraps them. The assignment text is in [ASSIGNMENT.md](ASSIGNMENT.md).

```
Sunday 05:00 (Israel)          MLflow                                   per visitor
scheduler ──► weekly job ──►  runs: params, metrics, the log     ──►  POST /rank ──► ranked brands
              test run         registry: bl_brand_ranker v1, v2…        loads @champion, swaps it live
              production run   aliases: @champion, @previous            CatBoost P(lead) × TabPFN $payout
```

## The brief, item by item

| The brief asks | Where it is done |
|---|---|
| Productize with MLflow, runnable locally, Docker allowed; do not change functions and logic | The two original files unchanged; `docker compose up` runs MLflow, training, the scheduler and the API ([Run it](#run-it)) |
| Both modes: train/test and production (registers artifacts) | `python -m pipeline.train --mode test` / `--mode production` ([Training](#training)) |
| Schedule the production run weekly, Sunday 05:00 | `pipeline/scheduler.py`, Israel time ([Training](#training)) |
| Log technical parameters so runs are comparable | Parameters and metrics on every run ([Training](#training)) |
| Save the researcher's log as an artifact | `researcher_log/` on every run ([Training](#training)) |
| Version artifacts; identify and roll back a serving version | `bl_brand_ranker` versions, `@champion` / `@previous`, header `X-Model-Version` ([Versions and rollback](#versions-and-rollback)) |
| Productize TabPFN in the most efficient way | Fit once per week, saved in the version, pinned to v3 as the brief names, 2 s deadline ([TabPFN](#tabpfn-made-efficient)) |
| Endpoint: a user's dict in, the ranked brand dict out, synchronous | `POST /rank` ([Serving](#serving)) |
| Simulate production traffic; median and tail latency; minimize latency | `scripts/load_test.py` ([Load simulation](#load-simulation), [Minimizing latency](#minimizing-latency-and-memory)) |
| Runnable code in main | This branch |

One difference from the brief's description is kept on purpose: the original production mode does not train on all data (it
skips the newest 6 days, like test mode). Changing it would change the researcher's logic, so it is kept and explained under
[Found in the original code and data](#found-in-the-original-code-and-data-kept-as-given-flagged).

## Run it

**Needs:** Docker (or Python 3.12+ for the local way), the data file, and a Prior Labs TabPFN token.

1. Put the data at `data/bl_full_data.csv` (git-ignored; it holds personal details and never enters git, MLflow or the image).
2. Create `.env` (git-ignored) with one line: `TABPFN_TOKEN=<your token>`.

### With Docker (everything at once)

```bash
docker compose up --build        # or: docker-compose up --build
```

| Service | What it does | Where |
|---|---|---|
| `mlflow` | Tracking server and model registry | http://localhost:5001 |
| `train` | One weekly job at start-up, so there is a champion to serve | exits when done |
| `scheduler` | The weekly job every Sunday 05:00, Israel time | logs its next run |
| `api` | The ranking endpoint | http://localhost:8100 |

The API answers `503` on `/health` until the first champion exists (about 2 minutes after start).

- **Memory:** give Docker at least 6 GB (measured: training peaks at 2.6 GB, the MLflow server uses 1.6 GB, the API
  0.5 GB). Docker Desktop: Settings → Resources. colima: `colima start --memory 8 --cpu 4`.
- **First build:** it downloads the Python image and the packages; about 4 minutes on a laptop.
- **colima on a Mac:** keep the repository under your home folder. colima shares only that folder with its VM, so the
  data mount would otherwise be empty.

### Locally

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pipeline.weekly                   # test run, then production run → a new version, promoted
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050   # the records: http://localhost:5050
.venv/bin/uvicorn serving.app:app --port 8100         # the endpoint
```

Local records live in `mlflow.db` and `mlruns/` (both git-ignored). Port 5050, because macOS keeps 5000 for AirPlay.

## Training

| Command | What it does |
|---|---|
| `python -m pipeline.train --mode test` | The original trainer with `train_test=True`: trains, grades the newest week, logs the accuracy |
| `python -m pipeline.train --mode production` | `train_test=False`: trains, registers a version, smoke-checks it, promotes it |
| `python -m pipeline.weekly` | Both, in that order; the production run is tagged with the test run's id |
| `python -m pipeline.scheduler` | Runs the weekly job every Sunday 05:00 Asia/Jerusalem (`--next 3` prints the next runs) |

The original trainer runs through a subclass (`ObservedFit` in `pipeline/train.py`) that only takes notes around three of its
steps. Each run records in MLflow:

- **Parameters:** data fingerprint (`data_sha256`) and date range, row counts after each filter, the train/test
  boundary, CatBoost and TabPFN settings, package versions, Python. MLflow adds the git commit when it runs from a git checkout
  (local runs; the Docker image carries no git history). Tag `trigger` says
  whether the scheduler or a person started it.
- **Metrics:** step timings; in test mode also the researcher's accuracy, read from the log:
  `week_f1_lead`, `week_precision_lead`, …, `day_*` (one point per test day), `payout_mape`, `payout_mae`, and both per
  threshold.
- **Artifact:** the researcher's log file, as written, under `researcher_log/`.

Experiments: `bl_ranking_evaluation` (test runs) and `bl_ranking_production` (production runs).

## Versions and rollback

Each production run registers a version of `bl_brand_ranker` that is complete on its own: the three model files, the
saved TabPFN fit, the first-name table, the original predictor file, the wrapper (`pipeline/ranker.py`), its settings and the
package list. A version is promoted only if it passes a smoke check (loaded like a server would, one visitor ranked).

```bash
.venv/bin/python -m pipeline.promote                  # show @champion and @previous
.venv/bin/python -m pipeline.promote --rollback       # @champion goes back to @previous
.venv/bin/python -m pipeline.promote --version 3      # make any version the champion
```

The endpoint checks `@champion` every minute and swaps versions without a restart (`POST /reload` to do it at once).

## Serving

```bash
curl -s -X POST localhost:8100/rank -H 'content-type: application/json' -d @examples/visitor.json
```

```json
{"xlt": {"rank": 1.0, "expected_payout": 187.66}, "businessloans.com": {"rank": 2.0, "expected_payout": 137.94}, …}
```

The body is exactly the original predictor's output. Headers `X-Model-Version` and `X-Payout-Source` (`tabpfn` or `fallback`)
say which version answered and where the payouts came from. `GET /health`, `GET /version` and `POST /reload` are for
operations. A request without `register_date` gets `422`; a visitor with fewer than 5 of 8 survey answers gets the
original predictor's own answer, `{"expected_payout": 0, "prob_lead": 0}`.

## TabPFN, made efficient

TabPFN runs on Prior Labs' servers. Its `fit` trains nothing: it uploads the 1,000 example rows the model reads at
every prediction. The original predictor fits on every request. Measured on the same rows (`scripts/tabpfn_options.py`):

| How it is called | Per request |
|---|---:|
| As given: `fit` + `predict` every request | 8.5 s, once 89 s |
| `fit` once, then `predict` | 1.0 s |
| + server-side cache (`fit_mode="fit_with_cache"`) | 0.89 s |
| + the fit saved and reloaded (`save_model`) | 0.74 s (a repeated visitor) |

What the system does: the weekly job fits once with the cache and saves the fit inside the version, so a server only
predicts. Every call has a **2 s deadline**: if TabPFN is late, fails or is out of quota, the brands are ranked with
each brand's median payout instead, and the header says `fallback`.

**The model is pinned to TabPFN v3** (`v3_default`), the version the brief names ("tabpfn-client (v3.0)"; the client
package itself is 0.6.x). The original code never names a model, so the server's default decides, and it has already moved to
v3.5 without notice: on the same rows the two differ by up to $88 a payout. That drift is why the pipeline pins it. v3.5 measured
more accurate on the test week, so it is a candidate for the researcher, one setting away
(`TABPFN_MODEL=v3.5_default`, then the weekly job grades and registers it):

| Model | Payout MAPE | Payout MAE | Per call (served, repeated visitor) |
|---|---:|---:|---:|
| v3, pinned | 6.84% | $2.35 | 0.79 s |
| v3.5, today's "auto" | 5.43% | $2.34 | 0.74 s |

## Proof

| Claim | Evidence |
|---|---|
| The researcher's logic is unchanged | `RUN_PARITY=1 pytest -s tests/test_parity.py`: the original predictor as given vs the endpoint on 10 real visitors, both on the same TabPFN model: identical brands and ranks; payouts within $0.26 (the cache rounds slightly differently) |
| Results are reproducible | Every test run with the same data and model scores the same: F1 0.47; payout MAPE 6.84% with v3 (5.43% with v3.5) |
| The schedule works | Runs tagged `trigger = scheduler`; tests cover the October clock change |

## Load simulation

`scripts/load_test.py` sends real registered visitors from the newest week at random (Poisson) arrival times, each
request on its own, so a slow answer never holds back the next one. The realistic rate comes from the data: at most
31 registered visitors in one minute, about 0.5 a second.

| Run | Rate | p50 | p90 | p95 | p99 | max | errors | fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| realistic | 0.53/s | 1.36 s | 1.63 s | 1.75 s | 1.79 s | 2.02 s | 0 | 1.7% |
| stress | 1.69/s | 1.58 s | 2.03 s | 2.04 s | 2.04 s | 2.04 s | 0 | 23% |

```bash
.venv/bin/python scripts/load_test.py --name realistic --rate 0.5 --requests 60 --seed 1
.venv/bin/python scripts/load_test.py --name stress    --rate 2   --requests 60 --seed 11
```

Measured against the Docker stack (TabPFN v3), 60 requests each, from a laptop over the internet to Prior Labs, so p99
rests on about one request. The deadline caps the tail at about 2 s. The stress run is about 100 requests a minute,
past Prior Labs' limit of 60 predictions a minute: a quarter of the rankings used the fallback, and still no request
took longer than 2.04 s or failed. Two cautions when re-running: give each run its own visitors (`--seed`), because
Prior Labs reuses rows it has just prepared and repeated visitors look faster; and leave a minute between runs, so the
rate limit starts empty. About 0.6 s of every request is network round trips to the hosted API; a new visitor costs
another ~0.4 s of server-side row preparation. Below that needs TabPFN running closer (see below).

## Minimizing latency and memory

The original predictor loads the names-dataset package when imported (2.1 GB) and uses it for one feature, the gender of the
first name. Each version now carries a 4.1 MB table of exactly that (714,220 first names) behind the same
`search()`; `scripts/names_table_check.py` runs the original gender function both ways on 3.5 million names: **0 differences**.

| | Before (full database) | After (table) |
|---|---:|---:|
| Version load | 6.6 s | 2.5 s |
| Peak memory | 2.27 GB | 0.76 GB |
| Request (repeated visitor, median; both on v3.5) | 0.78 s | 0.74 s |

## Tests

```bash
.venv/bin/python -m pytest -q          # schedule, log parser, HTTP layer, first-name table (fast, offline)
RUN_PARITY=1 .venv/bin/python -m pytest -s tests/test_parity.py      # slow, online, spends TabPFN tokens
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `TABPFN_TOKEN` | (required, in `.env`) | Prior Labs API token |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Where runs and versions are recorded |
| `PAYOUT_DEADLINE_S` | `2.0` | TabPFN deadline per request |
| `TABPFN_MODEL` | `v3_default` | The TabPFN model, pinned (`v3.5_default` is the measured challenger) |
| `BL_WEEKLY_CRON` | `0 5 * * 0` | The schedule (Israel time); override only to test |
| `BL_LOG_DIR` | `outputs/serving_logs` | Where the original predictor writes its log |

## Repository

```
bl_models_train.py, bl_exp_payout_predictor.py   the original scripts, unchanged
pipeline/   settings, train (both modes), researcher_log, promote, weekly, scheduler, ranker (the served model), visitors
serving/    app.py, the endpoint
scripts/    measurements: run_baseline, tabpfn_options, ranker_speed, names_table_check, load_test
tests/      automated checks
```

## Found in the original code and data (kept as given, flagged)

- **Production mode trains on the same rows as test mode.** The newest 6 days never reach the live model. Part of
  that holdout is useful: brands' verdicts arrive 1–2 days late (lead share falls from ~33% to 15% and 0% on the two
  newest days), so training on the newest days would teach "new visitors don't convert". The fix to propose is a
  ~2-day holdout, not none.
- **The two models learn different definitions of success.** CatBoost learns a verified "Lead" verdict; TabPFN learns
  from every paid row, and about half of the paid rows have no verified verdict. Brands that pay without reporting
  back are ranked down.
- **Verdicts are copied across a visit's rows** when one visitor went to two brands; the original brand check undoes it.
- **TabPFN's model is left to the server's default.** The brief names v3; "auto" now means v3.5. Pinned to v3 here.
- **TabPFN's 1,000 examples cover about 28 hours and 4 of the 10 brands.**
- **A 2.1 GB package for one feature, and that feature guesses gender from a name:** a fairness question for a
  lending funnel.

## For production (beyond this task)

- **Quota:** TabPFN-3/3.5 charge at least 10,000 tokens per call; the default 20M a month covers ~2,000 rankings, while
  the data implies ~30,000. Precomputing is not an option, because the inputs include the session's hour and the
  seconds it took to register. Options: a paid agreement with higher limits, or TabPFN in the company's own cloud (Prior Labs
  offers it through AWS and Azure), which also removes the ~0.6 s of internet round trips.
- **Log every request** (version, payout source, latency, later the outcome) for monitoring.
- **Databricks:** the pipeline points at a workspace through `MLFLOW_TRACKING_URI=databricks`; the data becomes a
  Delta table; the schedule becomes a Job (`0 0 5 ? * SUN`, Asia/Jerusalem); the endpoint becomes Model Serving.
  Registering in Unity Catalog additionally needs an output signature on the model (not tried here).

"""Task 1.11: simulate production traffic against the ranking endpoint; report median and tail latency.

    .venv/bin/python scripts/load_test.py --name realistic --rate 0.5 --requests 60
    .venv/bin/python scripts/load_test.py --name stress    --rate 2   --requests 120

Rates from the data: registered visitors (one ranking each) average 42 an hour, 190 in the busiest hour, and at most 31
in one minute, i.e. 0.5 a second at the very peak. "stress" is 4x that: twice Prior Labs' limit of 60 predictions a minute.
Visitors are real registered sessions from the newest week (in memory only). They arrive at random moments (a Poisson
process) at the given average rate, each request on its own, so a slow answer never holds back the next visitor.
Every request spends at least 10,000 TabPFN tokens. Results: outputs/load/<name>.json and a printed summary.
"""
import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.visitors import newest_week_visitors  # noqa: E402


async def send(client, url, visitor, at, results):
    await asyncio.sleep(at)
    start = time.perf_counter()
    try:
        response = await client.post(url, json=visitor)
        results.append({"seconds": time.perf_counter() - start, "status": response.status_code,
                        "source": response.headers.get("X-Payout-Source"), "version": response.headers.get("X-Model-Version")})
    except httpx.HTTPError as error:
        results.append({"seconds": time.perf_counter() - start, "status": type(error).__name__, "source": None, "version": None})


def percentile(sorted_values, p):
    """Nearest-rank percentile: the value below which p% of the requests finished."""
    return sorted_values[min(len(sorted_values) - 1, max(0, round(p / 100 * len(sorted_values)) - 1))]


async def run(url, rate, count, seed):
    rng = random.Random(seed)
    visitors = newest_week_visitors(count, seed=seed)
    arrivals, t = [], 0.0
    for _ in visitors:
        arrivals.append(t)
        t += rng.expovariate(rate)  # Poisson arrivals: random gaps averaging 1/rate seconds
    results = []
    async with httpx.AsyncClient(timeout=60) as client:
        started = time.perf_counter()
        await asyncio.gather(*(send(client, url, v, at, results) for v, at in zip(visitors, arrivals)))
        elapsed = time.perf_counter() - started
    return results, elapsed


def summarize(name, rate, results, elapsed):
    ok = sorted(r["seconds"] for r in results if r["status"] == 200)
    sources = [r["source"] for r in results if r["status"] == 200]
    summary = {
        "name": name, "target_rate_per_s": rate, "requests": len(results), "ok": len(ok),
        "errors": len(results) - len(ok), "achieved_rate_per_s": round(len(results) / elapsed, 3),
        "fallback_share": round(sources.count("fallback") / len(sources), 3) if sources else None,
        "versions": sorted({r["version"] for r in results if r["version"]}),
    }
    if ok:
        summary.update({f"p{p}_s": round(percentile(ok, p), 3) for p in (50, 90, 95, 99)})
        summary.update(mean_s=round(statistics.mean(ok), 3), max_s=round(ok[-1], 3))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--rate", type=float, required=True, help="average requests per second")
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--url", default="http://localhost:8100/rank")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    results, elapsed = asyncio.run(run(args.url, args.rate, args.requests, args.seed))
    summary = summarize(args.name, args.rate, results, elapsed)
    out = ROOT / "outputs" / "load"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps({"summary": summary, "requests": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

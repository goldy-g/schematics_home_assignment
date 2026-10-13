"""Prove the small first-name table gives exactly the answers of the full names database, and measure both.

    .venv/bin/python scripts/names_table_check.py

The original gender feature (detect_gender_with_confidence, the predictor's copy) is called twice per name: once with
the full names-dataset (the original import) and once with FirstNameGender, the table every version carries. Names
checked: every first name and every last name in the database, each also lower-cased and padded with spaces, and every
first name in the data file, written as it arrives. Only counts are printed, never a name. Results:
outputs/names_table_check.json.
"""
import gzip
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import settings  # noqa: E402

LOAD = {  # each in a fresh process: seconds to import the original predictor, and the process's peak memory
    "full": "import time; t = time.perf_counter(); import bl_exp_payout_predictor; t = time.perf_counter() - t",
    "small": ("import time, sys; sys.path.insert(0, 'pipeline'); from ranker import FirstNameGender; "
              "t = time.perf_counter(); FirstNameGender(sys.argv[1]).import_predictor(); t = time.perf_counter() - t"),
}
REPORT = "; import resource, json; print(json.dumps({'seconds': round(t, 2), 'peak_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)}))"


def measure(kind, table_path):
    out = subprocess.run([sys.executable, "-c", LOAD[kind] + REPORT, str(table_path)], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def main():
    import pandas as pd
    import bl_exp_payout_predictor as original  # the full database, as the original code loads it
    sys.path.insert(0, str(ROOT / "pipeline"))
    from ranker import FirstNameGender

    full = original.nd
    names = {"M": "Male", "F": "Female"}
    table = {name: {names[code]: share for code, share in info["gender"].items()}
             for name, info in full.first_names.items() if info["gender"]}
    path = Path(tempfile.mkdtemp()) / settings.FIRST_NAMES_FILE
    with gzip.open(path, "wt") as f:  # the same build as pipeline/train.py save_first_name_gender()
        json.dump(table, f, separators=(",", ":"))
    small = FirstNameGender(path)

    data = pd.read_csv(settings.DATA_FILE, usecols=["fname"], low_memory=False)["fname"].dropna().astype(str)
    keys = list(full.first_names) + list(full.last_names)
    candidates = keys + [k.lower() for k in keys] + [f"  {k} " for k in keys[:50_000]]
    candidates += [str(x).strip().capitalize() for x in data]  # what the original code passes in (its L190)

    detect = original.BLPayoutModelsPredict.detect_gender_with_confidence  # reads the module's global nd
    start, mismatches = time.perf_counter(), 0
    for name in candidates:
        original.nd = full
        expected = detect(None, name)
        original.nd = small
        if detect(None, name) != expected:
            mismatches += 1
    original.nd = full

    result = {
        "names_checked": len(candidates), "mismatches": mismatches, "seconds_to_check": round(time.perf_counter() - start),
        "first_names_in_database": len(full.first_names), "first_names_with_gender": len(table),
        "table_file_mb": round(path.stat().st_size / 2**20, 1),
        "load_full_database": measure("full", path), "load_small_table": measure("small", path),
    }
    out = ROOT / "outputs" / "names_table_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()

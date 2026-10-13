"""Read the numbers out of the researcher's log, so MLflow can compare runs. The log itself is saved as it is.

The log holds: three row counts, then (test mode only) CatBoost reports for the whole test week and for each test
day, then TabPFN's payout errors overall and for predictions at or above $5, $10, $15 and $20.
"""
import re

ROW_COUNTS = {
    "rows_raw": r"raw data rows: (\d+)",
    "rows_registered": r"can be leads: (\d+)",
    "rows_survey_ok": r"sufficient to processing: (\d+)",
}


def row_counts(text):
    """How many rows survived each of the original filters."""
    return {name: int(m.group(1)) for name, pattern in ROW_COUNTS.items() if (m := re.search(pattern, text))}


def catboost_reports(text):
    """One dict per classification report, in log order: the whole test week first, then test days 1 to 7.
    Each holds the scores for class 1 (the brand accepted the lead), the accuracy and the number of rows."""
    reports = []
    for block in text.split("CB accuracy report")[1:]:
        lead = re.search(r"^\s+1\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", block, re.M)
        accuracy = re.search(r"^\s+accuracy\s+([\d.]+)\s+(\d+)\s*$", block, re.M)
        report = {"accuracy": float(accuracy[1]), "rows": int(accuracy[2])} if accuracy else {}
        if lead:  # a day can have no accepted leads at all, so no class-1 line
            report.update(precision_lead=float(lead[1]), recall_lead=float(lead[2]), f1_lead=float(lead[3]))
        reports.append(report)
    return reports


def payout_errors(text):
    """TabPFN's errors: (overall MAPE, overall MAE), then {threshold: (MAPE, MAE)}."""
    number = r"([-+\d.eE]+)"
    mape = [float(x) for x in re.findall(r"mean_absolute_percentage_error:" + number, text)]
    mae = [float(x) for x in re.findall(r"mean_absolute_error:" + number, text)]
    thresholds = [int(x) for x in re.findall(r"thr: (\d+)", text)]
    if not mape:
        return None, {}
    return (mape[0], mae[0]), {thr: (mape[i + 1], mae[i + 1]) for i, thr in enumerate(thresholds)}

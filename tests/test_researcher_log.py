"""Reading the numbers out of the researcher's log (a shortened copy of its real format)."""
from pipeline import researcher_log

LOG = """2026-10-11 16:12:34,575 - root - INFO - raw data rows: 128442
2026-10-11 16:12:34,601 - root - INFO - register date exists - can be leads: 62724
2026-10-11 16:12:34,647 - root - INFO - register date exists - survey answers are sufficient to processing: 62286
2026-10-11 16:13:01,376 - root - INFO - CB accuracy report - sold_to_client for all test
              precision    recall  f1-score   support

           0       0.85      0.73      0.78      5260
           1       0.40      0.58      0.47      1658

    accuracy                           0.69      6918
   macro avg       0.62      0.65      0.63      6918
weighted avg       0.74      0.69      0.71      6918

2026-10-11 16:13:01,384 - root - INFO - CB accuracy report sold_to_client (BL) for day {day}
 :              precision    recall  f1-score   support

           0       0.80      0.79      0.80      1188
           1       0.55      0.56      0.56       537

    accuracy                           0.72      1725
   macro avg       0.67      0.68      0.68      1725
weighted avg       0.72      0.72      0.72      1725

2026-10-11 16:13:01,417 - root - INFO - CB accuracy report sold_to_client (BL) for day {day}
 :              precision    recall  f1-score   support

           0       1.00      1.00      1.00         1

    accuracy                           1.00         1
   macro avg       1.00      1.00      1.00         1
weighted avg       1.00      1.00      1.00         1

2026-10-11 16:13:04,817 - root - INFO - Overall:
 Payout mean_absolute_percentage_error:0.05430900455377897
2026-10-11 16:13:04,817 - root - INFO - Payout mean_absolute_error:2.3383725815257805
2026-10-11 16:13:04,817 - root - INFO - thr: 5
2026-10-11 16:13:04,818 - root - INFO - Payout mean_absolute_percentage_error:0.054
2026-10-11 16:13:04,818 - root - INFO - Payout mean_absolute_error:2.3
"""


def test_row_counts():
    assert researcher_log.row_counts(LOG) == {"rows_raw": 128442, "rows_registered": 62724, "rows_survey_ok": 62286}


def test_catboost_reports_week_then_days():
    week, day1, day_without_leads = researcher_log.catboost_reports(LOG)
    assert week == {"accuracy": 0.69, "rows": 6918, "precision_lead": 0.40, "recall_lead": 0.58, "f1_lead": 0.47}
    assert day1["f1_lead"] == 0.56 and day1["rows"] == 1725
    assert day_without_leads == {"accuracy": 1.0, "rows": 1}  # no class-1 line, so no lead scores


def test_payout_errors_overall_then_thresholds():
    overall, by_threshold = researcher_log.payout_errors(LOG)
    assert overall == (0.05430900455377897, 2.3383725815257805)
    assert by_threshold == {5: (0.054, 2.3)}


def test_production_log_has_no_scores():
    assert researcher_log.catboost_reports("raw data rows: 1") == []
    assert researcher_log.payout_errors("raw data rows: 1") == (None, {})

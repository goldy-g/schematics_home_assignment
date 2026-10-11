"""The weekly schedule: Sunday 05:00 in Israel, including the week the clocks change."""
from datetime import datetime
from zoneinfo import ZoneInfo

from pipeline.scheduler import next_runs

ISRAEL = ZoneInfo("Asia/Jerusalem")
UTC = ZoneInfo("UTC")


def test_next_run_is_the_coming_sunday_at_five():
    assert next_runs(datetime(2026, 10, 11, 20, 0, tzinfo=ISRAEL))[0] == datetime(2026, 10, 18, 5, 0, tzinfo=ISRAEL)


def test_early_sunday_morning_runs_the_same_day():
    assert next_runs(datetime(2026, 10, 11, 4, 59, tzinfo=ISRAEL))[0] == datetime(2026, 10, 11, 5, 0, tzinfo=ISRAEL)


def test_stays_at_five_local_when_the_clocks_change():
    # Israel leaves summer time on Sunday 2026-10-25 at 02:00. 05:00 local is 02:00 UTC before that day, 03:00 UTC after.
    runs = next_runs(datetime(2026, 10, 12, 0, 0, tzinfo=ISRAEL), 3)
    assert [r.astimezone(ISRAEL).hour for r in runs] == [5, 5, 5]
    assert [r.astimezone(UTC).hour for r in runs] == [2, 3, 3]

"""Start the weekly job on its cron schedule. It runs until stopped (as its own container in docker compose).

    python -m pipeline.scheduler             wait for each Sunday 05:00 (Israel), then run the weekly job
    python -m pipeline.scheduler --next 3    print the next 3 run times and exit

The rule is cron's "0 5 * * 0", read in Asia/Jerusalem time, so the job stays at 05:00 local when the clocks change.
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from croniter import croniter

from pipeline import settings

ZONE = ZoneInfo(settings.TIMEZONE)


def next_runs(after, count=1):
    """The next `count` run times after the moment `after`, in Israel time.
    croniter is given plain wall-clock time, then the zone is attached: given zone-aware times, croniter keeps the old
    UTC offset across a clock change and lands an hour off (06:00 instead of 05:00 on 2026-10-25)."""
    wall_clock = after.astimezone(ZONE).replace(tzinfo=None)
    schedule = croniter(settings.WEEKLY_CRON, wall_clock)
    return [schedule.get_next(datetime).replace(tzinfo=ZONE) for _ in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--next", type=int, metavar="N", help="print the next N run times and exit")
    args = parser.parse_args()
    if args.next:
        for when in next_runs(datetime.now(ZONE), args.next):
            print(f"{when:%a %Y-%m-%d %H:%M %Z}")
        return

    while True:
        when = next_runs(datetime.now(ZONE))[0]
        print(f"next weekly run: {when:%a %Y-%m-%d %H:%M %Z}", flush=True)
        while (seconds_left := (when - datetime.now(ZONE)).total_seconds()) > 0:
            time.sleep(min(seconds_left, 60))  # check every minute, so a sleeping laptop or a clock change can't skip it
        # A separate process: if the job crashes, the scheduler keeps going, and the job's memory is freed when it ends.
        result = subprocess.run([sys.executable, "-m", "pipeline.weekly"])
        print(f"weekly run finished, exit code {result.returncode}", flush=True)


if __name__ == "__main__":
    main()

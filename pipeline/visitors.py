"""Real visitors from the data, as request dicts, for the parity test and the load test. They stay in memory."""
import math

import pandas as pd

from pipeline import settings

FIELDS = list(settings.EXAMPLE_VISITOR)
SURVEY = ["credit_score", "industry", "loan_amount", "loan_reason", "monthly_revenue", "time_in_business", "device_type",
          "business_type"]


def newest_week_visitors(count, seed=0):
    """`count` registered visitors from the newest week who answered enough of the survey (their filter: 5 of 8)."""
    data = pd.read_csv(settings.DATA_FILE, low_memory=False)
    data = data[data["register_date"].notna() & (data[SURVEY].notna().sum(axis=1) >= 5)]
    data["session_dt"] = pd.to_datetime(data["session_dt"])
    week = data[data["session_dt"] > data["session_dt"].max() - pd.Timedelta(days=6)]
    rows = week.sample(count, random_state=seed, replace=count > len(week))
    rows["session_dt"] = rows["session_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    rows["cellphone"] = rows["cellphone"].astype("int64")
    # a missing answer arrives as JSON null, i.e. None (their code expects text or None, not a float NaN)
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
            for row in rows[FIELDS].to_dict("records")]

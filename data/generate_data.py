"""
Healthcare Staffing Demand Forecasting
======================================
Step 1: Generate synthetic Canadian healthcare workforce data
         (mirrors open data from CIHI - Canadian Institute for Health Information)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

# ─── Configuration ────────────────────────────────────────────────────────────
FACILITY_TYPES = {
    "Hospital":         {"base_demand": 85, "volatility": 0.18},
    "Long_Term_Care":   {"base_demand": 62, "volatility": 0.12},
    "Community_Clinic": {"base_demand": 38, "volatility": 0.22},
    "Urgent_Care":      {"base_demand": 55, "volatility": 0.28},
    "Mental_Health":    {"base_demand": 30, "volatility": 0.15},
}

PROVINCES = ["ON", "BC", "AB", "QC", "MB"]

FACILITIES = []
fid = 1
for ftype, params in FACILITY_TYPES.items():
    for province in PROVINCES:
        FACILITIES.append({
            "facility_id":   f"F{fid:03d}",
            "facility_name": f"{province} {ftype.replace('_', ' ')} Centre",
            "facility_type": ftype,
            "province":      province,
            "base_demand":   params["base_demand"],
            "volatility":    params["volatility"],
        })
        fid += 1

# ─── Date Range (2 years) ─────────────────────────────────────────────────────
START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2023, 12, 31)
dates      = pd.date_range(START_DATE, END_DATE, freq="h")


def seasonal_multiplier(dt):
    """Canadian healthcare seasonality: flu season Oct-Mar drives demand up."""
    month = dt.month
    if month in [12, 1, 2]:    return 1.30   # Winter peak (flu + cold)
    elif month in [10, 11]:     return 1.18   # Fall surge
    elif month in [3, 4]:       return 1.08   # Spring shoulder
    elif month in [7, 8]:       return 0.88   # Summer low
    else:                       return 1.00


def dow_multiplier(dt):
    """Day-of-week patterns."""
    dow = dt.weekday()
    mults = {0: 1.12, 1: 1.10, 2: 1.05, 3: 1.08, 4: 1.15,  # Mon–Fri
             5: 0.82, 6: 0.75}                                 # Sat–Sun
    return mults[dow]


def hour_multiplier(dt, facility_type):
    """Hourly demand curves differ by facility type."""
    h = dt.hour
    if facility_type == "Hospital":
        peaks = {8: 1.4, 9: 1.5, 10: 1.5, 11: 1.4, 14: 1.3, 15: 1.4,
                 16: 1.3, 19: 1.1, 20: 1.1, 21: 1.0}
    elif facility_type == "Urgent_Care":
        peaks = {9: 1.2, 10: 1.3, 17: 1.5, 18: 1.6, 19: 1.5, 20: 1.3, 21: 1.2}
    elif facility_type == "Community_Clinic":
        peaks = {9: 1.5, 10: 1.5, 11: 1.4, 14: 1.3, 15: 1.4}
    else:
        peaks = {9: 1.3, 10: 1.3, 14: 1.2, 15: 1.2}
    base = 0.4 if 0 <= h <= 5 else 0.7 if 6 <= h <= 7 else 1.0
    return peaks.get(h, base)


def holiday_multiplier(dt):
    """Canadian statutory holidays."""
    holidays_md = {
        (1, 1): 0.55,  (2, 20): 0.70, (4, 7): 0.65, (5, 22): 0.72,
        (7, 1): 0.60,  (8, 7): 0.70,  (9, 4): 0.72, (10, 9): 0.75,
        (11, 13): 0.85,(12, 25): 0.50,(12, 26): 0.55,
    }
    return holidays_md.get((dt.month, dt.day), 1.0)


def generate_covid_effect(dt):
    """COVID-19 waves impact on staffing demand."""
    if datetime(2022, 1, 1) <= dt <= datetime(2022, 3, 31):
        return 1.25   # Omicron wave
    elif datetime(2022, 6, 1) <= dt <= datetime(2022, 8, 31):
        return 0.95   # Summer lull
    elif datetime(2023, 1, 1) <= dt <= datetime(2023, 2, 28):
        return 1.10   # XBB wave
    return 1.0


# ─── Generate Records ─────────────────────────────────────────────────────────
print("Generating hourly staffing demand records...")
records = []

for facility in FACILITIES[:10]:  # Use 10 facilities for manageable dataset
    ftype = facility["facility_type"]
    base  = facility["base_demand"]
    vol   = facility["volatility"]

    for dt in dates:
        mult = (
            seasonal_multiplier(dt)
            * dow_multiplier(dt)
            * hour_multiplier(dt, ftype)
            * holiday_multiplier(dt)
            * generate_covid_effect(dt)
        )
        noise  = np.random.normal(0, vol * base * mult)
        demand = max(0, round(base * mult + noise, 1))

        records.append({
            "facility_id":   facility["facility_id"],
            "facility_name": facility["facility_name"],
            "facility_type": ftype,
            "province":      facility["province"],
            "timestamp":     dt,
            "date":          dt.date(),
            "hour":          dt.hour,
            "day_of_week":   dt.strftime("%A"),
            "month":         dt.month,
            "week_of_year":  dt.isocalendar()[1],
            "is_weekend":    int(dt.weekday() >= 5),
            "is_holiday":    int(holiday_multiplier(dt) < 1.0),
            "shift_demand":  demand,
        })

df = pd.DataFrame(records)
os.makedirs("data", exist_ok=True)
df.to_csv("data/staffing_demand_raw.csv", index=False)

# Also save facility metadata
pd.DataFrame(FACILITIES).to_csv("data/facilities.csv", index=False)

print(f"✓ Generated {len(df):,} records across {df['facility_id'].nunique()} facilities")
print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"  Columns: {list(df.columns)}")
print(f"  Saved to data/staffing_demand_raw.csv")

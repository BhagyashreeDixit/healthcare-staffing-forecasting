"""
feature_engineering.py
=======================
Transforms raw hourly demand into a rich feature matrix.
Outputs: data/features_daily.csv  (Prophet-ready, one row per facility-day)
         data/features_hourly.csv (ML-ready, hourly with all features)
"""

import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

CANADIAN_HOLIDAYS = {
    (1, 1):  "New Year's Day",
    (4, 7):  "Good Friday",
    (5, 22): "Victoria Day",
    (7, 1):  "Canada Day",
    (8, 7):  "Civic Holiday",
    (9, 4):  "Labour Day",
    (10, 9): "Thanksgiving",
    (11, 13):"Remembrance Day",
    (12, 25):"Christmas Day",
    (12, 26):"Boxing Day",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["facility_id", "timestamp"]).reset_index(drop=True)

    df["hour"]         = df["timestamp"].dt.hour
    df["dow"]          = df["timestamp"].dt.dayofweek
    df["day_name"]     = df["timestamp"].dt.day_name()
    df["month"]        = df["timestamp"].dt.month
    df["quarter"]      = df["timestamp"].dt.quarter
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["year"]         = df["timestamp"].dt.year
    df["is_weekend"]   = (df["dow"] >= 5).astype(int)
    df["is_night"]     = ((df["hour"] >= 23) | (df["hour"] <= 6)).astype(int)

    df["shift_type"] = df["hour"].apply(
        lambda h: "Day" if 7 <= h < 15 else ("Evening" if 15 <= h < 23 else "Night")
    )

    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"]   /  7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"]   /  7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["is_holiday"] = df["timestamp"].apply(
        lambda x: int((x.month, x.day) in CANADIAN_HOLIDAYS)
    )
    df["holiday_name"] = df["timestamp"].apply(
        lambda x: CANADIAN_HOLIDAYS.get((x.month, x.day), "")
    )

    season_map = {
        12:"Winter",1:"Winter",2:"Winter",
        3:"Spring",4:"Spring",5:"Spring",
        6:"Summer",7:"Summer",8:"Summer",
        9:"Fall",10:"Fall",11:"Fall"
    }
    df["season"]        = df["month"].map(season_map)
    df["is_flu_season"] = df["month"].isin([10,11,12,1,2,3]).astype(int)

    df = df.sort_values(["facility_id", "timestamp"])
    for lag in [1, 2, 24, 48, 168]:
        df[f"lag_{lag}h"] = df.groupby("facility_id")["shift_demand"].shift(lag)
    for window in [6, 12, 24, 168]:
        df[f"roll_mean_{window}h"] = (
            df.groupby("facility_id")["shift_demand"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
    df["roll_std_24h"] = (
        df.groupby("facility_id")["shift_demand"]
        .transform(lambda x: x.shift(1).rolling(24, min_periods=1).std())
    )

    fstats = (
        df.groupby("facility_id")["shift_demand"]
        .agg(fac_mean="mean", fac_std="std", fac_median="median")
        .reset_index()
    )
    df = df.merge(fstats, on="facility_id", how="left")
    df["demand_zscore"] = (
        (df["shift_demand"] - df["fac_mean"]) / df["fac_std"].replace(0, 1)
    )
    return df


def build_prophet_ready(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = df["timestamp"].dt.normalize()
    daily = (
        df.groupby(["facility_id", "facility_type", "province", "date"])
        .agg(
            y             = ("shift_demand",  "mean"),
            is_holiday    = ("is_holiday",    "max"),
            is_flu_season = ("is_flu_season", "max"),
        )
        .reset_index()
        .rename(columns={"date": "ds"})
    )
    return daily


if __name__ == "__main__":
    print("Loading raw data …")
    df_raw = pd.read_csv("data/raw_demand.csv", parse_dates=["timestamp"])
    print("Engineering features …")
    df_feat = engineer_features(df_raw)
    df_feat.to_csv("data/features_hourly.csv", index=False)
    print(f"✓ Hourly features {df_feat.shape} → data/features_hourly.csv")
    df_daily = build_prophet_ready(df_feat)
    df_daily.to_csv("data/features_daily.csv", index=False)
    print(f"✓ Daily features {df_daily.shape} → data/features_daily.csv")

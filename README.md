# 🏥 Healthcare Staffing Demand Forecasting

> **End-to-end Prophet forecasting pipeline for Canadian healthcare workforce planning**  
> Python · Prophet · Pandas · Matplotlib · SQL · Power BI

---

## 📋 Project Summary

Built an end-to-end forecasting pipeline using Canadian healthcare workforce open data (CIHI-style synthetic data), engineering time-based features and training a **Prophet** model to predict shift demand by day and hour across facility types.

| Metric | Result |
|--------|--------|
| **Forecast Accuracy** | **89%** (MAPE-based) |
| **Overstaffing Cost Reduction** | **22%** vs flat average baseline |
| **Facilities Modelled** | 10 across 5 Canadian provinces |
| **Forecast Horizon** | 90 days |
| **Data Granularity** | Hourly → Daily aggregation |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Ingestion                           │
│  CIHI Open Data (synthetic) → data/staffing_demand_raw.csv  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                Feature Engineering                          │
│  Cyclical encoding · Lag features · Rolling stats           │
│  Canadian holidays · Flu season · Shift categorization      │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌─────────────────────┐   ┌────────────────────────┐
│   Prophet Model     │   │   SQL Warehouse         │
│  Multiplicative     │   │  PostgreSQL schema      │
│  Yearly/Weekly      │   │  Materialized views     │
│  Canadian holidays  │   │  Analytical queries     │
└──────────┬──────────┘   └────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Outputs                                  │
│  forecasts_all.csv · model_metrics.csv · facility plots     │
│  Power BI Dashboard (dashboard/index.html)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
healthcare_staffing/
│
├── data/
│   ├── generate_data.py          # Synthetic CIHI-style data generator
│   ├── staffing_demand_raw.csv   # Generated: 174,970 hourly records
│   └── facilities.csv            # Facility metadata
│
├── models/
│   ├── feature_engineering.py   # Time features, lag/rolling, cyclical encoding
│   ├── train_prophet.py          # Prophet training, evaluation, forecast
│   └── saved/                    # Serialized models (if using Prophet)
│
├── sql/
│   └── schema_and_queries.sql    # Full PostgreSQL schema + 7 analytical queries
│
├── dashboard/
│   └── index.html                # Interactive Power BI-style dashboard
│
├── notebooks/
│   └── healthcare_forecasting.ipynb  # Full walkthrough notebook
│
├── outputs/
│   ├── model_metrics.csv         # Per-facility accuracy metrics
│   ├── forecasts_all.csv         # Historical + 90-day future forecasts
│   ├── actuals_daily.csv         # Daily aggregated actuals
│   ├── summary_dashboard.png     # Matplotlib summary plot
│   └── F001–F010_forecast.png    # Per-facility forecast plots
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Prophet requires `pystan`. On Apple Silicon:
> ```bash
> conda install -c conda-forge prophet
> ```

### 2. Generate data

```bash
cd healthcare_staffing
python data/generate_data.py
```

### 3. Train models & generate forecasts

```bash
python models/train_prophet.py
```

### 4. View the dashboard

Open `dashboard/index.html` in your browser.

### 5. (Optional) Load into PostgreSQL

```bash
psql -U youruser -d yourdb -f sql/schema_and_queries.sql
```

Then in Python:
```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('postgresql://user:pass@localhost/healthcare')
pd.read_csv('outputs/forecasts_all.csv').to_sql(
    'forecasts', engine, schema='staffing', if_exists='append', index=False
)
```

---

## 🔬 Feature Engineering

| Feature Category | Features |
|---|---|
| **Cyclical time** | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos` |
| **Lag features** | `lag_1h`, `lag_24h`, `lag_48h`, `lag_168h` (1-week lag) |
| **Rolling stats** | `rolling_mean_6h`, `rolling_mean_24h`, `rolling_std_24h` |
| **Calendar** | `is_holiday`, `is_flu_season`, `is_weekend`, `days_to_holiday` |
| **Shift type** | `Day` (07-15), `Evening` (15-23), `Night` (23-07) |
| **Facility stats** | `facility_mean`, `facility_std`, `demand_zscore` |

---

## 📊 Prophet Model Configuration

```python
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,          # aggregated to daily
    holidays=canadian_holidays_df,    # Canadian statutory holidays
    seasonality_mode='multiplicative', # better for healthcare patterns
    changepoint_prior_scale=0.15,     # moderate flexibility
    seasonality_prior_scale=12.0,
    interval_width=0.90               # 90% confidence interval
)
model.add_regressor('is_holiday', mode='multiplicative')
```

### Why multiplicative seasonality?
Healthcare demand is **proportional** to its base level — a Hospital with double the baseline also sees double the seasonal swing. Multiplicative mode captures this correctly.

---

## 💰 Cost Model

Overstaffing cost is simulated using standard Canadian healthcare rates:

| Scenario | Rate |
|----------|------|
| Excess staff-hour (overstaffing) | $65 CAD/hr |
| Staff shortage | $110 CAD/hr (overtime + agency) |

**Formula:**
```
cost_model    = Σ max(0, predicted - actual) × 8h × $65
cost_baseline = Σ max(0, flat_avg - actual) × 8h × $65
reduction     = (cost_baseline - cost_model) / cost_baseline × 100
```

---

## 🗃️ SQL Schema

The PostgreSQL schema includes:

- `staffing.facilities` — facility master data
- `staffing.shift_demand` — raw hourly records
- `staffing.daily_demand` — aggregated daily actuals
- `staffing.forecasts` — Prophet output (yhat, CI bounds)
- `staffing.model_metrics` — MAPE, RMSE, overstaffing reduction
- `staffing.staffing_recommendations` — FTE guidance from forecasts
- `staffing.mv_daily_summary` — materialized view for Power BI DirectQuery

---

## 📈 Power BI Dashboard

The `dashboard/index.html` file is a standalone interactive dashboard that mirrors a Power BI report, featuring:

- **Forecast vs Actuals** — per-facility time series with 90% CI band and 90-day future projection
- **Peak Staffing Windows** — hourly demand heatmap by facility type
- **Weekly Demand Trends** — seasonal patterns across 2 years
- **Shift Distribution** — Day / Evening / Night breakdown
- **Accuracy Comparison** — Prophet vs flat baseline
- **Metrics Table** — per-facility MAPE, RMSE, cost savings

> For production Power BI: connect via DirectQuery to `staffing.mv_daily_summary` in PostgreSQL.

---

## 📚 Data Sources

| Source | Description |
|--------|-------------|
| [CIHI](https://www.cihi.ca/en/access-data-and-reports) | Canadian Institute for Health Information open workforce data |
| [Statistics Canada](https://www.statcan.gc.ca) | Healthcare employment statistics |
| [Health Canada](https://www.canada.ca/en/health-canada.html) | Facility classifications |

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Data** | Pandas, NumPy |
| **Forecasting** | Prophet (Facebook/Meta), scikit-learn |
| **Visualization** | Matplotlib, Chart.js (dashboard) |
| **Database** | PostgreSQL 14+, SQLAlchemy |
| **Dashboard** | HTML/CSS/JS · Chart.js (mirrors Power BI) |
| **Notebook** | Jupyter |

---

## 📄 License

MIT License — free to use and adapt for academic or professional portfolio purposes.

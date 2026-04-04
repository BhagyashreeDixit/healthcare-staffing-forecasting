-- =============================================================================
-- Healthcare Staffing Demand Forecasting — SQL Schema & Queries
-- Database : PostgreSQL 14+
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS staffing;
SET search_path = staffing, public;

-- ── 1. Facilities ─────────────────────────────────────────────────────────────
CREATE TABLE facilities (
    facility_id   VARCHAR(12)  PRIMARY KEY,
    facility_name VARCHAR(120) NOT NULL,
    facility_type VARCHAR(30)  NOT NULL
        CHECK (facility_type IN (
            'Hospital','Long_Term_Care','Community_Clinic',
            'Urgent_Care','Mental_Health')),
    province      CHAR(2)      NOT NULL,
    region        VARCHAR(50),
    bed_capacity  SMALLINT,
    active        BOOLEAN      DEFAULT TRUE,
    created_at    TIMESTAMPTZ  DEFAULT NOW()
);

-- ── 2. Hourly shift demand ────────────────────────────────────────────────────
CREATE TABLE shift_demand (
    demand_id       BIGSERIAL    PRIMARY KEY,
    facility_id     VARCHAR(12)  NOT NULL REFERENCES facilities,
    recorded_at     TIMESTAMPTZ  NOT NULL,
    shift_date      DATE         GENERATED ALWAYS AS (recorded_at::DATE) STORED,
    hour_of_day     SMALLINT     GENERATED ALWAYS AS
                        (EXTRACT(HOUR FROM recorded_at)::SMALLINT) STORED,
    shift_type      VARCHAR(10)  NOT NULL CHECK (shift_type IN ('Day','Evening','Night')),
    staff_demanded  NUMERIC(8,2) NOT NULL CHECK (staff_demanded >= 0),
    staff_scheduled NUMERIC(8,2),
    is_holiday      BOOLEAN      DEFAULT FALSE,
    data_source     VARCHAR(50)  DEFAULT 'CIHI_OpenData',
    UNIQUE (facility_id, recorded_at)
);
CREATE INDEX idx_sd_fac_date ON shift_demand (facility_id, shift_date);
CREATE INDEX idx_sd_date     ON shift_demand (shift_date DESC);

-- ── 3. Daily aggregated actuals ───────────────────────────────────────────────
CREATE TABLE daily_demand (
    daily_id        BIGSERIAL    PRIMARY KEY,
    facility_id     VARCHAR(12)  NOT NULL REFERENCES facilities,
    demand_date     DATE         NOT NULL,
    avg_hourly_fte  NUMERIC(8,2) NOT NULL,
    total_fte_hours NUMERIC(10,2),
    peak_demand     NUMERIC(8,2),
    peak_hour       SMALLINT,
    is_holiday      BOOLEAN      DEFAULT FALSE,
    is_flu_season   BOOLEAN      DEFAULT FALSE,
    day_of_week     VARCHAR(10),
    week_of_year    SMALLINT,
    month_num       SMALLINT,
    quarter_num     SMALLINT,
    UNIQUE (facility_id, demand_date)
);
CREATE INDEX idx_dd_date ON daily_demand (demand_date DESC);

-- ── 4. Prophet forecasts ──────────────────────────────────────────────────────
CREATE TABLE forecasts (
    forecast_id   BIGSERIAL    PRIMARY KEY,
    facility_id   VARCHAR(12)  NOT NULL REFERENCES facilities,
    forecast_date DATE         NOT NULL,
    model_version VARCHAR(20)  DEFAULT 'prophet_v1',
    yhat          NUMERIC(10,2) NOT NULL,
    yhat_lower    NUMERIC(10,2),
    yhat_upper    NUMERIC(10,2),
    is_future     BOOLEAN      DEFAULT TRUE,
    generated_at  TIMESTAMPTZ  DEFAULT NOW(),
    UNIQUE (facility_id, forecast_date, model_version)
);
CREATE INDEX idx_fc_fac_date ON forecasts (facility_id, forecast_date);

-- ── 5. Model metrics ─────────────────────────────────────────────────────────
CREATE TABLE model_metrics (
    metric_id                SERIAL      PRIMARY KEY,
    facility_id              VARCHAR(12) REFERENCES facilities,
    facility_type            VARCHAR(30),
    model_version            VARCHAR(20) DEFAULT 'prophet_v1',
    evaluated_at             TIMESTAMPTZ DEFAULT NOW(),
    mape                     NUMERIC(6,2),
    rmse                     NUMERIC(10,2),
    mae                      NUMERIC(10,2),
    r2                       NUMERIC(7,4),
    accuracy_pct             NUMERIC(6,2),
    baseline_mape            NUMERIC(6,2),
    mape_improvement         NUMERIC(6,2),
    overstaffing_reduction   NUMERIC(6,2),
    overstaff_cost_model     NUMERIC(14,2),
    overstaff_cost_baseline  NUMERIC(14,2),
    train_days               INT,
    test_days                INT
);

-- ── 6. Staffing recommendations ───────────────────────────────────────────────
CREATE TABLE staffing_recommendations (
    rec_id          BIGSERIAL   PRIMARY KEY,
    facility_id     VARCHAR(12) REFERENCES facilities,
    rec_date        DATE        NOT NULL,
    shift_type      VARCHAR(10),
    recommended_fte NUMERIC(6,1),
    min_fte         NUMERIC(6,1),
    max_fte         NUMERIC(6,1),
    confidence_pct  NUMERIC(5,2),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- ANALYTICAL QUERIES
-- =============================================================================

-- Q1: Accuracy by facility type
SELECT
    f.facility_type,
    COUNT(DISTINCT mm.facility_id)           AS facilities,
    ROUND(AVG(mm.accuracy_pct), 2)           AS avg_accuracy_pct,
    ROUND(AVG(mm.mape), 2)                   AS avg_mape,
    ROUND(AVG(mm.r2), 4)                     AS avg_r2,
    ROUND(AVG(mm.overstaffing_reduction), 2) AS avg_overstaff_saved_pct,
    ROUND(SUM(mm.overstaff_cost_baseline - mm.overstaff_cost_model), 2) AS total_cad_saved
FROM model_metrics mm JOIN facilities f USING (facility_id)
WHERE mm.model_version = 'prophet_v1'
GROUP BY f.facility_type ORDER BY avg_accuracy_pct DESC;

-- Q2: Predicted vs actual (last 60 days)
SELECT
    dd.demand_date, dd.facility_id, f.facility_type, f.province,
    dd.avg_hourly_fte AS actual, fc.yhat AS predicted,
    fc.yhat_lower, fc.yhat_upper,
    ROUND(ABS(dd.avg_hourly_fte-fc.yhat)/NULLIF(dd.avg_hourly_fte,0)*100,2) AS abs_pct_err,
    dd.is_holiday, TO_CHAR(dd.demand_date,'Day') AS day_name
FROM daily_demand dd
JOIN facilities f USING (facility_id)
JOIN forecasts  fc ON fc.facility_id=dd.facility_id
                  AND fc.forecast_date=dd.demand_date
                  AND fc.model_version='prophet_v1'
WHERE dd.demand_date >= CURRENT_DATE - INTERVAL '60 days'
ORDER BY dd.demand_date, dd.facility_id;

-- Q3: Peak staffing windows by hour × facility type
SELECT
    f.facility_type, sd.hour_of_day, sd.shift_type,
    ROUND(AVG(sd.staff_demanded),2)    AS avg_demand,
    ROUND(MAX(sd.staff_demanded),2)    AS peak_demand,
    ROUND(STDDEV(sd.staff_demanded),2) AS demand_std,
    COUNT(*)                           AS n
FROM shift_demand sd JOIN facilities f USING (facility_id)
WHERE sd.shift_date >= CURRENT_DATE - INTERVAL '90 days'
  AND sd.is_holiday = FALSE
GROUP BY f.facility_type, sd.hour_of_day, sd.shift_type
ORDER BY f.facility_type, avg_demand DESC;

-- Q4: 90-day forward staffing recommendations
SELECT
    fc.forecast_date, f.facility_id, f.facility_name, f.facility_type, f.province,
    ROUND(fc.yhat,2)         AS forecast_avg_fte,
    ROUND(fc.yhat_lower,2)   AS lower_bound,
    ROUND(fc.yhat_upper,2)   AS upper_bound,
    CEIL(fc.yhat*8)          AS recommended_shift_fte,
    CEIL(fc.yhat_lower*8)    AS min_shift_fte,
    CEIL(fc.yhat_upper*8)    AS max_shift_fte,
    TO_CHAR(fc.forecast_date,'Day') AS day_name
FROM forecasts fc JOIN facilities f USING (facility_id)
WHERE fc.is_future=TRUE AND fc.model_version='prophet_v1'
  AND fc.forecast_date BETWEEN CURRENT_DATE AND CURRENT_DATE+INTERVAL '90 days'
ORDER BY fc.forecast_date, f.facility_type;

-- Q5: Facility-level demand variance
SELECT
    f.facility_id, f.facility_name, f.facility_type, f.province,
    ROUND(AVG(dd.avg_hourly_fte),2)                  AS mean_demand,
    ROUND(STDDEV(dd.avg_hourly_fte),2)               AS std_demand,
    ROUND(MIN(dd.avg_hourly_fte),2)                  AS min_demand,
    ROUND(MAX(dd.avg_hourly_fte),2)                  AS max_demand,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY dd.avg_hourly_fte),2) AS p25,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY dd.avg_hourly_fte),2) AS p75,
    ROUND(STDDEV(dd.avg_hourly_fte)/NULLIF(AVG(dd.avg_hourly_fte),0)*100,2)  AS cv_pct
FROM daily_demand dd JOIN facilities f USING (facility_id)
GROUP BY f.facility_id, f.facility_name, f.facility_type, f.province
ORDER BY cv_pct DESC;

-- Q6: Weekly trends
SELECT
    DATE_TRUNC('week',dd.demand_date)::DATE AS week_start,
    f.facility_type,
    ROUND(AVG(dd.avg_hourly_fte),2) AS avg_actual,
    ROUND(AVG(fc.yhat),2)           AS avg_forecast
FROM daily_demand dd
JOIN facilities  f USING (facility_id)
LEFT JOIN forecasts fc ON fc.facility_id=dd.facility_id
                       AND fc.forecast_date=dd.demand_date
                       AND fc.model_version='prophet_v1'
GROUP BY DATE_TRUNC('week',dd.demand_date), f.facility_type
ORDER BY week_start, f.facility_type;

-- Q7: Overstaffing cost analysis
WITH cost AS (
    SELECT
        mm.facility_id, f.facility_type, f.province,
        mm.overstaff_cost_model    AS cost_model,
        mm.overstaff_cost_baseline AS cost_baseline,
        mm.overstaff_cost_baseline - mm.overstaff_cost_model AS saved,
        mm.overstaffing_reduction  AS pct_reduction
    FROM model_metrics mm JOIN facilities f USING (facility_id)
    WHERE mm.model_version='prophet_v1'
)
SELECT *, ROUND(SUM(saved) OVER (),2) AS total_saved_all
FROM cost ORDER BY saved DESC;

-- Materialized view for Power BI DirectQuery
CREATE MATERIALIZED VIEW mv_dashboard AS
SELECT
    dd.demand_date, dd.facility_id, f.facility_name,
    f.facility_type, f.province,
    dd.avg_hourly_fte    AS actual_demand,
    dd.is_holiday, dd.is_flu_season,
    TO_CHAR(dd.demand_date,'Day')       AS day_of_week,
    EXTRACT(MONTH  FROM dd.demand_date) AS month_num,
    EXTRACT(QUARTER FROM dd.demand_date) AS quarter_num,
    fc.yhat              AS predicted_demand,
    fc.yhat_lower, fc.yhat_upper, fc.is_future,
    CASE WHEN NOT fc.is_future
         THEN ROUND(ABS(dd.avg_hourly_fte-fc.yhat)/NULLIF(dd.avg_hourly_fte,0)*100,2)
    END                  AS abs_pct_error,
    mm.accuracy_pct, mm.overstaffing_reduction
FROM daily_demand dd
JOIN facilities f USING (facility_id)
LEFT JOIN forecasts fc ON fc.facility_id=dd.facility_id
                       AND fc.forecast_date=dd.demand_date
                       AND fc.model_version='prophet_v1'
LEFT JOIN model_metrics mm ON mm.facility_id=dd.facility_id
                           AND mm.model_version='prophet_v1'
WITH DATA;
CREATE UNIQUE INDEX ON mv_dashboard (facility_id, demand_date);

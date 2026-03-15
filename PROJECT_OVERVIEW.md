# Project Overview

This document explains what the codebase does, how it is structured, and the key implementation details behind each dashboard section.

**Purpose**
This is a Streamlit dashboard for the Ugunja Hackathon that presents three analytics pillars for LPG logistics and fleet maintenance: route optimization ROI, predictive maintenance risk, and demand forecasting. The app consumes precomputed model outputs stored in CSV files.

**High-level architecture**
- `app.py` is the Streamlit entrypoint and routes to page modules.
- `ugunja_app/` contains shared configuration, data loading, styles, and sidebar UI.
- Each analytics pillar is a separate page module under `ugunja_app/pages/`.
- Data is loaded once with `st.cache_data` for performance.

**Data inputs**
- `data/demand_forecasting.csv` provides daily, neighborhood-level demand predictions and context features.
- `data/predictive_maintenance.csv` provides vehicle risk scores and predicted failure types over time.
- `data/route_optimization.csv` provides paired baseline vs AI route metrics.
- `data/training/` provides synthetic datasets used only for the demo training notebook.

**App flow and logic**
- Page configuration and CSS are set at startup to establish branding and layout.
- `load_data()` reads the three CSVs, parses dates, validates schema, and caches results.
- The sidebar drives page selection and displays a logo and project context.

**Route Optimization (ROI) page**
- Computes savings by comparing `Standard_Route` vs `Ugunja_AI_Route` totals.
- Uses user-selected sliders for cost per km and cost per hour to compute fuel and labor savings.
- Visualizes savings with a Waterfall chart and contrasts efficiency using a distance vs time scatter plot.

**Predictive Maintenance page**
- KPI summary shows fleet size, high-risk count (`predicted_risk_score > 0.8`), and average risk.
- A treemap summarizes risk by failure type and vehicle, with mileage driving tile size.
- Anomaly detection views include fuel consumption vs risk scatter and per-vehicle risk distributions.
- A vehicle drill-down displays fuel consumption trend and risk trend over time.

**Demand Forecasting page**
- KPI summary includes total predicted demand, top neighborhood, and peak day.
- Calendar heatmap visualizes day-of-week demand intensity by month.
- Time series decomposition uses `statsmodels` (weekly period) to show trend, seasonality, and residuals.
- Correlation heatmap evaluates the relationship between demand and features like day of week, weather, and holidays.

**Dependencies**
- Streamlit for UI and dashboard layout.
- Pandas for data loading and aggregation.
- Plotly for interactive visualization.
- Statsmodels for time series decomposition.
- Pytest (dev) for schema validation tests.
- Scikit-learn (dev) for the synthetic training notebook.

**Code quality notes**
- Data validation is minimal. If CSV schema changes, some visuals can break without clear errors.
- The logo is loaded from an external URL, which can fail in restricted environments.
- The training code for the ML models is not part of this repository; this app visualizes outputs.

**Suggested next improvements (optional)**
- Add a `models/` or `pipelines/` package for training and evaluation code if available.
- Add lightweight schema validation for CSV inputs.
- Add tests for key aggregations (e.g., ROI calculations, KPI summaries).
- Move configuration constants (colors, thresholds) into a `config.py`.

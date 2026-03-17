# Smart LPG Logistics & Predictive Maintenance 
##(Green Wells Energies Hackathon: https://green-wells-innovation.devpost.com/) 

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visuals-3F4F75?logo=plotly&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Time%20Series-6D6D6D)
![CI](https://github.com/OsbornNyakaru/ugunja_ml_streamlit/actions/workflows/ci.yml/badge.svg)

A Streamlit analytics dashboard that showcases route optimization ROI, predictive maintenance risk, and LPG demand forecasting for a simulated fleet and distribution network.

**What this app demonstrates**
- Route optimization ROI via side-by-side standard vs AI routes and a savings calculator.
- Predictive maintenance risk scoring with fleet overview and vehicle drill-downs.
- Demand forecasting insights with seasonality decomposition and demand driver correlations.

**Modeling context (from the hackathon work)**
- Predictive maintenance: Scikit-learn Random Forest with feature engineering on mileage, load cycles, and service history; reported ~83% accuracy on held-out test data.
- Demand forecasting: Pandas pipeline + Scikit-learn Logistic Regression to authenticate cylinder distribution and predict regional LPG shortages.

See `reports/model_evaluation.md` for the current verification status and what is needed to make results reproducible. This repo focuses on visualization of model outputs; training code and experiments are not included here.

**Project structure**
- `app.py` Streamlit entrypoint and app wiring.
- `ugunja_app/` App modules (data loading, styles, sidebar, pages).
- `data/demand_forecasting.csv` Model outputs by date and neighborhood.
- `data/predictive_maintenance.csv` Fleet risk scores and predicted failure types.
- `data/route_optimization.csv` Standard vs AI route metrics.
- `data/training/` Synthetic training datasets for the sample notebook.
- `requirements.txt` Runtime dependencies.
- `PROJECT_OVERVIEW.md` Detailed codebase walkthrough.
- `SKILLS.md` Skills snapshot aligned with this project.
- `notebooks/model_training.ipynb` Synthetic training notebook for demo purposes.

**Data schema (excerpt)**
| File | Key fields |
| --- | --- |
| `data/demand_forecasting.csv` | `date`, `neighborhood`, `day_of_week`, `weather`, `is_holiday`, `predicted_demand_cylinders` |
| `data/predictive_maintenance.csv` | `date`, `vehicle_id`, `mileage`, `avg_engine_temp_c`, `avg_fuel_consumption_l_100km`, `predicted_risk_score`, `predicted_failure_type` |
| `data/route_optimization.csv` | `route_id`, `route_type`, `num_stops`, `total_distance_km`, `total_time_hours` |

**Run locally**
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

**Tests**
```bash
pip install -r requirements-dev.txt
python -m pytest
```

**Lint & format**
```bash
pip install -r requirements-dev.txt
python -m ruff format app.py ugunja_app tests
python -m ruff check app.py ugunja_app tests --fix
```

**Release Notes**
- 2026-03-15: Modularized the app, added schema validation + tests, added synthetic training assets, and added CI.

**Notes for reviewers**
- The data is simulated for demo purposes to highlight the product experience.
- The UI emphasizes ROI, risk visibility, and demand planning outcomes.
- The app is modularized for clarity; production hardening would include auth, data validation gates, and monitored data pipelines.

If you want a quick technical walkthrough, start with `PROJECT_OVERVIEW.md`.

# Project Ugunja — Smart LPG Logistics (Notebook-First)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visuals-3F4F75?logo=plotly&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Prototype-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)

A production-grade, notebook-centric ML workflow for LPG fleet predictive maintenance, regional shortage prediction, and route optimisation. The project uses physically-informed synthetic telemetry to keep the pipeline fully reproducible without proprietary data.

**Primary notebooks**
- `notebooks/01_predictive_maintenance_demand.ipynb`
- `notebooks/02_route_optimisation.ipynb`

**Key outcomes**
- Predictive maintenance with Random Forest + engineered signals (RUL proxy, stress index, rolling stats).
- Demand/shortage prediction with Logistic Regression and robust evaluation.
- Route optimisation analysis with geo features, risk-aware routing, and visualisation.
- Target accuracy guideline: **~0.89** on synthetic data (exact results vary by seed).

**Pipeline overview**
```mermaid
flowchart LR
  A[Synthetic Telemetry] --> B[EDA & QA]
  B --> C[Feature Engineering]
  C --> D[Model Training]
  D --> E[Evaluation & Bias Checks]
  E --> F[Saved Artifacts]
```

**Notebook map**
```mermaid
flowchart TB
  N1[01_predictive_maintenance_demand.ipynb]
  N2[02_route_optimisation.ipynb]
  N1 -->|Shared utils| S[src/]
  N2 -->|Shared utils| S[src/]
  S --> G[generator.py]
  S --> F[engineering.py]
  S --> U[geo.py]
```

**Repository layout**
```text
ugunja_ml_streamlit/
  notebooks/
    01_predictive_maintenance_demand.ipynb
    02_route_optimisation.ipynb
  src/
    data/
      generator.py
    features/
      engineering.py
    utils/
      geo.py
  models/
  tests/
  requirements.txt
  Dockerfile
  README.md
```

**Quickstart**
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
jupyter lab
```

**Docker**
```bash
docker build -t ugunja-ml .
docker run -p 8888:8888 ugunja-ml
```

**Shared modules**
- `src/data/generator.py`: synthetic data generation aligned with the notebooks.
- `src/features/engineering.py`: shared feature engineering for maintenance and route tasks.
- `src/utils/geo.py`: depot/destination config, road conditions, and haversine distance.

**Visual outputs produced in notebooks**
- Feature importance plots for maintenance and shortage models.
- Confusion matrices and classification reports.
- Geo heatmaps and optimised route maps.

**Notes**
- Data is synthetic by design; treat metrics as illustrative, not production benchmarks.
- Model artefacts may be saved into `models/` from within the notebooks.

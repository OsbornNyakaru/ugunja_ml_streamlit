# Model Evaluation Summary

**Status**: Reported results (training/evaluation artifacts not included in this repository).

## Predictive Maintenance (Random Forest)
- **Reported accuracy**: ~83% on held-out test data.
- **Source**: Hackathon evaluation notes / resume claim.
- **Reproducibility**: Not reproducible in this repo because training data, feature pipelines, and evaluation scripts are not included.

## Demand Forecasting (Logistic Regression)
- **Outcome**: Pipeline used for distribution authentication and shortage prediction.
- **Reproducibility**: Not reproducible in this repo for the same reasons as above.

## What would make this fully verifiable
- Add training data (or a representative sample).
- Add feature engineering + model training code.
- Add an evaluation script that produces the metrics above.
- Record the exact split strategy and random seeds.

## Synthetic demo assets
- `notebooks/model_training.ipynb` trains on synthetic data in `data/training/`.
- These assets demonstrate the pipeline only and do not validate the reported ~83% accuracy.

# BioPerformance AI Readiness System

## Overview
The BioPerformance AI Readiness System is a machine learning pipeline designed to predict next-day athlete wellness. Fixed formulas and rule-based engines often fail to capture the non-linear, highly individualized nature of athletic fatigue and recovery. This project implements Approach B: an XGBoost gradient boosting model trained on temporal lag features, paired with SHAP (SHapley Additive exPlanations) for per-prediction explainability.

The system transforms raw time-series data (sleep, fatigue, soreness, mood, training load) into a tabular feature matrix, learns non-linear interactions between training load and recovery, and outputs fully auditable predictions to be synthesized into coaching insights.

## Architecture
The pipeline consists of three core components:
1. Feature Engineering: Converts raw daily check-ins and training sessions into temporal lag features (t0, t1, t2), 7-day and 28-day rolling statistics, Z-scores against baselines, and Acute:Chronic Workload Ratio (ACWR).
2. XGBoost Model: A gradient boosting regression model that learns individual athlete patterns. Utilizes TimeSeriesSplit to prevent data leakage.
3. SHAP Explainability: Calculates feature attributions for every prediction. The top 3 drivers are extracted to provide actionable, transparent reasoning for the coaching staff.

## Methodology
The implementation is grounded in established sports science literature:

- Rossi et al. (2022): Defines the temporal lag feature engineering blueprint.
- Taber et al. (2024): Validates XGBoost architecture for collegiate athlete performance prediction.
- Schliep & Schafer (2021): Informs the treatment of multivariate ordinal wellness data.

## Installation and Setup
### Prerequisites
- Python 3.12

### Installation
Clone the repository and install the required dependencies:

```md
git clone <repository-url>
cd bioperformance
python -m venv .venvsource .venv/bin/activate  
# On Windows: .venv\Scripts\activatepip install -r requirements.txt
```

## Usage
1. Generate Synthetic Data
Generate 90 days of synthetic athlete data for pipeline testing:

```bash
python src/data_synthesis.py
```

2. Build Feature Matrix
Transform raw data into temporal lag features and target variables:

```bash
python src/feature_engineering.py
```

3. Train and Evaluate Model
Train the XGBoost model using a time-based split and evaluate Mean Absolute Error (MAE):

```bash
python src/readiness_model.py
```

## Project Structure

```md
bioperformance/
├── data/                  # Generated CSV data (ignored in git)
├── src/                   # Source code modules
│   ├── data_synthesis.py  # Synthetic data generator
│   ├── feature_engineering.py # Temporal lag feature builder
│   └── readiness_model.py # XGBoost training and evaluation
├── tests/                 # Unit tests for data leakage prevention
├── .gitignore
├── README.md
└── requirements.txt
```

## Limitations and Future Work
- Cold Start: A group model fallback is required for athletes with fewer than 30 days of data.
- No Wearables: Phase 1 relies on subjective check-in data. Integrating HRV data is expected to push R-squared accuracy from 0.77 to 0.90.
- Class Imbalance: Future extension to injury prediction will require SMOTE or weighted loss functions to handle rare event modeling.
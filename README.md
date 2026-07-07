# BioPerformance AI Readiness System

## Overview

The BioPerformance AI Readiness System is a machine learning pipeline designed to predict next-day athlete wellness. Fixed formulas and rule-based engines fail to capture the non-linear, highly individualized nature of athletic fatigue and recovery. This project implements **Approach B**: an XGBoost gradient boosting model trained on temporal lag features, paired with SHAP (SHapley Additive exPlanations) for per-prediction explainability.

The system transforms raw time-series data (sleep, fatigue, soreness, mood, training load) into a tabular feature matrix, learns non-linear interactions between training load and recovery, and outputs fully auditable predictions for synthesis into coaching insights.

Validated via a **two-stage process**: synthetic data with known ground-truth patterns, then real human physiology from the PMData dataset (16 athletes, 1,747 observation days). All code is standalone Python modules ready for Phase 2 integration.

## Results

| Model | Synthetic MAE | **PMData MAE** | Target |
|---|---|---|---|
| Fatigue | 1.42 | **0.54** | - |
| Soreness | 1.32 | **0.42** | - |
| Mood | 0.75 | **0.59** | - |
| Sleep Quality | 1.46 | **0.59** | - |
| *Option 1 - Hooper Composite* | *2.73* | ***1.36*** | *< 2.0* |

The core approach trains four separate XGBoost regressors - one per wellness item - each with its own group model (cold-start fallback) and individual athlete model. This validates Schliep & Schafer (2021): each wellness item has a unique temporal signature lost in composite scores. Per-item models average 0.54 MAE vs 1.36 for the composite.

## Architecture

The pipeline consists of three core components:

1. **Feature Engineering** (`src/feature_engineering.py`): Converts raw daily check-ins and training sessions into temporal lag features (t0, t1, t2), 7-day and 28-day rolling statistics, Z-scores, sleep coefficient of variation, Acute:Chronic Workload Ratio (ACWR), and days since last rest. Outputs 52 engineered columns per athlete.

2. **XGBoost Model** (`src/readiness_model.py`): A gradient boosting regression model that learns individual athlete patterns. Uses **TimeSeriesSplit** cross-validation with expanding windows to prevent data leakage. A **Group Model** (pooled across all athletes) serves as cold-start fallback for athletes with fewer than 14 days of data. An **Individual Model** activates once sufficient data exists. Supports Option 1 (composite Hooper Index) and Option 2 (separate models per wellness item).

3. **SHAP Explainability** (`src/shap_analysis.py`): Calculates per-feature attributions for every prediction using TreeExplainer. Generates beeswarm (global importance) and waterfall (local decomposition) plots. The top 3 SHAP drivers are extracted to provide actionable, transparent reasoning for coaching staff.

## Methodology

The implementation is grounded in established sports science literature:

- **[Rossi et al. (2022)](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2022.896928)**: Defines the temporal lag feature engineering blueprint - fatigue_t0, t1, t2, 7d/28d rolling means.
- **[Taber et al. (2024)](https://www.nature.com/articles/s41598-024-51658-8)**: Validates XGBoost architecture for collegiate athlete performance prediction (>90% accuracy). Informs PDP inclusion alongside SHAP.
- **[Schliep & Schafer (2021)](https://doi.org/10.1515/jqas-2020-0051)**: Informs treatment of multivariate ordinal wellness data - separate models per item outperform composite scores.
- **[Gabbett et al. (2016)](https://bjsm.bmj.com/content/50/5/273)**: Acute:Chronic Workload Ratio (ACWR) for injury risk monitoring and days-since-last-rest feature.

## Installation and Setup

### Prerequisites
- Python 3.12

### Installation
```bash
git clone <repository-url> && cd bioperformance
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Generate or Download Data
```bash
# Generate 90 days of synthetic athlete data for pipeline testing
python src/data_synthesis.py

# Or: auto-download real PMData (16 athletes, ~50MB) from OSF API
python src/load_pmdata.py
```

### 2. Build Feature Matrix
Transform raw data into temporal lag features and target variables:
```bash
# Default: synthetic data
python src/feature_engineering.py

# Or: use real PMData
python src/feature_engineering.py --dataset pmdata
```

### 3. Run Leakage Tests
```bash
python tests/test_leakage.py
```

### 4. Train and Evaluate Models
```bash
python src/readiness_model.py
```

### 5. SHAP Explainability
```bash
python src/shap_analysis.py
```

### 6. Jupyter Notebook
```bash
jupyter notebook shap_analysis.ipynb
```

## Project Structure
```
bioperformance/
├── data/
│   ├── raw/                        # Immutable original data
│   │   └── pmdata_raw/             # PMData participant folders (p01-p16)
│   ├── synthetic/                  # Generated fake data
│   └── processed/                  # Cleaned, ML-ready data
├── src/                            # All Python code
│   ├── data_synthesis.py           # Synthetic data generator
│   ├── load_pmdata.py              # PMData auto-downloader + parser
│   ├── feature_engineering.py      # Temporal lag feature builder
│   ├── readiness_model.py          # XGBoost training + dual-model fallback
│   └── shap_analysis.py            # SHAP explainability script
├── tests/
│   └── test_leakage.py             # 4 unit tests (3 leakage + 1 MAE assertion)
├── reports/
│   ├── Evaluation_Report.md        # Accuracy metrics, methodology, findings
│   ├── Comparison_Memo.md          # ML vs Rule Engine analysis
│   └── Research_Notes.md           # Annotated paper list with implementation impact
├── shap_analysis.ipynb             # Jupyter notebook: SHAP beeswarm + waterfall + PDP
├── .gitignore
├── README.md
└── requirements.txt
```

## Data Engineering

Follows **Cookiecutter Data Science** convention: `raw/` (immutable original data) to `synthetic/` (quarantined generated data) to `processed/` (ML-ready feature matrices).

The PMData loader (`src/load_pmdata.py`) auto-discovers participant folders via OSF API pagination and downloads only PMSys CSVs (wellness.csv, srpe.csv). Column mapping handles `effective_time_frame` to `date`, `duration_min * perceived_exertion` to `srpe`. Wellness and sRPE are merged using a **left join** to preserve rest days - critical because an inner merge was silently discarding 70% of wellness data (1,109 rows), biasing the model toward training-day patterns.

Feature engineering produces 52 columns per athlete: temporal lags (t0, t1, t2), rolling 7d/28d means, SDs, Z-scores, sleep CV, ACWR, days since last rest. All features are backward-looking only - no future leakage.

## Cold-Start Strategy

| Model | Scope | When Used |
|---|---|---|
| **Group Model** | Pooled from all athletes | <14 days history |
| **Individual Model** | Single athlete's data | >=14 days history |

After switching from inner merge to left merge, all 16 PMData athletes qualify for individual models (72-147 days each). The group model fallback still triggers for brand-new athletes with no history.

## Limitations

- **Phase 1 scope (no wearables)**: Subjective check-in data has a known ceiling of R-squared ~0.77. Integrating wearable HRV data (RMSSD, LF/HF ratio) is expected to push accuracy to R-squared ~0.90.
- **Ordinal data**: Fatigue, mood, soreness, sleep quality are rated 1-10. This is ordinal, not continuous. XGBoost handles it robustly but ordinal regression may give better calibration for extreme values.
- **Cold start**: New athletes have no personal baseline. The group model fallback covers the first 14 days. Few-shot learning could reduce this to 7 days.
- **Class imbalance (future injury extension)**: If extended to injury prediction, SMOTE or weighted loss will be mandatory. A model that always predicts "no injury" achieves 95% apparent accuracy while being useless.

## Future Research

- **Phase 2 wearables**: HRV (RMSSD, LF/HF), sleep stages, biomarkers - expected to push R-squared from ~0.77 to ~0.90.
- **Few-shot learning**: Can pre-training on population data reduce cold-start from 14 to 7 days?
- **Bayesian hierarchical models**: Schliep & Schafer (2021) approach - more principled for ordinal data, trade-off in complexity.
- **LSTM/Transformer**: End-to-end sequence modeling vs manual lag features. Trade-off in interpretability.

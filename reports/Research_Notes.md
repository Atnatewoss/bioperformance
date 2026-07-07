# Research Notes & Annotated Paper List

This document annotates the five core research papers driving the BioPerformance Track B architecture. It outlines the key findings of each paper and exactly how they influenced (or changed) the implementation approach.

## Paper 1: Rossi et al. (2022) -- Primary Methodology Guide

**Title:** Wellness Forecasting by External and Internal Workloads in Elite Soccer Players: A Machine Learning Approach  
**Journal:** Frontiers in Physiology

**Key Findings:** Demonstrated that Random Forest models using temporal lag features (t0, t1, t2) and rolling windows could accurately predict next-day wellness in Serie A players. Established that training load (sRPE) and its derived metrics (ACWR) are primary drivers of fatigue.

**What Changed Our Approach:** This paper dictated our entire feature engineering blueprint. Instead of feeding raw daily data into the model, we implemented Rossi's exact lag structure (fatigue_t0, fatigue_t1, fatigue_t2, 7d_mean) in feature_engineering.py. It also confirmed that time-based train/test splits are mandatory for this data type.

## Paper 2: Taber et al. (2024) -- Target Market Validation

**Title:** A Holistic Approach to Performance Prediction in Collegiate Athletics  
**Journal:** Scientific Reports (Nature)

**Key Findings:** Validated XGBoost as a superior architecture for NCAA athletes, achieving >90% accuracy. Found that weekly load and sleep consistency were top predictive features. Highlighted the use of Partial Dependence Plots (PDPs) for explainability.

**What Changed Our Approach:** This paper solidified our choice to use XGBRegressor over standard Random Forests or linear models. It also mandated the inclusion of PDP plots alongside SHAP in our explainability notebook to satisfy their validated methodology.

## Paper 3: Schliep & Schafer (2021) -- Critical Methodology Decision

**Title:** Distributed Lag Models for Cumulative Effects of Training on Multivariate Ordinal Wellness Data  
**Journal:** Journal of Quantitative Analysis in Sports

**Key Findings:** Analyzed two full MLS seasons and found that each wellness item (soreness, mood, etc.) has a completely different temporal lag signature. Soreness peaks on Day 2, while mood recovers by Day 1. Summing these items into a single composite score washes out these distinct signals.

**What Changed Our Approach:** This paper forced a pivot in our target variable strategy. While we started with Option 1 (Composite Hooper Index) to get the pipeline working, we implemented Option 2 (training 4 separate models for fatigue, soreness, mood, and sleep). Our benchmarks proved Option 2 was vastly more accurate, validating the paper's core thesis.

## Paper 4: Xu, Sun et al. (2025) -- Injury Prediction & Class Imbalance

**Title:** Construction and Application of a Model for Predicting Athletes Injury Risk Based on Machine Learning  
**Journal:** BMC Medical Informatics

**Key Findings:** Highlighted that ML injury prediction suffers from severe class imbalance (injuries are rare). Standard models will always predict "no injury" and appear highly accurate while being functionally useless. SMOTE or weighted loss functions are mandatory.

**What Changed Our Approach:** This paper defined the boundary for Phase 1 vs. Future Extensions. Because our Phase 1 dataset is restricted to subjective wellness and training load, we explicitly scoped out injury prediction. We documented SMOTE and scale_pos_weight as mandatory requirements for the future Phase 2 injury extension.

## Paper 5: Explainable AI in Sports Science (2025) -- SHAP Context

**Title:** Explainable AI in Sports Science: A Scoping Review  
**Journal:** Discover Artificial Intelligence (Springer)

**Key Findings:** SHAP (SHapley Additive exPlanations) dominates the XAI landscape in sports science. However, a critical gap exists: these explanations have not been validated by domain experts (coaches), leading to a disconnect between math and coach-friendly language.

**What Changed Our Approach:** This paper justified the entire explainability layer. We implemented shap.TreeExplainer to generate waterfall plots and extract the Top 3 feature drivers per prediction. This structured data is explicitly designed to bridge the gap identified by the paper -- serving as the structured input for the Claude LLM to translate into coach-friendly insights in Phase 2.

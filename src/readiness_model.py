import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from feature_engineering import build_feature_matrix, get_target_vector

# Drop columns during training to prevent data leakage
DROP_COLS = ['date', 'athlete_id', 'target_hooper_tomorrow', 
             'target_fatigue_tomorrow', 'target_soreness_tomorrow', 
             'target_mood_tomorrow', 'target_sleep_quality_tomorrow',
             'fatigue', 'soreness', 'mood', 'sleep_quality', 'srpe']

def train_group_model(df, target_col='target_hooper_tomorrow'):
    """Trains a single model on all athletes pooled together with TimeSeriesSplit CV."""
    print(f"\n--- Training Group Model for {target_col} (Cold Start Fallback) ---")

    df_sorted = df.sort_values('date').reset_index(drop=True)
    X = df_sorted.drop(columns=DROP_COLS, errors='ignore')
    y = df_sorted[target_col]

    # TimeSeriesSplit cross-validation (expanding window per brief Section 5.3)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes = []
    for train_idx, val_idx in tscv.split(X):
        X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
        y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X_cv_train, y_cv_train)
        cv_maes.append(mean_absolute_error(y_cv_val, model.predict(X_cv_val)))

    print(f"TimeSeriesSplit CV MAEs: {[f'{m:.2f}' for m in cv_maes]}")
    print(f"Average CV MAE: {np.mean(cv_maes):.2f} points")

    # Refit on full training data for final model
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model

def train_individual_model(df, athlete_id, target_col='target_hooper_tomorrow'):
    """Trains an individual model for a specific athlete if they have enough data."""
    athlete_df = df[df['athlete_id'] == athlete_id].copy()

    # Check if athlete has enough days of data
    if len(athlete_df) < 14:
        print(f"Athlete {athlete_id} only has {len(athlete_df)} days. Fallback to Group Model.")
        return None

    print(f"--- Training Individual Model for Athlete {athlete_id} ({target_col}) ---")
    athlete_df = athlete_df.sort_values('date').reset_index(drop=True)
    X = athlete_df.drop(columns=DROP_COLS, errors='ignore')
    y = athlete_df[target_col]

    split_idx = int(len(athlete_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"Athlete {athlete_id} Individual Model MAE: {mae:.2f} points")
    return model

def predict_readiness_with_fallback(athlete_id, athlete_history_days, group_model, individual_models, feature_row):
    """
    Demonstrates the Cold Start logic (Section 5.3).
    If < 14 days -> Use Group Model
    If >= 14 days -> Use Individual Model
    """
    if athlete_history_days < 14 or athlete_id not in individual_models or individual_models[athlete_id] is None:
        return group_model.predict(feature_row)[0]
    else:
        return individual_models[athlete_id].predict(feature_row)[0]

def main():
    feature_path = 'data/processed/features.csv'

    # 1. Load the processed features
    print(f"Loading processed features from: {feature_path}")
    df = pd.read_csv(feature_path, parse_dates=['date'])
    df_sorted = df.sort_values('date').reset_index(drop=True)

    # 80/20 chronological train/test split
    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    print(f"Train Set Rows: {len(train_df)} | Test Set Rows: {len(test_df)}")

    # --- Option 1: Hooper Index (Composite) ---
    print("\n" + "="*50 + "\nOPTION 1: Composite Hooper Index Model\n" + "="*50)
    group_model = train_group_model(train_df, 'target_hooper_tomorrow')

    individual_models = {}
    for athlete_id in train_df['athlete_id'].unique():
        ind_model = train_individual_model(train_df, athlete_id, 'target_hooper_tomorrow')
        if ind_model is not None:
            individual_models[athlete_id] = ind_model

    # Evaluation with fallback
    fallback_preds = []
    y_test_hooper = test_df['target_hooper_tomorrow'].values
    for _, row in test_df.iterrows():
        athlete_id = row['athlete_id']
        athlete_history_days = len(train_df[train_df['athlete_id'] == athlete_id])
        X_inst = pd.DataFrame([row]).drop(columns=DROP_COLS, errors='ignore')

        pred = predict_readiness_with_fallback(athlete_id, athlete_history_days, group_model, individual_models, X_inst)
        fallback_preds.append(pred)

    fallback_mae = mean_absolute_error(y_test_hooper, fallback_preds)
    print(f"\nOverall Option 1 Fallback MAE: {fallback_mae:.2f} points (Target: < 2.0)")

    # --- Option 2: Individual Items (Fatigue, Soreness, Mood, Sleep Quality) ---
    print("\n" + "="*50 + "\nOPTION 2: Individual Wellness Item Models\n" + "="*50)
    items = ['fatigue', 'soreness', 'mood', 'sleep_quality']

    for item in items:
        target_col = f'target_{item}_tomorrow'
        print(f"\n--- Model Set for: {item.upper()} ---")

        item_group_model = train_group_model(train_df, target_col)
        item_ind_models = {}
        for athlete_id in train_df['athlete_id'].unique():
            ind_model = train_individual_model(train_df, athlete_id, target_col)
            if ind_model is not None:
                item_ind_models[athlete_id] = ind_model

        # Fallback eval for item
        item_preds = []
        y_test_item = test_df[target_col].values
        for _, row in test_df.iterrows():
            athlete_id = row['athlete_id']
            athlete_history_days = len(train_df[train_df['athlete_id'] == athlete_id])
            X_inst = pd.DataFrame([row]).drop(columns=DROP_COLS, errors='ignore')

            pred = predict_readiness_with_fallback(athlete_id, athlete_history_days, item_group_model, item_ind_models, X_inst)
            item_preds.append(pred)

        item_mae = mean_absolute_error(y_test_item, item_preds)
        print(f"Overall Option 2 Fallback MAE for {item}: {item_mae:.2f} points")

if __name__ == "__main__":
    main()

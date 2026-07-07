import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from feature_engineering import build_feature_matrix, get_target_vector
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

SYNTHETIC_DATA = 'data/synthetic/synthetic_athlete_data.csv'

def test_no_target_in_features():
    print("Running test_no_target_in_features...")
    df = pd.read_csv(SYNTHETIC_DATA, parse_dates=['date'])

    # Split into checkins and sessions to prevent suffix collision on 'srpe'
    checkins = df[['date', 'athlete_id', 'fatigue', 'soreness', 'mood', 'sleep_quality']]
    sessions = df[['date', 'athlete_id', 'srpe']]

    feature_df = build_feature_matrix(athlete_id=1, checkins=checkins, sessions=sessions)

    assert 'target_hooper_tomorrow' not in feature_df.columns, "Target column found in features!"
    assert 'hooper_index' not in feature_df.columns, "Raw Hooper Index found in features!"
    print("[OK] PASSED: No target leakage in feature columns.")

def test_temporal_integrity():
    print("Running test_temporal_integrity...")
    df = pd.read_csv(SYNTHETIC_DATA, parse_dates=['date'])

    checkins = df[['date', 'athlete_id', 'fatigue', 'soreness', 'mood', 'sleep_quality']]
    sessions = df[['date', 'athlete_id', 'srpe']]

    feature_df = build_feature_matrix(athlete_id=1, checkins=checkins, sessions=sessions)

    # Day 2's 'fatigue_t1' (yesterday) should equal Day 1's 'fatigue_t0' (today)
    day1_fatigue = feature_df.loc[0, 'fatigue_t0']
    day2_yesterday_fatigue = feature_df.loc[1, 'fatigue_t1']

    assert day1_fatigue == day2_yesterday_fatigue, "Temporal lag is misaligned!"
    print("[OK] PASSED: Temporal lags are strictly backward-looking.")

def test_target_shift():
    print("Running test_target_shift...")
    df = pd.read_csv(SYNTHETIC_DATA, parse_dates=['date'])

    checkins = df[['date', 'athlete_id', 'fatigue', 'soreness', 'mood', 'sleep_quality']]
    sessions = df[['date', 'athlete_id', 'srpe']]

    feature_df = build_feature_matrix(athlete_id=1, checkins=checkins, sessions=sessions)

    feature_df['actual_tomorrow_fatigue'] = feature_df['fatigue'].shift(-1)
    y = get_target_vector(feature_df, target_col='fatigue', lag=1)

    assert y.iloc[0] == feature_df['actual_tomorrow_fatigue'].iloc[0], "Target vector is not shifted correctly!"
    print("[OK] PASSED: Target vector correctly represents next-day data.")

def test_group_model_mae_below_threshold():
    feature_path = 'data/processed/features.csv'
    print(f"Running test_group_model_mae_below_threshold...")
    df = pd.read_csv(feature_path, parse_dates=['date'])
    df_sorted = df.sort_values('date').reset_index(drop=True)

    DROP_COLS = ['date', 'athlete_id', 'target_hooper_tomorrow',
                 'target_fatigue_tomorrow', 'target_soreness_tomorrow',
                 'target_mood_tomorrow', 'target_sleep_quality_tomorrow',
                 'fatigue', 'soreness', 'mood', 'sleep_quality', 'srpe']

    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    X_train = train_df.drop(columns=DROP_COLS, errors='ignore')
    y_train = train_df['target_hooper_tomorrow']
    X_test = test_df.drop(columns=DROP_COLS, errors='ignore')
    y_test = test_df['target_hooper_tomorrow']

    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print(f"Group Model MAE: {mae:.2f} points")
    assert mae < 2.0, f"MAE {mae:.2f} exceeds target of 2.0!"
    print("[OK] PASSED: Group model MAE is below 2.0 threshold.")

if __name__ == "__main__":
    test_no_target_in_features()
    test_temporal_integrity()
    test_target_shift()
    test_group_model_mae_below_threshold()
    print("\nAll leakage tests passed successfully!")

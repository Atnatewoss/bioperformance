import pandas as pd
import sys
import os

# Add src to path to import our functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from feature_engineering import build_feature_matrix, get_target_vector

def test_no_target_in_features():
    """Ensure the target variable is not accidentally included in the feature matrix."""
    print("Running test_no_target_in_features...")
    df = pd.read_csv('data/synthetic_athlete_data.csv', parse_dates=['date'])
    feature_df = build_feature_matrix(df)
    
    # Check that none of the columns are the target
    assert 'target_hooper_tomorrow' not in feature_df.columns, "Target column found in features!"
    assert 'hooper_index' not in feature_df.columns, "Raw Hooper Index found in features!"
    print("✅ PASSED: No target leakage in feature columns.")

def test_temporal_integrity():
    """Ensure features for Day N do not contain data from Day N+1."""
    print("Running test_temporal_integrity...")
    df = pd.read_csv('data/synthetic_athlete_data.csv', parse_dates=['date'])
    feature_df = build_feature_matrix(df)
    
    # Get athlete 1's data
    athlete_1 = feature_df[feature_df['athlete_id'] == 1].sort_values('date').reset_index(drop=True)
    
    # Day 2's 'fatigue_t1' (yesterday) should equal Day 1's 'fatigue_t0' (today)
    day1_fatigue = athlete_1.loc[0, 'fatigue_t0']
    day2_yesterday_fatigue = athlete_1.loc[1, 'fatigue_t1']
    
    assert day1_fatigue == day2_yesterday_fatigue, "Temporal lag is misaligned!"
    print("✅ PASSED: Temporal lags are strictly backward-looking.")

def test_target_shift():
    """Ensure the target vector correctly points to the NEXT day."""
    print("Running test_target_shift...")
    df = pd.read_csv('data/synthetic_athlete_data.csv', parse_dates=['date'])
    feature_df = build_feature_matrix(df)
    
    # Calculate what tomorrow's actual fatigue should be
    feature_df['actual_tomorrow_fatigue'] = feature_df.groupby('athlete_id')['fatigue'].shift(-1)
    
    # Get our target vector (using fatigue instead of hooper for this specific test)
    y = get_target_vector(feature_df, target_col='fatigue', lag=1)
    
    # Compare the first row
    assert y.iloc[0] == feature_df['actual_tomorrow_fatigue'].iloc[0], "Target vector is not shifted correctly!"
    print("✅ PASSED: Target vector correctly represents next-day data.")

if __name__ == "__main__":
    test_no_target_in_features()
    test_temporal_integrity()
    test_target_shift()
    print("\nAll leakage tests passed successfully!")
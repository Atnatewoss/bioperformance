import pandas as pd
import numpy as np

def build_feature_matrix(df):
    """
    Transforms raw time-series data into a tabular feature matrix for XGBoost.
    Follows Rossi 2022 temporal lag methodology.
    """
    df = df.sort_values(['athlete_id', 'date']).copy()
    
    # 1. Temporal Lag Features (t0, t1, t2)
    lag_fields = ['fatigue', 'soreness', 'mood', 'sleep_quality', 'srpe']
    for field in lag_fields:
        # Explicitly create t0 so it matches our schema
        df[f'{field}_t0'] = df[field]
        # t1 is yesterday
        df[f'{field}_t1'] = df.groupby('athlete_id')[field].shift(1)
        # t2 is two days ago
        df[f'{field}_t2'] = df.groupby('athlete_id')[field].shift(2)
        
    # 2. Rolling Statistics (7-day and 28-day)
    for field in ['fatigue', 'soreness', 'srpe']:
        # 7-day rolling mean (includes today)
        df[f'{field}_7d_mean'] = df.groupby('athlete_id')[field].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        # 28-day baseline
        df[f'{field}_28d_mean'] = df.groupby('athlete_id')[field].transform(
            lambda x: x.rolling(window=28, min_periods=7).mean()
        )
        df[f'{field}_28d_sd'] = df.groupby('athlete_id')[field].transform(
            lambda x: x.rolling(window=28, min_periods=7).std()
        )
        
        # Z-score (Delta vs 28-day baseline)
        # (Today - 28d_mean) / 28d_sd
        df[f'{field}_zscore_28d'] = (df[field] - df[f'{field}_28d_mean']) / df[f'{field}_28d_sd']
        
    # 3. ACWR (Acute:Chronic Workload Ratio)
    # Acute Load = Sum of sRPE over the last 7 days
    # Chronic Load = Average weekly sRPE over the last 28 days (Sum of 28 days / 4)
    # ACWR = Acute Load / Chronic Load
    df['acute_load_7d'] = df.groupby('athlete_id')['srpe'].transform(
        lambda x: x.rolling(window=7, min_periods=1).sum()
    )
    df['chronic_load_28d'] = df.groupby('athlete_id')['srpe'].transform(
        lambda x: x.rolling(window=28, min_periods=7).sum()
    ) / 4.0
    
    df['acwr_ratio'] = df['acute_load_7d'] / df['chronic_load_28d']
    
    # Handle infinities/NaNs from division by zero
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.bfill(inplace=True) # Backfill early NaNs
    
    return df

def get_target_vector(df, target_col='hooper_index', lag=1):
    """
    Returns next-day target variable.
    lag=1 means we want to predict tomorrow's score using today's features.
    """
    df = df.copy()
    
    # If target is Hooper Index (sum of items), calculate it first
    if target_col == 'hooper_index':
        df['hooper_index'] = df['fatigue'] + df['soreness'] + df['mood'] + df['sleep_quality']
    
    # Shift UP by 1: Row for Day 1 gets the target value of Day 2
    df['target_next_day'] = df.groupby('athlete_id')[target_col].shift(-lag)
    
    return df['target_next_day']

if __name__ == "__main__":
    # Test the pipeline
    raw_data = pd.read_csv('data/synthetic_athlete_data.csv', parse_dates=['date'])
    
    # Build Features
    feature_df = build_feature_matrix(raw_data)
    
    # Get Target (Option 1: Hooper Index for now, to get pipeline working)
    y = get_target_vector(feature_df, target_col='hooper_index', lag=1)
    
    # Combine for final dataset
    feature_df['target_hooper_tomorrow'] = y
    
    # Drop rows where target is NaN (the very last day for each athlete has no "tomorrow")
    final_df = feature_df.dropna(subset=['target_hooper_tomorrow'])
    
    print("Feature Matrix Shape:", final_df.shape)
    print("Sample Features:")
    print(final_df[['date', 'fatigue_t0', 'srpe_7d_mean', 'acwr_ratio', 'target_hooper_tomorrow']].head(10))
    
    final_df.to_csv('data/processed_features.csv', index=False)
    print("\nSaved processed features to data/processed_features.csv")
import os
import pandas as pd
import numpy as np

def build_feature_matrix(athlete_id, checkins, sessions, pain=None, biomarkers=None, baselines=None):
    """
    Transforms raw time-series data into a tabular feature matrix for XGBoost.
    Follows Rossi 2022 temporal lag methodology.
    Signature matches BioPerformance brief Section 5.1.
    """
    # Filter data for the specific athlete
    athlete_checkins = checkins[checkins['athlete_id'] == athlete_id].copy()
    athlete_sessions = sessions[sessions['athlete_id'] == athlete_id].copy()

    # Merge checkins and sessions
    df = pd.merge(athlete_checkins, athlete_sessions[['date', 'athlete_id', 'srpe']], on=['date', 'athlete_id'], how='left')
    df['srpe'] = df['srpe'].fillna(0)  # Rest days have 0 load
    df = df.sort_values('date').copy()

    # 1. Temporal Lag Features (t0, t1, t2) - No groupby needed since df is 1 athlete
    lag_fields = ['fatigue', 'soreness', 'mood', 'sleep_quality', 'srpe']
    for field in lag_fields:
        df[f'{field}_t0'] = df[field]
        df[f'{field}_t1'] = df[field].shift(1)
        df[f'{field}_t2'] = df[field].shift(2)

    # 2. Rolling Statistics (7-day and 28-day)
    for field in ['fatigue', 'soreness', 'mood', 'sleep_quality', 'srpe']:
        df[f'{field}_7d_mean'] = df[field].rolling(window=7, min_periods=1).mean()
        df[f'{field}_28d_mean'] = df[field].rolling(window=28, min_periods=7).mean()
        df[f'{field}_28d_sd'] = df[field].rolling(window=28, min_periods=7).std()
        df[f'{field}_zscore_28d'] = (df[field] - df[f'{field}_28d_mean']) / df[f'{field}_28d_sd']

    # Sleep consistency (coefficient of variation) — Taber 2024 found this a top predictor
    df['sleep_cv'] = (df['sleep_quality_28d_sd'] / df['sleep_quality_28d_mean']) * 100

    # 3. ACWR (Acute:Chronic Workload Ratio)
    df['acute_load_7d'] = df['srpe'].rolling(window=7, min_periods=1).sum()
    df['chronic_load_28d'] = df['srpe'].rolling(window=28, min_periods=7).sum() / 4.0
    df['acwr_ratio'] = df['acute_load_7d'] / df['chronic_load_28d']

    # 4. Days Since Last Rest (Gabbett 2016 core recovery metric)
    rest_groups = (df['srpe'] == 0).cumsum()
    df['days_since_last_rest'] = df.groupby(rest_groups).cumcount()

    # Handle infinities/NaNs from division by zero
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.bfill(inplace=True)

    return df

def get_target_vector(df, target_col='hooper_index', lag=1):
    """
    Returns next-day target variable.
    lag=1 means we want to predict tomorrow's score using today's features.
    """
    df = df.copy()
    if target_col == 'hooper_index':
        df['hooper_index'] = df['fatigue'] + df['soreness'] + df['mood'] + df['sleep_quality']

    df['target_next_day'] = df[target_col].shift(-lag)
    return df['target_next_day']

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Build feature matrix from raw athlete data')
    parser.add_argument('--dataset', default='synthetic', choices=['synthetic', 'pmdata'],
                        help="Which dataset to process (default: synthetic)")
    parser.add_argument('path', nargs='?', default=None,
                        help="Direct path to CSV (overrides --dataset)")
    args = parser.parse_args()

    if args.path:
        data_path = args.path
    else:
        data_path = {
            'synthetic': 'data/synthetic/synthetic_athlete_data.csv',
            'pmdata': 'data/processed/pmdata_cleaned.csv'
        }[args.dataset]

    print(f"Loading data from: {data_path}")
    raw_data = pd.read_csv(data_path, parse_dates=['date'])

    # Split raw_data into checkins and sessions
    checkins = raw_data[['date', 'athlete_id', 'fatigue', 'soreness', 'mood', 'sleep_quality']].copy()
    sessions = raw_data[['date', 'athlete_id', 'srpe']].copy()

    # Run build_feature_matrix for each unique athlete and combine
    athlete_dfs = []
    for athlete_id in raw_data['athlete_id'].unique():
        athlete_features = build_feature_matrix(athlete_id, checkins, sessions)

        # Option 1 Target: Hooper Index
        athlete_features['target_hooper_tomorrow'] = get_target_vector(athlete_features, target_col='hooper_index', lag=1)

        # Option 2 Targets: Individual Items
        for item in ['fatigue', 'soreness', 'mood', 'sleep_quality']:
            athlete_features[f'target_{item}_tomorrow'] = get_target_vector(athlete_features, target_col=item, lag=1)

        # Drop rows where target is NaN (the very last day has no "tomorrow")
        athlete_features = athlete_features.dropna(subset=['target_hooper_tomorrow'])
        athlete_dfs.append(athlete_features)

    final_df = pd.concat(athlete_dfs, ignore_index=True)
    output_path = 'data/processed/features.csv'
    print("Feature Matrix Shape:", final_df.shape)
    os.makedirs('data/processed', exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"\nSaved processed features to {output_path}")

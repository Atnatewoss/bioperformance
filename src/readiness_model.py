import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

def train_and_evaluate():
    # 1. Load the processed data
    print("Loading processed features...")
    df = pd.read_csv('data/processed_features.csv', parse_dates=['date'])
    
    # 2. Define Features (X) and Target (y)
    # We drop the target, the date, and the raw check-in fields (because we only want to use lags/rolling stats)
    # We also drop athlete_id for the Group Model (cold start scenario)
    drop_cols = ['date', 'athlete_id', 'target_hooper_tomorrow', 
                 'fatigue', 'soreness', 'mood', 'sleep_quality', 'srpe']
    
    X = df.drop(columns=drop_cols)
    y = df['target_hooper_tomorrow']
    
    # 3. Time-Based Train/Test Split
    # Sort by date to ensure strict chronological order
    df_sorted = df.sort_values('date').reset_index(drop=True)
    X_sorted = df_sorted.drop(columns=drop_cols)
    y_sorted = df_sorted['target_hooper_tomorrow']
    
    # 80% train, 20% test (strictly time-ordered)
    split_idx = int(len(df_sorted) * 0.8)
    
    X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
    y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]
    
    print(f"Train set: {len(X_train)} rows (Dates: {df_sorted['date'].iloc[0].date()} to {df_sorted['date'].iloc[split_idx-1].date()})")
    print(f"Test set:  {len(X_test)} rows (Dates: {df_sorted['date'].iloc[split_idx].date()} to {df_sorted['date'].iloc[-1].date()})")
    
    # 4. Train XGBoost Model
    print("\nTraining XGBoost model...")
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"\n--- RESULTS ---")
    print(f"MAE: {mae:.2f} Hooper Index points")
    if mae < 2.0:
        print("SUCCESS: MAE is below the 2.0 target!")
    else:
        print("WARNING: MAE is above 2.0. We may need to tune hyperparameters or improve features.")
        
    # 6. Feature Importance (Sanity Check)
    print("\n--- TOP 5 FEATURES ---")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(5).to_string(index=False))
    
    return model, X_train

if __name__ == "__main__":
    model, X_train = train_and_evaluate()
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

def run_shap_analysis():
    # 1. Load and prepare data (Same as readiness_model.py)
    print("Loading data...")
    df = pd.read_csv('data/processed_features.csv', parse_dates=['date'])
    
    drop_cols = ['date', 'athlete_id', 'target_hooper_tomorrow', 
                 'fatigue', 'soreness', 'mood', 'sleep_quality', 'srpe']
    
    df_sorted = df.sort_values('date').reset_index(drop=True)
    X = df_sorted.drop(columns=drop_cols)
    y = df_sorted['target_hooper_tomorrow']
    
    split_idx = int(len(df_sorted) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 2. Train Model
    print("Training XGBoost model for SHAP...")
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # 3. Compute SHAP values
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    # We use the newer shap.Explanation API for better plotting
    shap_values = explainer(X_test)
    
    # 4. Global Feature Importance (Beeswarm Plot)
    print("Generating global feature importance plot...")
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig('data/shap_beeswarm_global.png')
    print("Saved global plot to data/shap_beeswarm_global.png")
    
    # 5. Local Explainability (Waterfall Plot for a single prediction)
    # Let's look at the first prediction in the test set
    instance_idx = 0
    print(f"\n--- Analyzing Prediction for Test Row {instance_idx} ---")
    print(f"Actual Target (Tomorrow's Hooper): {y_test.iloc[instance_idx]}")
    print(f"Model Prediction: {model.predict(X_test.iloc[[instance_idx]])[0]:.2f}")
    
    plt.figure()
    shap.plots.waterfall(shap_values[instance_idx], show=False)
    plt.tight_layout()
    plt.savefig('data/shap_waterfall_local.png')
    print("Saved local waterfall plot to data/shap_waterfall_local.png")
    
    # 6. Extract Top 3 Drivers for LLM Synthesis
    print("\n--- Top 3 SHAP Drivers for this Prediction ---")
    
    instance_shap = shap_values[instance_idx]
    
    # Create a dataframe of features and their SHAP values
    shap_df = pd.DataFrame({
        'feature_value': instance_shap.data,
        'shap_value': instance_shap.values
    }, index=X_test.columns) # Use feature names as index
    
    # Sort by absolute SHAP value to find the strongest pushers
    shap_df['abs_shap'] = np.abs(shap_df['shap_value'])
    top_drivers = shap_df.sort_values('abs_shap', ascending=False).head(3)
    
    for feature_name, row in top_drivers.iterrows():
        direction = "increased" if row['shap_value'] > 0 else "decreased"
        print(f"- {feature_name} (value: {row['feature_value']:.1f}) {direction} readiness by {abs(row['shap_value']):.2f} points")

if __name__ == "__main__":
    run_shap_analysis()
    
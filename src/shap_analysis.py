import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import json
import os

# Drop columns during training to prevent data leakage
DROP_COLS = ['date', 'athlete_id', 'target_hooper_tomorrow',
             'target_fatigue_tomorrow', 'target_soreness_tomorrow',
             'target_mood_tomorrow', 'target_sleep_quality_tomorrow',
             'fatigue', 'soreness', 'mood', 'sleep_quality', 'srpe']

ITEMS = ['fatigue', 'soreness', 'mood', 'sleep_quality']

MODEL_KWARGS = dict(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)


def train_item_model(df, item):
    """Trains one per-item XGBoost model with an 80/20 chronological split.

    Returns the model, X_test, the test target, and the number of training
    rows available per athlete (used as days_of_history downstream).
    """
    df_sorted = df.sort_values('date').reset_index(drop=True)
    X = df_sorted.drop(columns=DROP_COLS, errors='ignore')
    y = df_sorted[f'target_{item}_tomorrow']

    split_idx = int(len(df_sorted) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBRegressor(**MODEL_KWARGS)
    model.fit(X_train, y_train)

    train_df = df_sorted.iloc[:split_idx]
    history_by_athlete = train_df['athlete_id'].value_counts().to_dict()
    return model, X_test, y_test, history_by_athlete


def _clean_for_json(values):
    """Replaces NaN/inf with None so the dict serializes to strict JSON."""
    return {k: (None if pd.isna(v) else float(v)) for k, v in values.items()}


def run_shap_analysis():
    feature_path = 'data/processed/features.csv'
    print(f"Loading data from: {feature_path}")
    df = pd.read_csv(feature_path, parse_dates=['date'])

    # Preserve per-row athlete/date labels aligned to X_test rows.
    df_sorted = df.sort_values('date').reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    test_meta = df_sorted.iloc[split_idx:][['date', 'athlete_id']].reset_index(drop=True)

    os.makedirs('data/processed', exist_ok=True)

    # Per-item SHAP (Option 2 production scheme): one model + one SHAP run per
    # wellness item. The full 52-feature SHAP dict is exported per test row so
    # the PLN layer (pln_facts.shap_drivers_to_facts) can pick its own top-3.
    sample_entries = []
    for item in ITEMS:
        print(f"\n--- Item model: {item.upper()} ---")
        model, X_test, y_test, history_by_athlete = train_item_model(df, item)

        mae = np.mean(np.abs(model.predict(X_test) - y_test))
        print(f"Test MAE: {mae:.2f} points")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)

        # Per-item beeswarm (global importance for this item's model).
        plt.figure()
        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        plot_path = f'data/processed/shap_beeswarm_{item}.png'
        plt.savefig(plot_path)
        print(f"Saved {item} beeswarm to {plot_path}")

        preds = model.predict(X_test)
        shap_matrix = shap_values.values
        for idx in range(len(X_test)):
            athlete_id = int(test_meta.loc[idx, 'athlete_id'])
            row_shap = {
                col: shap_matrix[idx, j]
                for j, col in enumerate(X_test.columns)
            }
            sample_entries.append({
                "item": item,
                "athlete_id": athlete_id,
                "date": str(test_meta.loc[idx, 'date'].date()),
                "prediction": float(preds[idx]),
                "actual": float(y_test.iloc[idx]),
                "days_of_history": int(history_by_athlete.get(athlete_id, 0)),
                "shap_values": _clean_for_json(row_shap),
            })

    out_path = 'data/processed/shap_per_item.json'
    with open(out_path, 'w') as f:
        json.dump(sample_entries, f, indent=2)
    print(f"\nSaved {len(sample_entries)} per-item SHAP rows to {out_path}")


if __name__ == "__main__":
    run_shap_analysis()

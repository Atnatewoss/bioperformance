import pandas as pd
import numpy as np
from datetime import timedelta

def generate_athlete_data(athlete_id, start_date, days=90):
    """Generates synthetic daily check-in and training data for one athlete."""
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    
    # Base wellness scores (1-10 scale)
    base_fatigue = np.random.normal(5, 1.5, days)
    base_soreness = np.random.normal(4, 1.5, days)
    base_mood = np.random.normal(6, 1.0, days)
    base_sleep_quality = np.random.normal(6, 1.5, days)
    
    # Training Load (sRPE) - Random sessions, some rest days
    srpe = np.random.choice([0, 100, 200, 300, 400, 500], size=days, p=[0.2, 0.1, 0.2, 0.2, 0.2, 0.1])
    
    # Introduce a relationship: High sRPE yesterday -> Higher fatigue today
    # We shift sRPE by 1 to represent yesterday's load affecting today's check-in
    fatigue = base_fatigue + (np.roll(srpe, 1) / 200)
    soreness = base_soreness + (np.roll(srpe, 1) / 150)
    
    # Clip to 1-10 scale
    fatigue = np.clip(fatigue, 1, 10)
    soreness = np.clip(soreness, 1, 10)
    
    df = pd.DataFrame({
        'date': dates,
        'athlete_id': athlete_id,
        'fatigue': fatigue,
        'soreness': soreness,
        'mood': np.clip(base_mood, 1, 10),
        'sleep_quality': np.clip(base_sleep_quality, 1, 10),
        'srpe': srpe
    })
    return df

if __name__ == "__main__":
    # Generate data for 5 athletes
    all_data = []
    for i in range(1, 6):
        # Stagger start dates slightly so they aren't identical
        start = pd.Timestamp('2026-03-01') + timedelta(days=i)
        all_data.append(generate_athlete_data(athlete_id=i, start_date=start, days=90))
    
    df = pd.concat(all_data, ignore_index=True)
    df.to_csv('data/synthetic_athlete_data.csv', index=False)
    print("Generated synthetic data for 5 athletes. Saved to data/synthetic_athlete_data.csv")
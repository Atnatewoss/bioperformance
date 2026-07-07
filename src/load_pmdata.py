import pandas as pd
import os
import urllib.request
import json

API_BASE = 'https://api.osf.io/v2/nodes/vx4bk/files/osfstorage/'

def get_json(url):
    req = urllib.request.Request(url)
    req.add_header('Accept', 'application/json')
    return json.loads(urllib.request.urlopen(req).read())

def discover_participant_folders():
    """Finds all pXX folder IDs by walking all pages of the pmdata folder."""
    folder_ids = {}
    pmdata_folder_id = '5e99d05ef135350590d5316d'
    url = API_BASE + pmdata_folder_id + '/'

    while url:
        data = get_json(url)
        for item in data['data']:
            name = item['attributes']['name']
            if name.startswith('p') and len(name) == 3 and item['attributes']['kind'] == 'folder':
                p_num = int(name[1:])
                folder_ids[p_num] = item['id']
        url = data['links'].get('next')

    return folder_ids

def download_pmdata_files(raw_dir='data/raw/pmdata_raw'):
    """Downloads only the PMSys files (wellness.csv, srpe.csv) for each athlete via OSF API.

    We do NOT download the 1.4GB zip from OSF. That zip contains Fitbit wearable data
    (HR, sleep stages), food images, and Google Forms responses — all of which belong
    to Phase 2 per the brief Section 6 ("No wearables yet"). Phase 1 uses only subjective
    check-ins + training load. The OSF API lets us fetch exactly the CSVs we need.
    """
    os.makedirs(raw_dir, exist_ok=True)

    # Check if already downloaded
    existing = [d for d in os.listdir(raw_dir) if d.startswith('p')]
    if len(existing) >= 9:
        print(f"PMData already downloaded ({len(existing)} participants).")
        return

    p_folder_ids = discover_participant_folders()
    print(f"Found {len(p_folder_ids)} participant folders: {sorted(p_folder_ids.keys())}")

    for athlete_num in sorted(p_folder_ids.keys()):
        folder_id = p_folder_ids[athlete_num]
        p_url = API_BASE + folder_id + '/'
        p_data = get_json(p_url)

        # Find the pmsys folder
        pmsys_id = None
        for item in p_data['data']:
            if item['attributes']['name'] == 'pmsys' and item['attributes']['kind'] == 'folder':
                pmsys_id = item['id']
                break

        if pmsys_id is None:
            print(f"Athlete p{athlete_num:02d}: no pmsys folder, skipping")
            continue

        # List files in pmsys
        pmsys_data = get_json(API_BASE + pmsys_id + '/')
        file_ids = {}
        for item in pmsys_data['data']:
            if item['attributes']['kind'] == 'file':
                file_ids[item['attributes']['name']] = item['id']

        # Create athlete directory
        athlete_dir = os.path.join(raw_dir, f"p{athlete_num:02d}")
        pmsys_dir = os.path.join(athlete_dir, 'pmsys')
        os.makedirs(pmsys_dir, exist_ok=True)

        for fname in ['wellness.csv', 'srpe.csv']:
            if fname not in file_ids:
                print(f"Athlete p{athlete_num:02d}: missing {fname}, skipping")
                continue

            file_id = file_ids[fname]
            dl_url = f"https://files.osf.io/v1/resources/vx4bk/providers/osfstorage/{file_id}"
            dst_path = os.path.join(pmsys_dir, fname)
            print(f"Downloading p{athlete_num:02d}/{fname}...")
            urllib.request.urlretrieve(dl_url, dst_path)

    print(f"Download complete. Files saved to {raw_dir}/")

def load_pmdata(raw_dir='data/raw/pmdata_raw'):
    """Loads PMData dataset, extracts wellness and sRPE, and formats for feature_engineering.py pipeline."""
    download_pmdata_files(raw_dir)

    all_athletes = []
    print(f"Loading athlete data from {raw_dir}...")

    for athlete_num in range(1, 17):
        pmsys_path = os.path.join(raw_dir, f"p{athlete_num:02d}", 'pmsys')
        wellness_file = os.path.join(pmsys_path, 'wellness.csv')
        srpe_file = os.path.join(pmsys_path, 'srpe.csv')

        if not os.path.exists(wellness_file) or not os.path.exists(srpe_file):
            continue

        try:
            # Normalize timestamps to date-only for merge.
            # Wellness check-ins are logged in the morning (7-10am) and sRPE sessions
            # in the evening (5-10pm). Raw timestamps never match, but they refer to
            # the same training day. Per brief Section 5.1 ("one row per day") and
            # Rossi (2022) / Taber (2024) daily-row methodology, we extract just the
            # date component to combine morning wellness + evening load into one row.
            wellness = pd.read_csv(wellness_file)
            wellness = wellness[['effective_time_frame', 'fatigue', 'mood', 'soreness', 'sleep_quality']]
            wellness['date'] = pd.to_datetime(wellness['effective_time_frame']).dt.date
            wellness.drop(columns=['effective_time_frame'], inplace=True)

            srpe = pd.read_csv(srpe_file)
            srpe['srpe'] = srpe['duration_min'] * srpe['perceived_exertion']
            srpe['date'] = pd.to_datetime(srpe['end_date_time']).dt.date
            daily_srpe = srpe.groupby('date')['srpe'].sum().reset_index()

            athlete_df = pd.merge(wellness, daily_srpe, on='date', how='left')
            athlete_df['srpe'] = athlete_df['srpe'].fillna(0)
            athlete_df['athlete_id'] = athlete_num

            all_athletes.append(athlete_df)
            print(f"Loaded p{athlete_num:02d}: {len(athlete_df)} rows")
        except Exception as e:
            print(f"Skipping athlete {athlete_num} due to error: {e}")

    if not all_athletes:
        print("No athlete data loaded. Check the data directory structure.")
        return None

    final_df = pd.concat(all_athletes, ignore_index=True)
    final_df['date'] = pd.to_datetime(final_df['date'])

    final_df.to_csv('data/processed/pmdata_cleaned.csv', index=False, date_format='%Y-%m-%d')
    print(f"\nLoaded PMData for {final_df['athlete_id'].nunique()} athletes ({len(final_df)} rows). Saved to data/processed/pmdata_cleaned.csv")
    return final_df

if __name__ == "__main__":
    df = load_pmdata()
    if df is not None:
        print("\nPreview of cleaned PMData:")
        print(df.head())

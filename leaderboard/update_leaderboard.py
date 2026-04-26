import os
import json
import pandas as pd
from evaluate import evaluate

# 1. Setup absolute paths to prevent FileNotFoundError on GitHub Actions 
# Get the directory where update_leaderboard.py is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root
BASE_DIR = os.path.dirname(CURRENT_DIR)

# Define paths relative to the project root [cite: 131, 285]
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'submissions')
GROUND_TRUTH = os.path.join(BASE_DIR, 'data', 'test_motion_data.csv')
LEADERBOARD_FILE = os.path.join(BASE_DIR, 'leaderboard.json')

def update():
    results = []

    # Ensure the submissions directory exists 
    if not os.path.exists(SUBMISSIONS_DIR):
        print(f"Error: {SUBMISSIONS_DIR} does not exist.")
        return

    # Iterate through all submission files [cite: 130]
    for fname in os.listdir(SUBMISSIONS_DIR):
        if not fname.endswith(".csv"):
            continue

        team_name = fname.replace("_submission.csv", "")
        path = os.path.join(SUBMISSIONS_DIR, fname)

        try:
            # Call the evaluate function from evaluate.py [cite: 109, 130]
            metrics = evaluate(path, GROUND_TRUTH)
            results.append({
                "team": team_name,
                "accuracy": metrics["accuracy"],
                "f1_weighted": metrics["f1_weighted"]
            })
            print(f"✓ Scored: {team_name}")
        except Exception as e:
            print(f"X Failed for {team_name}: {e}")

    # Sort by F1 Score (primary) then Accuracy (secondary) [cite: 139, 140]
    results.sort(key=lambda x: (x['f1_weighted'], x['accuracy']), reverse=True)

    # Add ranking numbers [cite: 143]
    for i, r in enumerate(results):
        r["rank"] = i + 1

    # Write the results to leaderboard.json in the project root [cite: 131, 285]
    with open(LEADERBOARD_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Successfully updated {LEADERBOARD_FILE}")

if __name__ == "__main__":
    update()
"""
test_automation_model.py
========================
Run this to generate a test submission with KNOWN accuracy.
Use it to verify:
  1. update_leaderboard.py scores correctly
  2. leaderboard.json updates with the right F1/accuracy
  3. GitHub Actions triggers and reflects the score on the live page
 
USAGE:
    python test_automation_model.py --mode perfect   # ~100% accuracy
    python test_automation_model.py --mode good      # ~80% accuracy  
    python test_automation_model.py --mode baseline  # ~74% accuracy (matches baseline)
    python test_automation_model.py --mode bad       # ~40% accuracy
    python test_automation_model.py --mode random    # pure random predictions
 
The team name in the output file will reflect the mode so you can see
multiple entries ranked correctly on the leaderboard.
"""
 
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
 
BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE, "data", "train_motion_data.csv")
TEST_PATH  = os.path.join(BASE, "data", "test_motion_data_nolabels.csv")
SUBMISSIONS_DIR = os.path.join(BASE, "submissions")
 
CLASSES = ["Normal", "Aggressive", "Drowsy"]
 
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    return train, test
 
def get_real_predictions(train, test):
    """Train a proper Random Forest and return real predictions."""
    le = LabelEncoder()
    train["Class"] = le.fit_transform(train["Class"])
    X_train = train.drop(columns=["Class"])
    y_train = train["Class"]
    X_test  = test.copy()
 
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return le.inverse_transform(preds)
 
def corrupt_predictions(pred_labels, error_rate):
    """
    Randomly flip a fraction of predictions to wrong classes.
    error_rate=0.0 -> perfect, error_rate=0.6 -> 60% wrong
    """
    rng = np.random.default_rng(seed=99)
    result = pred_labels.copy()
    n = len(result)
    n_corrupt = int(n * error_rate)
    corrupt_idx = rng.choice(n, size=n_corrupt, replace=False)
    for idx in corrupt_idx:
        current = result[idx]
        wrong_classes = [c for c in CLASSES if c != current]
        result[idx] = rng.choice(wrong_classes)
    return result
 
def random_predictions(n):
    """Pure random predictions."""
    rng = np.random.default_rng(seed=42)
    return rng.choice(CLASSES, size=n)
 
def save_submission(pred_labels, team_name):
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    submission = pd.DataFrame({
        "row_id": range(len(pred_labels)),
        "Class":  pred_labels
    })
    output_path = os.path.join(SUBMISSIONS_DIR, f"{team_name}_submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"\n✓ Submission saved: {output_path}")
    print(f"  Rows: {len(submission)}")
    print(f"  Class distribution:\n{submission['Class'].value_counts().to_string()}")
    return output_path
 
def main():
    parser = argparse.ArgumentParser(description="Generate test submissions with known accuracy levels.")
    parser.add_argument(
        "--mode",
        choices=["perfect", "good", "baseline", "bad", "random"],
        default="good",
        help="Accuracy level of the generated submission"
    )
    args = parser.parse_args()
 
    MODE_CONFIG = {
        "perfect":  {"error_rate": 0.00, "team": "TestTeam_Perfect"},
        "good":     {"error_rate": 0.20, "team": "TestTeam_Good"},
        "baseline": {"error_rate": 0.26, "team": "TestTeam_Baseline"},
        "bad":      {"error_rate": 0.60, "team": "TestTeam_Bad"},
        "random":   {"error_rate": None, "team": "TestTeam_Random"},
    }
 
    config = MODE_CONFIG[args.mode]
    print(f"\nMode     : {args.mode}")
    print(f"Team name: {config['team']}")
 
    print("\nLoading data...")
    train, test = load_data()
    n_test = len(test)
    print(f"Test rows: {n_test}")
 
    print("Generating predictions...")
    if args.mode == "random":
        pred_labels = random_predictions(n_test)
    else:
        real_preds = get_real_predictions(train, test)
        pred_labels = corrupt_predictions(list(real_preds), config["error_rate"])
 
    pred_labels = list(pred_labels)
    save_submission(pred_labels, config["team"])
 
    print(f"""
Next steps to test automation:
-------------------------------
1. git add submissions/{config['team']}_submission.csv
2. git commit -m "Test submission: {config['team']}"
3. git push origin main
 
Then check the Actions tab on GitHub to see the workflow run.
The leaderboard should update within ~1 minute.
Expected ranking order if you run all modes:
  1. TestTeam_Perfect   (highest F1)
  2. TestTeam_Good
  3. TestTeam_Baseline
  4. TestTeam_Bad
  5. TestTeam_Random    (lowest F1)
""")
 
if __name__ == "__main__":
    main()
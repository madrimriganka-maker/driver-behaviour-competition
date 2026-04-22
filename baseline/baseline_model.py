import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train = pd.read_csv(os.path.join(BASE, "data", "train_motion_data.csv"))
test  = pd.read_csv(os.path.join(BASE, "data", "test_motion_data_nolabels.csv"))

# Encode labels
le = LabelEncoder()
train["Class"] = le.fit_transform(train["Class"])  # Normal=1, Aggressive=0, Drowsy=2 (varies)

X_train = train.drop(columns=["Class"])
y_train = train["Class"]
X_test = test.copy()

# Train
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
preds = clf.predict(X_test)
pred_labels = le.inverse_transform(preds)

# Save submission
submission = pd.DataFrame({
    "row_id": range(len(pred_labels)),
    "Class": pred_labels
})
output_path = os.path.join(BASE, "submissions", "TeamSeneritasubmission.csv")
submission.to_csv(output_path, index=False)
print(f"Submission saved to: {output_path}")

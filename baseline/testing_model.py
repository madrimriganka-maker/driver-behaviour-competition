import os
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train = pd.read_csv(os.path.join(BASE, "data", "train_motion_data.csv"))
test  = pd.read_csv(os.path.join(BASE, "data", "test_motion_data_nolabels.csv"))

le = LabelEncoder()
train["Class"] = le.fit_transform(train["Class"])

X_train = train.drop(columns=["Class"])
y_train = train["Class"]
X_test  = test.copy()

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

clf = GradientBoostingClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
pred_labels = le.inverse_transform(preds)

submission = pd.DataFrame({
    "row_id": range(len(pred_labels)),
    "Class": pred_labels
})

output_path = os.path.join(BASE, "submissions", "TeamBKMS_submission.csv")
import os
os.makedirs(os.path.join(BASE, "submissions"), exist_ok=True)  # creates folder if missing
output_path = os.path.join(BASE, "submissions", "TeamBKMS_submission.csv")
submission.to_csv(output_path, index=False)
print(f"Done. Saved to {output_path}")
submission.to_csv(output_path, index=False)
print(f"Done. Saved to {output_path}")
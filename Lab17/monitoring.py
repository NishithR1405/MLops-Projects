import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from evidently import Report
from evidently.presets import DataDriftPreset

# ---------------------------------------------------------
# Step 1: Load dataset
# ---------------------------------------------------------
df = pd.read_csv("marketing.csv")

# Encode the diagnosis target (M = Malignant -> 1, B = Benign -> 0)
# so it becomes numeric like the rest of the features.
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Drop the id column (not a useful feature) and keep numeric columns only
df = df.drop(columns=["id"])
df = df.select_dtypes(include=["int64", "float64"]).dropna()

# ---------------------------------------------------------
# Step 2: Split into reference (training-time) and current (production) data
# ---------------------------------------------------------
reference_data, current_data = train_test_split(df, test_size=0.5, random_state=42)

X_train = reference_data.drop(columns=["diagnosis"])
y_train = reference_data["diagnosis"]

X_current = current_data.drop(columns=["diagnosis"])
y_current = current_data["diagnosis"]

# ---------------------------------------------------------
# Step 3: Train model
# ---------------------------------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
current_acc = accuracy_score(y_current, model.predict(X_current))

print(f"Model trained successfully")
print(f"Reference accuracy: {train_acc:.4f}")
print(f"Current accuracy:   {current_acc:.4f}")

# ---------------------------------------------------------
# Step 4: Generate the Evidently drift report
# ---------------------------------------------------------
report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(reference_data=reference_data, current_data=current_data)
snapshot.save_html("drift_report.html")

print("Drift report generated successfully")

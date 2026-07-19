import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("churn.csv")
df = df.dropna()

for column in df.columns:
    if df[column].dtype == "object" or pd.api.types.is_string_dtype(df[column]):
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))

df = df.drop("customerID", axis=1, errors="ignore")

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

print("Model trained and saved successfully")

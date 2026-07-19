import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("churn.csv")
df = df.dropna()

for column in df.columns:
    if df[column].dtype == "object" or pd.api.types.is_string_dtype(df[column]):
        df[column] = df[column].astype("category").cat.codes

df = df.drop("customerID", axis=1, errors="ignore")

model = joblib.load("model.pkl")

X = df.drop("Churn", axis=1)

st.title("Explainable AI Dashboard")

st.subheader("Dataset Preview")
st.write(X.head())

row_index = st.slider("Select Customer Record", 0, len(X) - 1, 0)
input_data = X.iloc[[row_index]]

prediction = model.predict(input_data)
st.subheader("Prediction Result")
st.write(prediction)

explainer = shap.TreeExplainer(model)

# Use a sample of the dataset for the summary plot so the dashboard stays
# responsive in real time (SHAP on all 7043 rows is too slow for interactive use)
X_sample = X.sample(n=min(300, len(X)), random_state=42)
shap_values = explainer.shap_values(X_sample)

st.subheader("SHAP Summary Plot")
plt.close("all")
# RandomForestClassifier binary output -> shap_values has shape (n, features, 2);
# select the "Churn = Yes" class slice for the summary plot
sv_summary = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values
shap.summary_plot(sv_summary, X_sample, show=False)
fig1 = plt.gcf()
st.pyplot(fig1)
plt.close(fig1)

st.subheader("SHAP Force Plot")
plt.close("all")
expected_value = (
    explainer.expected_value[1]
    if hasattr(explainer.expected_value, "__len__")
    else explainer.expected_value
)
row_shap_raw = explainer.shap_values(input_data)
row_shap_values = (
    row_shap_raw[0, :, 1] if row_shap_raw.ndim == 3 else row_shap_raw[1][0]
)
shap.plots.waterfall(
    shap.Explanation(
        values=row_shap_values,
        base_values=expected_value,
        data=input_data.iloc[0],
        feature_names=X.columns.tolist(),
    ),
    show=False,
)
fig2 = plt.gcf()
st.pyplot(fig2)
plt.close(fig2)

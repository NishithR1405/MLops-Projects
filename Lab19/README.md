# P19 — Explainable AI in Production using SHAP

Customer Churn Prediction Explainability Dashboard (DATA303 MLOps Lab Sheet P19)

## Folder contents
- `train_model.py` — trains a RandomForestClassifier on the churn dataset and saves `model.pkl`
- `app.py` — Streamlit dashboard: dataset preview, customer-record slider, live prediction, SHAP summary plot, SHAP waterfall/force plot
- `churn.csv` — Telco Customer Churn dataset (7,043 rows x 21 columns)
- `model.pkl` — trained model
- `requirements.txt` — pinned dependencies
- `training_output.txt` — captured console output from training
- `screenshots/` — full set of screenshots from every step
- `final_screenshots/` — the 5 curated deliverable screenshots used in the report

## How to run
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
Then open http://localhost:8501 in a browser.

## Notes on issues found and fixed while completing this lab

1. **Dead dataset link.** The URL in the original lab sheet
   (`raw.githubusercontent.com/blastchar/telco-customer-churn/...`) returns 404.
   The identical dataset was retrieved from a verified mirror
   (`mindsdb-examples` repository, same IBM/Kaggle source data).

2. **pandas 3.0 dtype change.** `df[column].dtype == "object"` no longer
   matches text columns in pandas 3.0 (they default to a `str`-backed dtype),
   so the lab sheet's encoding loop silently skipped every categorical
   column, including `customerID`, and training crashed. Fixed by also
   checking `pd.api.types.is_string_dtype(df[column])`, and by explicitly
   dropping the non-predictive `customerID` column.

3. **SHAP + Streamlit re-run bug.** Pre-creating an empty `plt.subplots()`
   figure before calling `shap.summary_plot()` / `shap.plots.waterfall()`
   raised `ValueError: Argument must be an image or collection in this Axes`
   on Streamlit re-runs (e.g. moving the slider). Fixed by calling
   `plt.close("all")` before each SHAP plot call and capturing the actual
   figure SHAP drew with `plt.gcf()`.

4. **Performance.** SHAP TreeExplainer on the full 7,043-row dataset is too
   slow for a responsive dashboard. The summary plot uses a random sample of
   300 rows (standard practice); the per-customer waterfall plot is always
   computed exactly for the selected row.

See `Explainability_Report.docx` for the full write-up, screenshots, model
metrics, and discussion-question answers.

from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Create Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Customer Churn Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    # Get JSON data from request
    data = request.json

    # Convert JSON to DataFrame
    input_data = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Convert prediction to text
    result = "Churn" if prediction == 1 else "No Churn"

    # Return JSON response
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
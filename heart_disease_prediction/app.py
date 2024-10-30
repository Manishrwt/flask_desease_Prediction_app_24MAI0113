# app.py
from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Home route to display the form
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route to handle form submission and predict
@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    age = float(request.form["age"])
    cholesterol = float(request.form["cholesterol"])
    blood_pressure = float(request.form["blood_pressure"])

    # Prepare data for model input
    input_data = np.array([[age, cholesterol, blood_pressure]])
    prediction = model.predict(input_data)

    # Result based on prediction
    if prediction[0] == 1:
        result = "Positive: The patient is likely to have heart disease."
    else:
        result = "Negative: The patient is unlikely to have heart disease."

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

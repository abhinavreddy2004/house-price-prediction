from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS  # Import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = joblib.load("house_price_model.pkl")

@app.route("/")
def home():
    return "Welcome to House Price Prediction API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Convert data to numpy array
        features = np.array(data["features"]).reshape(1, -1)

        # Check if feature length matches model input
        expected_features = model.n_features_in_
        if features.shape[1] != expected_features:
            return jsonify({"error": f"Expected {expected_features} features, but got {features.shape[1]}"})

        # Make prediction
        prediction = model.predict(features)[0]

        return jsonify({"predicted_price": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

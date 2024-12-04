from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("logistic_regression_model.pkl")

@app.route('/')
def home():
    return "Welcome to the ML Model API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.json
    try:
        # Ensure data is in the correct format
        features = np.array(data['features']).reshape(1, -1)
        # Make prediction
        prediction = model.predict(features).tolist()
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

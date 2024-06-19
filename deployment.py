from flask import Flask, request, jsonify
import joblib
import pandas as pd
from modeldevelopment import model
# Save the model
joblib.dump(model, 'traffic_model.pkl')

# Load the model
model = joblib.load('traffic_model.pkl')

# Create a Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = pd.DataFrame(data)
    prediction = model.predict(features)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)


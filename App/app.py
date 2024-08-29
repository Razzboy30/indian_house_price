from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model # type: ignore


app = Flask(__name__)

# Load the model and scaler using joblib
model = joblib.load('xgb.pkl')
scaler = joblib.load('scl.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(final_features)
    prediction = model.predict(scaled_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Estimated Price: ${output}')

if __name__ == "__main__":
    app.run(debug=True)

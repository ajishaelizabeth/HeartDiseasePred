import numpy as np
from tensorflow import keras
from keras.models import load_model
from flask import Flask, request, render_template, redirect, url_for, session
import pyrebase

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "secret_key"

# Firebase configuration (update with your project info)
config = {
    "apiKey": "your_api_key",
    "authDomain": "your_auth_domain",
    "projectId": "your_project_id",
    "storageBucket": "your_storage_bucket",
    "messagingSenderId": "your_messaging_sender_id",
    "appId": "your_app_id",
    "measurementId": "your_measurement_id",
    "databaseURL": "your_database_url"
}

# Initialize Firebase
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()

# Load the LSTM model
lstm_model = load_model('lstm_model.h5')

# Define home route
@app.route("/")
def home():
    return render_template("index.html")

# Define heart route for heart disease prediction
@app.route('/heart')
def heart_disease():
    return render_template("heart.html")

# Define calculate route for prediction
@app.route('/calculate', methods=['POST'])
def calculate():
    # List of form fields for the model
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'restecg', 'thalach', 'exang', 'oldpeak', 'thal']

    # Retrieve features from the form and convert to float
    features = [float(request.form.get(name)) for name in feature_names]

    # Convert features to a NumPy array and reshape for LSTM input
    final_input = np.array([features]).reshape(1, len(features), 1)

    # Make a prediction using the LSTM model
    prediction = lstm_model.predict(final_input)
    
    # Convert prediction to class (0 or 1) based on threshold of 0.5
    prediction_class = int(np.round(prediction[0, 0]))
    
    # Determine the result message based on the predicted class
    if prediction_class == 0:
        result = "Probability of having heart disease is very low."
    else:
        result = "You are at high risk of having heart disease."

    # Render the heart.html template and pass the prediction result
    return render_template("heart.html", predict=result)

# Add other routes and functions as needed...

if __name__ == '__main__':
    app.run(debug=True)

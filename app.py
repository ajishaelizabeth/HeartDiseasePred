from flask import Flask, request, render_template
import numpy as np
from tensorflow import keras
from keras.models import load_model

app = Flask(__name__)

# Load your machine learning model
loaded_model = load_model(r'C:\Users\ajish\OneDrive\Desktop\VS-CODE\Project\mediflask\lstm_model.h5')  # Provide the correct path to your model file


@app.route("/")
def home():
    # Render the `heart.html` template as the home page
    return render_template("heart.html")

@app.route('/expenditure', methods=['POST'])
def predict_heart_disease():
    # Retrieve the form data
    form_data = request.form

    # Extract the values from the form data
    age = float(form_data.get('age'))
    sex = float(form_data.get('sex'))
    cp = float(form_data.get('cp'))
    trestbps = float(form_data.get('trestbps'))
    restecg = float(form_data.get('restecg'))
    thalach = float(form_data.get('thalach'))
    exang = float(form_data.get('exang'))
    oldpeak = float(form_data.get('oldpeak'))
    thal = float(form_data.get('thal'))

    # Create a NumPy array with the input values
    input_features = np.array([[age, sex, cp, trestbps, restecg, thalach, exang, oldpeak, thal]])

    # Reshape the array to match the input shape expected by the model
    input_features = input_features.reshape((1, 1, 9))

    # Make a prediction using the model
    prediction = loaded_model.predict(input_features)
    prediction_class = np.round(prediction[0, 0])

    # Determine the result based on the prediction
    if prediction_class == 0:
        result = "Probability of having heart disease is very low."
    else:
        result = "You are at high risk of having heart disease."

    # Render the result in a template
    return render_template('heart_result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)

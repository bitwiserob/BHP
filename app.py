from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model
model = load_model('boston_housing_model.keras')

# Mean and standard deviation from training data for standardization
mean = np.load('mean_values.npy')
std = np.load('std_values.npy')



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get data from form
        features = np.array([[
            float(request.form['CRIM']),
            float(request.form['ZN']),
            float(request.form['INDUS']),
            float(request.form['CHAS']),
            float(request.form['NOX']),
            float(request.form['RM']),
            float(request.form['AGE']),
            float(request.form['DIS']),
            float(request.form['RAD']),
            float(request.form['TAX']),
            float(request.form['PTRATIO']),
            float(request.form['B']),
            float(request.form['LSTAT']),
        ]])

        # Standardize the input features
        features = (features - mean) / std

        # Predict using the model
        predicted_price = model.predict(features)[0][0]

        return render_template('index.html', prediction=predicted_price)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
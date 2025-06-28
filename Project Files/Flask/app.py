import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open("model.pkl", "rb"))
scale = pickle.load(open("scale.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input values from form
    input_features = [float(x) for x in request.form.values()]
    features_values = np.array(input_features).reshape(1, -1)

    # Column names must match training data
    names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day', 'hours', 'minutes', 'seconds']
    
    # Create a DataFrame
    data = pd.DataFrame(features_values, columns=names)

    # Scale the data
    data_scaled = scale.transform(data)

    # Predict using the loaded model
    prediction = model.predict(data_scaled)[0]

    # Render result page with input + prediction
    return render_template("result.html",
                           holiday=request.form['holiday'],
                           temp=request.form['temp'],
                           rain=request.form['rain'],
                           snow=request.form['snow'],
                           weather=request.form['weather'],
                           year=request.form['year'],
                           month=request.form['month'],
                           day=request.form['day'],
                           hours=request.form['hours'],
                           minutes=request.form['minutes'],
                           seconds=request.form['seconds'],
                           result=int(prediction))

if __name__ == "__main__":
    app.run(debug=True, port = 5000)

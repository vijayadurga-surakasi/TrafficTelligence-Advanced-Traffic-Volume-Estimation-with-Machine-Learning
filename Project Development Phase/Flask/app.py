import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
import pickle
with open(r'model.pkl', 'rb') as file:
    file.seek(0)
    model = pickle.load(open("model.pkl", 'rb'))
    scale = pickle.load(open("scale.pkl", 'rb'))

@app.route('/') 
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
   
    input_feature = [float(x) for x in request.form.values()]
    features_values = [np.array(input_feature)]
    names = [['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day',
              'hours', 'minutes', 'seconds']]
    data = pandas.DataFrame(features_values, columns=names)

    prediction = model.predict(data)

    text = f"Estimated Traffic Volume: {int(prediction[0])}"
    return render_template("result.html", result=text)


if __name__ == "__main__":

    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
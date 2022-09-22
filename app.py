import pickle
from flask import Flask, request, app, jsonify, url_for, render_template

import numpy as np
import pandas as pd


app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('logistic_reg.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

#@app.route('/predict_api',methods = ['POST'])

def predict_api():
    data = request.json['data']
    new_data =  np.array(list(data.values())).reshape(1,-1)
    print(new_data)
    output = regmodel.predict(new_data)
    print(output)
    print(output[0])

    return jsonify(int(output[0]))


@app.route('/predict',methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1,-1)
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text = "The prediction for loan eligibility is {} (1 = Yes, 0 = No)".format(output))

if __name__ =="__main__":
    app.run(debug=True)





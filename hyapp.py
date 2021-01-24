import pandas as pd, numpy as np
import pickle
import flask
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
@app.route('/')
def Hm():
    return render_template('hyindex.html')

model_accum = pickle.load(open('pipeline_accum.pkl','rb'))
model_flag = pickle.load(open('pipeline_flag.pkl','rb'))

@app.route('/predict', methods=['POST'])
def predict():
    inputt = [int(x) for x in request.form.values()]
    xtest = np.array(inputt)
    xtest_df = pd.DataFrame(xtest)
    xtest_df.set_index(xtest,inplace=True)
    Xtest_df = pd.DataFrame(index = xtest_df.index)

    prediction_accum = model_accum.predict(Xtest_df)
    prediction_flag = model_flag.predict(Xtest_df)
    
    return render_template('hypredict.html',prediction_text= 'Hydraulic accumulation prediction is ' + format(prediction_accum) + ' and stability flag prediction is ' + format(prediction_flag))

if __name__ == "__main__":
    app.run(debug=True)

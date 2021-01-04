import json
import pandas as pd
import os
import joblib, pickle
from azureml.core import Model


def init():
    global daone
    model_path = Model.get_model_path('registered_class.sav')
    daone = joblib.load(model_path)

def run(data):
    try:
        trynn = json.loads(data)
        data = pd.DataFrame(trynn['data'])
        result = daone.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error

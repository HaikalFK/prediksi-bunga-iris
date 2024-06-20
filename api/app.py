import pandas as pd
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

with open("model/pemodelan_numpy.pkl", "rb") as model_file:
    model_numpy = pickle.load(model_file)

with open("model/pemodelan_pandas.pkl", "rb") as model_file:
    model_pandas = pickle.load(model_file)

LABEL = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
FEATURES = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

@app.route('/')
def index():
    return jsonify({"status": "SUCCESS", "message": "Service is Up"})

@app.route('/predict/numpy', methods=['POST'])
def predict_numpy():
    data = request.json
    sl = data.get('sepal length', 0.0)
    sw = data.get('sepal width', 0.0)
    pl = data.get('petal length', 0.0)
    pw = data.get('petal width', 0.0)

    new_data = [[sl, sw, pl, pw]]
    res = model_numpy.predict(new_data)
    result = LABEL[res[0]]

    return jsonify({"status": "SUCCESS",
                    "input type": "numpy array",
                    "input": {'sepal length': sl,
                              'sepal width': sw,
                              'petal length': pl,
                              'petal width': pw},
                    "result": result})

@app.route('/predict/pandas', methods=['POST'])
def predict_pandas():
    data = request.json
    sl = data.get('sepal length', 0.0)
    sw = data.get('sepal width', 0.0)
    pl = data.get('petal length', 0.0)
    pw = data.get('petal width', 0.0)

    new_data = [[sl, sw, pl, pw]]
    new_data = pd.DataFrame(new_data, columns=FEATURES)
    res = model_pandas.predict(new_data)
    result = LABEL[res[0]]

    return jsonify({"status": "SUCCESS",
                    "input type": "pandas dataframe",
                    "input": {'sepal length': sl,
                              'sepal width': sw,
                              'petal length': pl,
                              'petal width': pw},
                    "result": result})

if __name__ == '__main__':
    app.run(debug=True)

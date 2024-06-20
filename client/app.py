from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sl = request.form['sl']
        sw = request.form['sw']
        pl = request.form['pl']
        pw = request.form['pw']
        model_type = request.form['model']

        data = {
            'sl': float(sl),
            'sw': float(sw),
            'pl': float(pl),
            'pw': float(pw)
        }

        if model_type == 'numpy':
            response = requests.post('http://127.0.0.1:5000/predict/numpy', json=data)
        else:
            response = requests.post('http://127.0.0.1:5000/predict/pandas', json=data)

        if response.status_code == 200:
            result = response.json()['result']
        else:
            result = "Error in prediction"

        return render_template('index.html', prediction=result)
    except KeyError as e:
        return f"Missing parameter: {e}", 400
    except Exception as e:
        return f"An error occurred: {e}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)

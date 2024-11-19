from flask import Flask, render_template, request
import numpy as np
from model.model import load_model, scaler

app = Flask(__name__)
classifier = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'pclass': int(request.form['pclass']),
            'sex': 1 if request.form['sex'] == "male" else 0,
            'age': float(request.form['age']),
            'sibsp': int(request.form['sibsp']),
            'parch': int(request.form['parch']),
            'fare': float(request.form['fare']),
            'embarked': request.form['embarked']
        }
        embarked_C = 1 if data['embarked'] == "C" else 0
        embarked_Q = 1 if data['embarked'] == "Q" else 0

        user_input = np.array([data['pclass'], data['age'], data['sibsp'], data['parch'], data['fare'], data['sex'], embarked_C, embarked_Q]).reshape(1, -1)
        prediction = classifier.predict(scaler.transform(user_input))
        result = "Survived" if prediction[0] == 1 else "Not Survived"
        
        return render_template('result.html', prediction=result)
    except Exception:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
    
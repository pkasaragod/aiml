import joblib
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
model = pickle.load(open('lr_cancer_model.pkl', 'rb'))

app = Flask(__name__,template_folder='templates')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict_value', methods=['POST'])
def predict_value():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    if output == 1:
        res_val = "Malignant 😨!"
    else:
        res_val = "benign"

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))


if __name__ == "__main__":
    app.run(debug=True)



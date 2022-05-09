# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the saved KNN model
filename = 'drug5_rf_classifier.pkl'
model = pickle.load(open(filename, 'rb'))


def process_input(sex, bp, cholesterol, age, Na_to_K):
    arr = np.zeros(10)

    if sex == 'Male':
        arr[0] = 1
    if bp == 'Low':
        arr[1] = 1
    elif bp == 'Normal':
        arr[2] = 1
    if cholesterol == 'Normal':
        arr[3] = 1
    if 29 < age <= 39:
        arr[4] = 1
    elif 39 < age <= 49:
        arr[5] = 1
    elif age <= 19:
        arr[6] = 1
    arr[7] = Na_to_K
    if 50 <= age:
        arr[8] = 1
    if Na_to_K > 15:
        arr[9] = 1
    
    return arr.reshape(1, -1)


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['Age'])
        sex = str(request.form['Sex'])
        bp = str(request.form['BP'])
        cholesterol = str(request.form['Cholesterol'])
        Na_to_K = float(request.form['Na_to_K'])
        
        processed_arr = process_input(sex, bp, cholesterol, age, Na_to_K)
        my_prediction = model.predict(processed_arr)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)

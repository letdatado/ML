# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the saved KNN model
filename = 'banknotes_auth_knn_classifier.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        variance = float(request.form['variance'])
        skewness = float(request.form['skewness'])
        kurtosis = float(request.form['kurtosis'])
        entropy = float(request.form['entropy'])
        
        new = np.array([[variance, skewness, kurtosis, entropy]])
        my_prediction = model.predict(new)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)

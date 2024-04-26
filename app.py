from flask import Flask, render_template, request
import requests, pickle, joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

scaler = StandardScaler()

@app.route('/')
def index():
    return render_template('index.html',result = '')

@app.route('/predict', methods = ["GET","POST"])
def predict():
    #gender = 1 if requests.form.get('gender') == 'Male' else 0
    age = int(request.form.get('age'))
    ann_income = int(request.form.get('annual_income'))
    sp_score = int(request.form.get('spending_score'))
    model = joblib.load('spend_model.pkl')
    ans = model.predict([[age,ann_income,sp_score]])
    if ans == [4] :
        return  render_template('index.html',result="This customer has very high chances of spending inside your Mall!")
    elif ans == [3] or ans == [0]:
        return render_template('index.html',result="This customer has medium chances of spending inside your Mall!")
    
    return render_template('index.html',result="This customer has low chances of spending inside your Mall!")

if __name__ == "__main__":
    app.run(debug=True)
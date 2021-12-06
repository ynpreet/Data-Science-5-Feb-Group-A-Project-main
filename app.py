from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd

def prob(country,year):
    if country =="India":
        model=pickle.load(open('ridge_ind_model.pkl', 'rb'))
        if year == 2019:
             array= np.array([year,0,0,0,0,0,0,0])
             array= array.reshape(1, -1)
        else:
            df = pd.read_csv(r'Data-FER\India.csv')
            array = df.loc[df['Year'] == year]
        return model.predict(array)
    elif country =="Switzerland":
        model=pickle.load(open('lr_swz_model.pkl', 'rb'))
        if year == 2019:
             array= np.array([year,0,0,0,0,0,0,0])
             array= array.reshape(1, -1)
        else:
            df = pd.read_csv(r'Data-FER\Switzerland.csv')
            array = df.loc[df['Year'] == year]
        return model.predict(array)
    elif country =="China":
        model=pickle.load(open('rf_chn_model.pkl', 'rb'))
        if year == 2019:
             array= np.array([year,0,0,0,0,0,0,0])
             array= array.reshape(1, -1)
        else:
            df = pd.read_csv(r'Data-FER\China.csv')
            array = df.loc[df['Year'] == year]
        return model.predict(array)
    elif country =="Japan":
        model=pickle.load(open('rf_jpn_model.pkl', 'rb'))
        if year == 2019:
             array= np.array([year,0,0,0,0,0,0,0])
             array= array.reshape(1, -1)
        else:
            df = pd.read_csv(r'Data-FER\Japan.csv')
            array = df.loc[df['Year'] == year]
        return model.predict(array)
    elif country =="Canada":
        model=pickle.load(open('ridge_can_model.pkl', 'rb'))
        if year == 2019:
             array= np.array([year,0,0,0,0,0,0,0])
             array= array.reshape(1, -1)
        else:
            df = pd.read_csv(r'Data-FER\Canada.csv')
            array = df.loc[df['Year'] == year]
        return model.predict(array)
    else:
        model=pickle.load(open('ridge_uk_model.pkl', 'rb'))
        if year == 2019:
             array= np.array([year,0,0,0,0,0,0,0])
             array= array.reshape(1, -1)
        else:
            df = pd.read_csv(r'Data-FER\UK.csv')
            array = df.loc[df['Year'] == year]
        return model.predict(array)

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    count1 = request.form.get('Country1')  # country1 is field name in html file in index
    count2 = request.form.get('Country2')  # country2 is field name in html file in index
    year= int(request.form.get('YEAR'))        # year is field name in html file in index
    amount = int(request.form.get('amount'))     # amount is field name in html file in index
    #year_arr= np.array([year,0,0,0,0,0,0,0])
    #year_arr= year_arr.reshape(1, -1)
    #because the model needs GDP PPP vector1 etc so I replaced it with zero
    result1 = prob(count1,year)
    result2 = prob(count2,year)
    final_result= (amount*result2)/result1
    return render_template('index.html',pred='Value of the Currency is {}'.format(final_result))


        
    
if __name__ == '__main__':
    app.run(debug=True)

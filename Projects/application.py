from flask import Flask,render_template,request
import  numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


application=Flask(__name__)
app=application

ridgecv_model=pickle.load(open('python/model_deployment/model/ridgecv.pkl','rb'))
standard_scaler=pickle.load(open('python/model_deployment/model/scaler.pkl','rb'))


@app.route('/')
def index():
    return "Welcome to the Home page"

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridgecv_model.predict(new_data)

        return render_template('index.html',results=result[0])

    else:
        return render_template('index.html')



if __name__=='__main__':
    app.run(host='0.0.0.0')


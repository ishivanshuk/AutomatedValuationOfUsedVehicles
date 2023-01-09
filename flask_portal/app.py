from flask import Flask, render_template, request, redirect, flash, send_from_directory
import pandas as pd
import pickle
from werkzeug.utils import secure_filename
import cv2
import PIL
import pytesseract

import os

from utils import *

app = Flask(__name__)
app.config['UPLOADS_FOLDER'] = UPLOADS_FOLDER
app.config['SECRET_KEY'] = 'my secret'
estimated_price=0

@app.route('/')

def home():
    custdf=pd.read_csv('unique.csv', names=['company','model'])
    company_names = custdf['company'].unique()
    model_names = custdf.values.tolist()

    if request.method== 'POST':
        model = pickle.load(open('price_prediction_model.pkl','rb'))    
        df = pd.read_csv('cleaned df.csv')

        year= int(request.form['datepicker'])

        distance= request.form['distance']

        seller_type= request.form['sellertype']        
        if seller_type=='individual':
            seller_type=int(0)
        elif seller_type=='dealer':
            seller_type=int(1)
 
        fuel= request.form['fuel']   
        if fuel=='petrol':
            fuel=int(0)
        elif fuel=='diesel':
            fuel=int(1)
        elif fuel=='cng':
            fuel=int(2)
        else:
            fuel=int(3)                 
       
        transmission= request.form['transmissiontype'] 
        if transmission=='manual':
            transmission=int(0)
        else:
            transmission=int(1)
        
        owner= request.form['owner']
        if owner=='first':
            owner=int(1)
        elif owner=='second':
            owner=int(2)
        elif owner=='third':
            owner=int(3)
        else:
            owner=int(4)
        
        mileage= request.form['mileage']
        seats=5                                        
        modelname= 'indigo cr4'
        company=request.form['company']
       
        #engine power, max power(bhp) to be api extracted from dataset --- done
        ep = df.loc[df['model_name'] == modelname]['engine (cc)']
        enginepower = ep.iloc[0]
        mp = df.loc[df['model_name'] == modelname]['max_power (bhp)']
        maxpower = mp.iloc[0]
     
        estimated_price= model.predict(pd.DataFrame([[year, distance, fuel, seller_type, transmission, owner, seats, company, modelname, mileage, int(enginepower) ,int(maxpower)]], columns=['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'seats', 'company_name', 'model_name', 'mileage (kmpl)', 'engine (cc)', 'max_power (bhp)']))
        
        print(estimated_price)
    return render_template('index.html', company_names=company_names, model_names=model_names)

@app.route('/output', methods=['GET', 'POST'])
def func():
    custdf=pd.read_csv('unique.csv', names=['company','model'])
    company_names = custdf['company'].unique()
    model_names = custdf.values.tolist()
    match_check=1
    match_check_number=1
    if request.method== 'POST':
        model = pickle.load(open('price_prediction_model.pkl','rb'))    
        vehicle_number= request.form['car_number']
        date=request.form['certificate_date']
        df = pd.read_csv('cleaned df.csv')

        year= int(request.form['datepicker'])

        distance= request.form['distance']

        seller_type= request.form['sellertype']        
        if seller_type=='individual':
            seller_type=int(0)
        elif seller_type=='dealer':
            seller_type=int(1)
 
        fuel= request.form['fuel']   
        if fuel=='petrol':
            fuel=int(0)
        elif fuel=='diesel':
            fuel=int(1)
        elif fuel=='cng':
            fuel=int(2)
        else:
            fuel=int(3)                 
       
        transmission= request.form['transmissiontype'] 
        if transmission=='manual':
            transmission=int(0)
        else:
            transmission=int(1)
        
        owner= request.form['owner']
        if owner=='first':
            owner=int(1)
        elif owner=='second':
            owner=int(2)
        elif owner=='third':
            owner=int(3)
        else:
            owner=int(4)
        
        mileage= request.form['mileage']
        seats=5                                        
        modelname= request.form['model_name']
        company=request.form['company']
        

        #engine power, max power(bhp) to be api extracted from dataset --- done
        ep = df.loc[df['model_name'] == modelname]['engine (cc)']
        enginepower = ep.iloc[0]
        mp = df.loc[df['model_name'] == modelname]['max_power (bhp)']
        maxpower = mp.iloc[0]

        estimated_price= model.predict(pd.DataFrame([[year, distance, fuel, seller_type, transmission, owner, seats, company, modelname, mileage, int(enginepower) ,int(maxpower)]], columns=['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'seats', 'company_name', 'model_name', 'mileage (kmpl)', 'engine (cc)', 'max_power (bhp)']))
        
        print(estimated_price)
        #OCR begins

        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'

        img = cv2.imread('uploads/images/document_1.jpg')

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        cv2.imshow('Result',img)

        print(pytesseract.image_to_string(img))
        cv2.imshow('Result',img)
        ##cv2.waitKey(0)

        hImg,wImg,_ = img.shape
        boxes = pytesseract.image_to_boxes(img)
        for b in boxes.splitlines():
            print(b)

        for b in boxes.splitlines():    
            b = b.split(' ')
            print(b)
            x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])

        ##cv2.rectangle(img,(x,hImg-y),(w,hImg-h),(0,0,255),3)
        #cv2.putText(img,b[0],(x,hImg-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),2)

        ##detecting words
        hImg,wImg,_ = img.shape
        boxes = pytesseract.image_to_data(img)
        print(boxes)

        for x,b in enumerate(boxes.splitlines()):
            if x!=0:
                b = b.split()
                if len(b)==12:
                    print(b)

        true_vehicle_no = 0
        for x,b in enumerate(boxes.splitlines()):
            if x!=0:
                b = b.split()
                if len(b)==12:
                    #print (b[11])
                    if b[11] == vehicle_number:
                        true_vehicle_no=1
                        break  
                  
        print (true_vehicle_no) 
        if match_check_number==true_vehicle_no:
               match_check_number="match"
        else:
            match_check_number="do not match"         

        true_valid_upto = 0
        for x,b in enumerate(boxes.splitlines()):
            if x!=0:
                b = b.split()
                if len(b)==12:
                    #print (b[11])
                    if b[11] == date:
                        true_valid_upto=1
                        break  

        print (true_valid_upto)
        if match_check==true_valid_upto:
            match_check="match"
        else:
            match_check="do not match"
    #OCR ends
    
    
    if match_check=="do not match":
        estimated_price=0.95*estimated_price

    if match_check_number=="do not match (upload valid certificate for better value)":
        estimated_price=0.80*estimated_price

    print(vehicle_number)
    print(date)
    return render_template('output.html', estimated_price=int(estimated_price), match_check_number=match_check_number, match_check=match_check, company_names=company_names, model_names=model_names, modelname=modelname, company=company )

@app.route('/uploads/images/<path:filename>')
def send_attachment(filename):
  return send_from_directory(app.config['UPLOADS_FOLDER'], 
    filename=filename, as_attachment=True)

if __name__== '__main__':
    app.run(port=3000, debug=True)
    
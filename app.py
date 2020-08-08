import tensorflow as tf
import pickle
import pandas as pd
from PIL import Image
model = tf.keras.models.load_model('GovMalaria.h5')
model1 = tf.keras.models.load_model('GovCovidXray.h5')
model2 = pickle.load(open('GovtDiabetes-model.pkl', 'rb'))
model3 = pickle.load(open('CovidSymtoms.pkl', 'rb'))

import streamlit as st

image = Image.open('medicimg.png')
st.image(image, use_column_width=True,format='PNG')

html_temp = """
    <div style="background-color:#010200;padding:6px">
    <h2 style="color:white;text-align:center;">G. S. MediSafe Application</h2>
    </div>
    """ 
st.markdown(html_temp,unsafe_allow_html=True)

import cv2
from PIL import Image, ImageOps
import numpy as np

def malaria(image_data, model):
    
        size = (94,94)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(94, 94),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction

def covid(image_data, model1):
    
        size = (94,94)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(94, 94),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model1.predict(img_reshape)
        
        return prediction

def main():
    activities = ["Home","Malaria Detection","Pneumonia Detection","Diabetes detection","Covid-19 Symptom Detection","Contact Information","Disclaimer"]  
    
    st.subheader("Select Activity :")
    choice = st.selectbox("", activities)
    
    #st.write("This is a simple image classification web app to predict rock-paper-scissor hand sign")
    #file = st.file_uploader("Please upload an image file", type=["jpg", "png","jpeg"])
    
    if choice == 'Malaria Detection':
        image = Image.open('mimg.png')
        st.image(image, use_column_width=True,format='PNG')
        st.subheader("Malaria Detection :")
        
        st.write("This is Malaria Detection section, Please upload the Image Cell of a patient to be tested, for Malaria Detection")
        file = st.file_uploader("Please upload an image file", type=["jpg", "png","jpeg"])
        
        if file is None:
            st.text("Please upload an image file")
        else:
            image = Image.open(file)
            st.image(image, use_column_width=True)
            prediction = malaria(image, model)
            if prediction <= 0.5:
                st.subheader("Result: No Malaria Detected")
            else:
                st.subheader("Result: Malaria Detected")
                
    elif choice == 'Pneumonia Detection':
        image = Image.open('pimg.png')
        st.image(image, use_column_width=True,format='PNG')
        st.subheader("Pneumonia Detection")
        
        st.write("This is Pneumonia Detection section, Please upload the Chest X-Ray Image to be tested, for Pneumonia Detection")
        file = st.file_uploader("Please upload an image file", type=["jpg", "png","jpeg"])
        
        if file is None:
            st.text("Please upload an image file")
        else:
            image = Image.open(file)
            st.image(image, use_column_width=True)
            prediction1 = covid(image, model1)
            if prediction1 <= 0.5:
                st.subheader("Result: No Pnemonia Detected")
            else:
                st.subheader("Result: Pnemonia Detected")
    
    elif choice == 'Diabetes detection':
        image = Image.open('diabetesimg.png')
        st.image(image, use_column_width=True,format='PNG')
        st.subheader("Diabetes detection:")
        
        st.write("This is Diabetes Detection section, Please Enter the genuine inputs, to predict diabetes")
        
        Pregnancies = st.slider('Pregnancies:', 0, 15)
        Glucose = st.slider('Glucose Level( in concentration a 2 hours in an oral glucose tolerance test ):', 50, 200)
        BloodPressure = st.slider('Blood Pressure ( in mm Hg ):',30, 130)
        SkinThickness = st.slider('Skin Thickness ( Triceps skin fold thickness (mm) ):', 0, 100)
        Insulin =  st.slider('Insulin Level ( in mu U/ml ):', 10, 850)
        BMI = st.slider('BMI ( weight in kg/(height in m)^2 ):', 20, 60)
        DiabetesPedigreeFunction =  st.slider('Diabetes Pedigree Function:' , 0.0, 1.20)
        Age = st.slider('Age:', 20, 100)
        data = {'Pregnancies': Pregnancies,
                'Glucose': Glucose,
                'BloodPressure': BloodPressure,
                'SkinThickness': SkinThickness,
                'Insulin': Insulin,
                'BMI': BMI,
                'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
                'Age': Age}
        features = pd.DataFrame(data, index=[0])
        pred_proba = model2.predict_proba(features)
        pred_proba = pred_proba.round(2)
        
        st.subheader('Prediction Percentages:')
        
        st.write('**Probablity of Not have Diabetes ( in % )**:',pred_proba[0][0]*100 )
        st.write('**Probablity of having Diabetes ( in % )**:',pred_proba[0][1]*100 )

        
    elif choice == 'Covid-19 Symptom Detection':
        image = Image.open('covidimg.png')
        st.image(image, use_column_width=True,format='PNG')
        st.subheader("Covid-19 Symptom Detection:")
        
        st.write("This is Covid-19 Symptom Detection section, Please Enter the genuine inputs, to predict the Risk from Covid-19")
        
        age = st.slider('Age', 0, 100)
        gender = st.selectbox("Gender? 0:Male or 1:Female",["0","1"])
        bodytemperature = st.slider('Body Temperature:',96, 106)
        DryCough = st.selectbox("Are you having Dry Cough?",["0","1"]) 
        sourthroat = st.selectbox("Are you having Sore Throat?",["0","1"])
        weakness = st.selectbox("Are you having Weakness?",["0","1"])
        breathingproblem = st.selectbox("Are you having Breathing problems?",["0","1"])
        diabetes = st.selectbox("Are you having Diabetes?",["0","1"])
        drowsiness = st.selectbox("Are you having Drowsiness?",["0","1"])
        travelhistory = st.selectbox("Are you having Travel History?",["0","1"])
        data = {'age': age,
                'gender': gender,
                'bodytemperature': bodytemperature,
                'DryCough': DryCough,
                'sourthroat': sourthroat,
                'weakness': weakness,
                'breathingproblem': breathingproblem,
                'diabetes': diabetes,
                'drowsiness': drowsiness,
                'travelhistory': travelhistory}
                
        features = pd.DataFrame(data, index=[0])
        pred_prob = model3.predict_proba(features)
        pred_prob = pred_prob.round(2)
        
        st.subheader('Prediction Probability')
        
        st.write('**Probablity of Low Risk ( in % )**:',pred_prob[0][0]*100)
        st.write('**Probablity of Medium Risk ( in % )**:',pred_prob[0][1]*100)
        st.write('**Probablity of High Risk ( in % )**:',pred_prob[0][2]*100)
        
    elif choice == 'Home':
        st.header("Welcome to G. S. MediSafe Application!")
        st.header("Details:")
        st.write("This Objective of G. S. MediSafe Application, is to deploy this Application in remote areas and help out the poor and helpless people with the medical checkup, which normally costs the Huge amount for Reports.")
        st.header("Overview:")
        st.write("We see lot of people are suffering a lot without the help of the proper medical checkup, Also in most of the cases many cases arise a situation which may lead to dealth due to lack of timely medical checkup, which was main motive to come up with G. S. MediSafe Application.")
        st.write("This G.S MediSafe Application serves the following Sections:")
        st.write("1. Malaria Detection")
        st.write("2. Pneumonia Detection")
        st.write("3. Diabetes Detection, and")
        st.write("4. Covid-19 Symptom Detection")
        st.header("Malaria Detection:")
        st.write("Malaria is a life-threatening disease caused by parasites that are transmitted to people through the bites of infected mosquitoes. This Application deals with diagnosis process which enable accurate diagnosis of the disease and hence holds the promise of delivering reliable health-care to resource-scarce areas.")
        st.write("This Malaria detection section, provides accurate predictions to detect malaria, by inserting Image Cell of patient, which detects whether person is having Malaria or Not!")
        st.header("Pneumonia Detection:")
        st.write("Pneumonia is a life-threatening infectious disease affecting one or both lungs in humans commonly caused by bacteria called Streptococcus pneumoniae. One in three deaths in India is caused due to pneumonia as reported by World Health Organization (WHO). Chest X-Rays which are used to diagnose pneumonia need expert radiotherapists for evaluation. Thus, developing an automatic system for detecting pneumonia would be beneficial for treating the disease without any delay particularly in remote areas.")
        st.write("Also, as we know, due to this Covid-19 Pandemic, doctors are busy dealing with many diverse patients, so this application can help doctors predicting patient has Pneumonia or Not, by just uploading the Chest X-ray of the patient and getting quick accurate predictions.")
        st.header("Diabetes Detection:")
        st.write("The objective of the Diabetes Detection Section is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset, taken from National Institute of Diabetes and Digestive and Kidney Diseases. Several constraints are placed on the selection of these instances from a larger database.")
        st.header("Covid-19 Symptom Detection:")
        st.write("The Objective of Covid-19 Symptom Detection Section, is to ensure, at what extent are we away from dangerous Corona Virus, this section takes Parameters included in dataset, and results what percentage of Low, Medium and High amount of risk the user is facing in a probablity percentage format. In any instance the High-risk percentage is more, it is suggested to visit the doctor as early as possible.")
        st.write("Thank you.")
        html_temp = """
        <div style="background-color:#010200;padding:6px">
        <h4 style="color:white;text-align:center;">Designed and Developed by: Gaurav Rajesh Sahani</h4>
        </div>
        """ 
        st.markdown(html_temp,unsafe_allow_html=True)
        
    elif choice == 'Contact Information':
        st.subheader("G. S. MediSafe Application GitHub Link:")
        st.markdown("Thank you for using G. S. MediSafe Application, here's the GitHub link for the code.")
        st.markdown("Github Link for the code : [Code Link](https://github.com/GauravSahani1417)")
        
        st.subheader("Connect with me:")
        st.markdown("Designed and Developed by [Gaurav Rajesh Sahani](https://www.linkedin.com/in/gaurav-sahani-6177a7179/)")
     
    elif choice == 'Disclaimer':
        st.subheader("Disclaimer about G. S. MediSafe Application:")
        st.write("Though the Results obtained in every section are Accurate, its not recommended to only rely on this application, since this G. S. MediSafe Application is a prototype, also healthcare domain is very sensitive as it directly deals with patient's health. It is recommended to validate every result obtained.")
        st.write("Thank you!")
        
        
st.set_option('deprecation.showfileUploaderEncoding', False)
if __name__ == '__main__':
        main()
            
                
                
                
                
                
                

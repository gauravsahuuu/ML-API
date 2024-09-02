import pickle
import streamlit as st
from streamlit_option_menu import option_menu


#loading models 
dia_model=pickle.load(open("C:\Users\PYARE LAL\Multiple_Disease_Prediction_Model\diabetese_trained_model.pkl",'rb'))
heart_model=pickle.load(open("C:\Users\PYARE LAL\Multiple_Disease_Prediction_Model\heart_trained_model.pkl",'rb'))

#sidebar
with st.sidebar:
    selected=option_menu('Multiple Disease Prediction System',
                         ['Diabetese Predicion',
                          'Heart Disease Prediction'],
                          icons=['activity','heart'],default_index=0)
    
#diabtese predition page
if (selected=='Diabetese Prediction'):

    #page title
    st.title('Diabetese Prediction using ML')

    #getting input from user
    col1,col2,col3=st.columns(3)

   with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
    with col2:
        Glucose = st.text_input("Glucose Level")
    with col3:
        BloodPressure = st.text_input("BloodPressure Value")
    with col1:
        SkinThickness = st.text_input("SkinThickness Value")
    with col2:
        Insulin = st.text_input("Insulin Value")
    with col3:
        BMI = st.text_input("BMI Value")
    with col1:
        DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction Value")
    with col2:
        Age = st.text_input("Age")

#code for prediction
diab_diagnosis=''

#creating button for prediction
if st.button('Diabetes Test Result'):
       diab_prediction=dia_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    if (diab_prediction[0]==1):
       diab_diagnosis='The person is DIABETIC'
    else:
       diab_diagnosis='The person is not DIABETIC'

    st.success(diab_diagnosis)  

    
#Heart Disease Prediction
if (selected=='Heart Disease'):

    #page title
    st.title('Heart Disease Prediction using ML')

    #getting input from user
    col1,col2,col3=st.columns(3)

   with col1:
        age = st.text_input("Age")
    with col2:
        sex = st.text_input("Sex")
    with col3:
        cp = st.text_input("Chest Pain Types")
    with col1:
        trestbps = st.text_input("Resting Blood Pressure")
    with col2:
        chol = st.text_input("Serum Cholestroal in mg/dl")
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

#code for prediction
heart_diagnosis=''

#creating button for prediction
if st.button('Heart Disease Test Result'):
       heart_prediction=heart_model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    if (heart_prediction[0]==1):
       heart_diagnosis='The person has heart disease'
    else:
       heart_diagnosis='The person does not have heart disease'

    st.success(heart_diagnosis)  

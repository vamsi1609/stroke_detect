import streamlit as st
import pickle
import joblib
import numpy as np
st.title(
    'Stroke Detection via Machine Learning'
)
name = st.text_input('Input your name')
gender = st.radio("Gender",('Female', 'Male'))
gender = 0 if gender =='Female' else 1

age = st.slider("Age",1,100,20)
hypertension = st.radio("Hypertension",('Yes', 'No'))
hypertension = 0 if hypertension =='No' else 1

heart_disease = st.radio("Heart Disease",('Yes', 'No'))
heart_disease = 0 if heart_disease =='No' else 1

ever_married = st.radio("Marital Status",('Yes', 'No'))
ever_married = 0 if ever_married =='No' else 1

work_type = st.selectbox("Work Type", ("Govt_job","Never_worked","Private","Self_employed","children"))
if(work_type=="Govt_job"):
    work_type=0
elif(work_type=="Never_worked"):
    work_type=1
elif(work_type=="Private"):
    work_type=2
elif(work_type=="Private"):
    work_type=3
else:
    work_type=4


Residence_type = st.radio("Residance type",('Rural', 'Urban'))
Residence_type = 0 if Residence_type =='Rural' else 1

avg_glucose_level = st.slider("Average Glucose Level", 40.00, 290.00)
bmi = st.slider("BMI",5.0,110.0)
smoking_status = st.selectbox("Smoking Status",('Unknown', 'formerly smoked', 'never smoked', 'smokes'))

if(smoking_status=="Unknown"):
    smoking_status=0
elif(smoking_status=="formerly smoked"):
    smoking_status=1
elif(smoking_status=="never smoked"):
    smoking_status=2
else:
    smoking_status=3

x_pred = np.array([[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]])
Filename = 'models/model.joblib'
with open(Filename, 'rb') as file:  
    KNN_Model = joblib.load(file)

y = KNN_Model.predict(x_pred)

y = "No" if y==0 else "Yes"

if st.button('Predict'):
    if (y=="No"):
        st.write(f"""
            ## Hi {name}, you might not have a stroke
        """)
    else:
        st.write(f"""
            ## Hi {name}, sorry you might get or might have had a stroke
        """)

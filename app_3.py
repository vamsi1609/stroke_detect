import streamlit as st
import pickle
import numpy as np
from PIL import Image
import pandas as pd

st.title(
    'Stroke Detection via Machine Learning'
)

st.write(" __________________________ ")

data = pd.read_csv('data\raw\healthcare-dataset-stroke-data.csv')


name = st.sidebar.text_input('Input your name')
gender = st.sidebar.radio("Gender",('Female', 'Male'))
gender = 0 if gender =='Female' else 1

age = st.sidebar.slider("Age",1,100,20)
hypertension = st.sidebar.radio("Hypertension",('Yes', 'No'))
hypertension = 0 if hypertension =='No' else 1

heart_disease = st.sidebar.radio("Heart Disease", ('Yes', 'No'))
heart_disease = 0 if heart_disease =='No' else 1

ever_married = st.sidebar.radio("Marital Status", ('Yes', 'No'))
ever_married = 0 if ever_married =='No' else 1

work_type = st.sidebar.selectbox("Work Type", ("Govt_job", "Never_worked", "Private", "Self_employed", "children"))
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


Residence_type = st.sidebar.radio("Residance type", ('Rural', 'Urban'))
Residence_type = 0 if Residence_type =='Rural' else 1

avg_glucose_level = st.sidebar.slider("Average Glucose Level", 40.00, 290.00)
bmi = st.sidebar.slider("BMI", 5.0, 110.0)
smoking_status = st.sidebar.selectbox("Smoking Status", ('Unknown', 'formerly smoked', 'never smoked', 'smokes'))

if(smoking_status=="Unknown"):
    smoking_status=0
elif(smoking_status=="formerly smoked"):
    smoking_status=1
elif(smoking_status=="never smoked"):
    smoking_status=2
else:
    smoking_status=3

x_pred = np.array([[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]])
Filename = 'models/RF_KNN_Model.pkl'
with open(Filename, 'rb') as file:  
    KNN_Model = pickle.load(file)

y = KNN_Model.predict(x_pred)

y = "No" if y==0 else "Yes"

if st.sidebar.button('Predict'):
    if (y=="No"):
        st.write(f"""
            ## Hi {name}, you are at a lower risk of having a stroke.
        """)
        image = Image.open('happy.jpg')
        st.image(image)
    else:
        st.write(f"""
            ## Hi {name}, sorry you might get or might have had a stroke
        """)
        image = Image.open('worry.jpg')
        st.image(image)



data.drop(columns=['id'], inplace=True)
data['bmi'].fillna(np.round(data['bmi'].mean(), 1), inplace=True)
data = data[data['gender'] != 'Other']

data['age_norm'] = (data['age']-data['age'].min()) / (data['age'].max()-data['age'].min())

data['average_glucose_level_norm'] = (data['avg_glucose_level']-data['avg_glucose_level'].min()) / (data['avg_glucose_level'].max()-data['avg_glucose_level'].min())

data['bmi_norm'] = (data['bmi']-data['bmi'].min()) / (data['bmi'].max()-data['bmi'].min())

data['age_binned'] = pd.cut(data['age'], np.arange(0, 91, 5))

data['average_glucose_level_binned'] = pd.cut(data['avg_glucose_level'], np.arange(0, 301, 10))

data['bmi_binned'] = pd.cut(data['bmi'], np.arange(0, 101, 5))


def get_100_percent_stacked_bar_chart(column, width=0.5):
    # Get the count of records by column and stroke
    df_breakdown = data.groupby([column, 'stroke'])['age'].count()
    # Get the count of records by gender
    df_total = data.groupby([column])['age'].count()
    # Get the percentage for 100% stacked bar chart
    df_pct = df_breakdown / df_total * 100
    # Create proper DataFrame's format
    df_pct = df_pct.unstack()
    return df_pct.plot.bar(stacked=True, figsize=(6, 6), width=width)


def get_stacked_bar_chart(column):
    # Get the count of records by column and stroke
    df_pct = data.groupby([column, 'stroke'])['age'].count()
    # Create proper DataFrame's format
    df_pct = df_pct.unstack()
    return df_pct.plot.bar(stacked=True, figsize=(6, 6), width=1)


get_stacked_bar_chart('age_binned')
get_100_percent_stacked_bar_chart('age_binned', width=0.9)

get_stacked_bar_chart('bmi_binned')
get_100_percent_stacked_bar_chart('bmi_binned', width=0.9)

get_stacked_bar_chart('average_glucose_level_binned')
get_100_percent_stacked_bar_chart('average_glucose_level_binned', width=0.9)

get_100_percent_stacked_bar_chart('hypertension')
get_100_percent_stacked_bar_chart('heart_disease')

get_100_percent_stacked_bar_chart('gender')
get_100_percent_stacked_bar_chart('Residence_type')

get_100_percent_stacked_bar_chart('work_type')
# data.groupby(['work_type'])[['age']].agg(['count', 'mean'])

get_100_percent_stacked_bar_chart('ever_married')
# data.groupby(['ever_married'])[['age']].agg(['count', 'mean'])

st.pyplot()
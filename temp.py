# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("diabetes.csv")
df.head()

df.info()
df.describe()
df.value_counts('Outcome').plot(kind='bar')
sns.heatmap(df.corr(), annot=True)
y = df['Outcome']#get y
X = df.drop('Outcome', axis=1)#get x
random_state=26
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=random_state)#split dataset to train and test
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier


estimators = [
    ('rf', RandomForestClassifier(n_estimators=12, random_state=random_state)),
    ('svm', SVC(C=11, random_state=random_state))
]

combined_model = StackingClassifier(estimators=estimators, final_estimator=SVC(C=11, random_state=random_state))
combined_model.fit(X_train, y_train)
combined_model.score(X_test, y_test)
from sklearn.metrics import confusion_matrix, classification_report

y_pred = combined_model.predict(X_test)#predict test data
cm = confusion_matrix(y_test, y_pred) #confusion matrix
target_names = ['0','1']

ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(target_names); ax.yaxis.set_ticklabels(target_names);

print("Report model\n")
print(classification_report(y_pred, y_test, target_names=target_names))

import pickle

#DataSet





import streamlit as st

st.write(""" # Diabetes Prediction App""")

st.sidebar.header('User Input Values')

def user_input_features():
    global Preg_input
    Preg_input= st.sidebar.text_input("Please Input number of Pregnancies:",6)
    global Glucose_input 
    Glucose_input = st.sidebar.text_input("Glucose Level:",148)
    global BloodPressure_input 
    BloodPressure_input = st.sidebar.text_input("Input Blood Pressure",72)    
    global SkinThickness_input
    SkinThickness_input = st.sidebar.text_input("Input Skin Thickness",35)
    global Insulin_input 
    Insulin_input = st.sidebar.text_input("Please Enter Insulin Level",0)
    global BMI_input 
    BMI_input = st.sidebar.text_input("Please Enter BMI",33.6)
    
    global Age_input 
    Age_input = st.sidebar.text_input("Please Enter Age",50)
    
    data = {
        "Pregnancies": Preg_input,
        "Glucose": Glucose_input,
        "Blood Pressure": BloodPressure_input,
        "Skin Thickness": SkinThickness_input,
        "Insulin": Insulin_input,
        "BMI": BMI_input,
       
        "Age": Age_input,
        }
    
    
    
    features = pd.DataFrame(data, index=[0])
    
    return features

sldf = user_input_features()

print("this is pregnancies", Preg_input)



# load
scaler = pickle.load(open("scaler.pkl", 'rb'))

with open('model.pkl', 'rb') as f:
    combined_model = pickle.load(f)


Pregnancies=Preg_input
Glucose=Glucose_input
BloodPressure=BloodPressure_input
SkinThickness=SkinThickness_input
Insulin=Insulin_input
BMI=BMI_input
DiabetesPedigreeFunction=0.627
Age=Age_input


input_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
input_data = scaler.transform(input_data)
output = combined_model.predict(input_data)[0]

def Output_func():
    if output==1:
        result = "This person has high chances of having diebetes!"
    else:
        result = "This is a healthy person!"
    return result

result1 = Output_func()

st.subheader(result1)
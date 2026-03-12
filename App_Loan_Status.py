import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import streamlit as st
st.title("🏦 Loan Status Prediction System")
loan_dataset = pd.read_csv('loan.csv')
loan_dataset=loan_dataset.dropna()
loan_dataset.replace({'Loan_Status':{'N':0,'Y':1}},inplace=True)
loan_dataset=loan_dataset.replace(to_replace='3+',value=4)
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Female':0,'Male':1},'Self_Employed':{'No':0,'Yes':1},'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Not Graduate':0,'Graduate':1}},inplace=True)
X=loan_dataset.drop(columns=["Loan_ID","Loan_Status"],axis=1)
Y=loan_dataset['Loan_Status']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)
classifier=svm.SVC(kernel="linear")
classifier.fit(X_train,Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
st.sidebar.write("Accuracy on training data = ", training_data_accuracy)
st.sidebar.write("Accuracy on test data = ", test_data_accuracy)
st.header("Enter Applicant Details")
Gender = st.selectbox("Gender", [0,1])
Married = st.selectbox("Married", [0,1])
Dependents = st.number_input("Dependents",0,4)
Education = st.selectbox("Education (0=Not Graduate,1=Graduate)",[0,1])
Self_Employed = st.selectbox("Self Employed",[0,1])
ApplicantIncome = st.number_input("Applicant Income")
CoapplicantIncome = st.number_input("Coapplicant Income")
LoanAmount = st.number_input("Loan Amount")
Loan_Amount_Term = st.number_input("Loan Term")
Credit_History = st.selectbox("Credit History",[0,1])
Property_Area = st.selectbox("Property Area (0=Rural,1=Semiurban,2=Urban)",[0,1,2])
input_data = np.array([[Gender,Married,Dependents,Education,Self_Employed,
ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,
Credit_History,Property_Area]])
if st.button("Predict Loan Status"):
    prediction = classifier.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

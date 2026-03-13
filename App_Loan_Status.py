import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
col1, col2 = st.columns(2)
with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Gender = 1 if Gender == 'Male' else 0
    Married = st.selectbox("Married", ["Yes","No"])
    Married = 1 if Married == "Yes" else 0
    Dependents = st.number_input("Dependents",0,4)
    Education = st.selectbox("Education", ["Graduate", "Not graduate"])
    Education = 1 if Education == "Graduate" else 0
    Self_Employed = st.selectbox("Self Employed",["Yes","No"])
    Self_Employed = 1 if Self_Employed == "Yes" else 0
with col2:
    ApplicantIncome = st.number_input("Applicant Income")
    CoapplicantIncome = st.number_input("Coapplicant Income")
    LoanAmount = st.number_input("Loan Amount")
    Loan_Amount_Term = st.number_input("Loan Term")
    Credit_History = st.selectbox("Credit History",["Yes","No"])
    Credit_History = 1 if Credit_History == "Yes" else 0
    Property_Area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
    if Property_Area == "Rural":
        Property_Area = 0
    elif Property_Area == "Semiurban":
        Property_Area = 1
    else:
        Property_Area = 2
input_data = np.array([[Gender,Married,Dependents,Education,Self_Employed,
ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,
Credit_History,Property_Area]])
if st.button("Predict Loan Status"):
    prediction = classifier.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
prob_classifier = svm.SVC(kernel = "linear", probability = True)
prob_classifier.fit(X_train, Y_train)
if st.button("Show Approval Probability"):
    probability = prob_classifier.predict_proba(input_data)
    approval_prob = probability[0][1]
    rejection_prob = probability[0][0]
    st.subheader("📊 Loan Approval Probability Meter")
    st.progress(int(approval_prob * 100))
    st.write("✅ Approval Probability:", round(approval_prob * 100, 2), "%")
    st.write("❌ Rejection Probability:", round(rejection_prob * 100, 2), "%")
st.markdown('---')
st.header("📈 Loan Data Analytics Dashboard")
analysis_option = st.selectbox("Select Analysis",["Loan Status Distribution", "Applicant Income Distribution","Loan Amount Distribution","Property Area vs Loan Approval"])
if analysis_option == "Loan Status Distribution":
    fig, ax = plt.subplots()
    sns.countplot(x = 'Loan_Status', data = loan_dataset, ax = ax)
    st.pyplot(fig)
elif analysis_option == "Applicant Income Distribution":
    fig, ax = plt.subplots()
    sns.histplot(loan_dataset['ApplicantIncome'], bins = 30, kde = True, ax = ax)
    st.pyplot(fig)
elif analysis_option == "Loan Amount Distribution":
    fig, ax = plt.subplots()
    sns.histplot(loan_dataset['LoanAmount'], bins = 30, kde = True, ax = ax)
    st.pyplot(fig)
else:
    fig, ax = plt.subplots()
    sns.countplot(x = 'Property_Area', hue = 'Loan_Status', data = loan_dataset, ax = ax)
    st.pyplot(fig)

    
    


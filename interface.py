import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
heart_data = pd.read_csv("heart.csv")
X = heart_data.drop(columns='target', axis=1)
df=pd.DataFrame(heart_data)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
st.title("Heart disease prediction system")
st.line_chart(df)
st.sidebar.title("Enter input values ")
form = st.sidebar.form(key='my_form')
age=form.number_input(label="Age")
gender=form.radio("Gender",["Male","Female"])
cp=form.number_input(label="Chest Pain")
bp=form.number_input(label = "Enter Rest blood pressure")
chol=form.number_input(label = "Enter Serum cholesterol")
fbs=form.number_input(label = "Enter Fasting blood sugar")
restecg=form.number_input(label = "Enter Rest electrocardiograph")
thalch=form.number_input(label = "Enter MaxHeart rate")
exang=form.number_input(label = "Enter Exercise-induced angina")
oldpeak=form.number_input(label = "Enter ST depression")
slope=form.number_input(label = "Enter slope")
ca=form.number_input(label = "Enter No. of vessels ")
thal=form.number_input(label = "Enter thalassemia")
Select=form.selectbox("Algorithm",["RANDOM FOREST","KNN","NAIVE BAYES","DECISION TREE","LOGISTIC REGRESSION","SVM"])
submit_button = form.form_submit_button(label='Submit')
if gender=="Male":
    gender=1
else:
    gender=0
if submit_button:
    if Select=="RANDOM FOREST":
        from sklearn.ensemble import RandomForestClassifier
        model= RandomForestClassifier()
        model.fit(X_train,Y_train)
        X_train_prediction = model.predict(X_train)
        training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
        input_data = (int(age),int(gender),int(cp),int(bp),int(chol),int(fbs),int(restecg),int(thalch),int(exang),int(oldpeak),int(slope),int(ca),int(thal))
        input_data_as_numpy_array= np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model.predict(input_data_reshaped)
        if prediction[0]==1:
            st.subheader("Result: Positive")
            st.write("Accurracy: ","%.2f"%(training_data_accuracy*100),"%")
        else:
            st.subheader("Result: Negative")
            st.write("Accurracy: ","%.2f"%(training_data_accuracy*100),"%")
    elif Select=="KNN":
        from sklearn.neighbors import KNeighborsClassifier
        model= KNeighborsClassifier()
        model.fit(X_train,Y_train)
        X_train_prediction = model.predict(X_train)
        training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
        input_data = (int(age),int(gender),int(cp),int(bp),int(chol),int(fbs),int(restecg),int(thalch),int(exang),int(oldpeak),int(slope),int(ca),int(thal))
        input_data_as_numpy_array= np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model.predict(input_data_reshaped)
        if prediction[0]==1:
            st.subheader("Result: Positive")
            st.write("Accurracy: ","%.2f"%(training_data_accuracy*100),"%")
        else:
            st.subheader("Result: Negative")
            st.write("Accurracy: ","%.2f"%(training_data_accuracy*100),"%")        
    elif Select=="NAIVE BAYES":
        from sklearn.naive_bayes import GaussianNB
        model=GaussianNB()
        model.fit(X_train,Y_train)
        X_train_prediction = model.predict(X_train)
        training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
        input_data = (int(age),int(gender),int(cp),int(bp),int(chol),int(fbs),int(restecg),int(thalch),int(exang),int(oldpeak),int(slope),int(ca),int(thal))
        input_data_as_numpy_array= np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model.predict(input_data_reshaped)
        if prediction[0]==1:
            st.subheader("Result: Positive")
            st.write("Accurracy: ","%.2f"%(training_data_accuracy*100),"%")
        else:
            st.subheader("Result: Negative")
            st.write("Accurracy: ","%.2f"%(training_data_accuracy*100),"%")
    elif Select=="DECISION TREE":
        from sklearn.tree import DecisionTreeClassifier
        model= DecisionTreeClassifier()
        model.fit(X_train,Y_train)
        X_train_prediction = model.predict(X_train)
        training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
        input_data = (int(age),int(gender),int(cp),int(bp),int(chol),int(fbs),int(restecg),int(thalch),int(exang),int(oldpeak),int(slope),int(ca),int(thal))
        input_data_as_numpy_array= np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model.predict(input_data_reshaped)
        if prediction[0]==1:
            st.subheader("Result: Positive")
            st.write("Accurracy: ","%.2f"%(training_data_accuracy*100),"%")
        else:
            st.subheader("Result: Negative")
            st.write("Accurracy: ","%.2f"%(training_data_accuracy*100),"%")
    elif Select=="LOGISTIC REGRESSION":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(X_train, Y_train)
        X_train_prediction = model.predict(X_train)
        training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
        input_data = (int(age),int(gender),int(cp),int(bp),int(chol),int(fbs),int(restecg),int(thalch),int(exang),int(oldpeak),int(slope),int(ca),int(thal))
        input_data_as_numpy_array= np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model.predict(input_data_reshaped)
        if prediction[0]==1:
            st.subheader("Result: Positive")
            st.write("Accurracy: ","%.2f"%(training_data_accuracy*100),"%")
        else:
            st.subheader("Result: Negative")
            st.write("Accurracy: ","%.2f"%(training_data_accuracy*100),"%")
    elif Select=='SVM':
        from sklearn.svm import SVC
        model = SVC(C=1,random_state=1,kernel='linear')
        model.fit(X_train, Y_train)
        X_train_prediction = model.predict(X_train)
        training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
        input_data = (int(age),int(gender),int(cp),int(bp),int(chol),int(fbs),int(restecg),int(thalch),int(exang),int(oldpeak),int(slope),int(ca),int(thal))
        input_data_as_numpy_array= np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model.predict(input_data_reshaped)
        if prediction[0]==1:
            st.subheader("Result: Positive")
            st.write("Accurracy: ","%.2f"%(training_data_accuracy*100),"%")
        else:
            st.subheader("Result: Negative")
            st.write("Accurracy: ","%.2f"%(training_data_accuracy*100),"%")

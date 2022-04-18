from msilib.schema import Feature
from pyexpat import features
from random import random
from re import X
from numpy import column_stack
from sklearn import datasets
import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# make containers
header = st.container()
data_sets = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("app of ship")
    st.text("in this project we will work on ship data")

with data_sets:
    st.header("ship was shunk")
    st.text("in this project we will work on Titanic ship data")
    #import data
    df = sns.load_dataset("titanic")
    df = df.dropna()
    st.write(df.head(10))

    st.subheader("how much people?")
    st.bar_chart(df['sex'].value_counts())

    # other plots
    st.subheader("class difference")
    st.bar_chart(df['class'].value_counts())


with features:
    st.header("These are our app features")
    st.text("in this project we will work on ship data and add many features")
    st.markdown('1. **Feature 1:** This will tell us')
    st.markdown('2. **Feature 2:** This will tell us')


with model_training:
    st.header("How many men had been drowned?")            
    st.text("in this project we will work on ship data, in this we will up and down our parameters")
    # Making columns
    input, display = st.columns(2)

# in first columns is your selection point
max_depth = input.slider("How many people do you know", min_value=10, max_value= 100, value =20, step=5)

# n_estimatores
n_estimators = input.selectbox("how many should be there in a RF?", options=[50,100,200,300, 'No limit'])

# adding list of features
input.write(df.columns)


#input features from users
input_features = input.text_input('which features we should use')

# machine learning model

model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

# Here we put 1 condation
if n_estimators == 'No limit':
    model = RandomForestClassifier(max_depth=max_depth)
else:
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)    




#define X and y

X = df[[input_features]]
y = df[['fare']]

#fit our model
model.fit(X, y)
pred = model.predict(y)

# Display metrices

display.subheader("Mean absolute error of model is: ")
display.write(mean_absolute_error(y, pred))
display.subheader("Mean squared error of model is: ")
display.write(mean_squared_error(y, pred))
display.subheader("R squared score of model is: ")
display.write(r2_score(y, pred))
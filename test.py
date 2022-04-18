import streamlit as st
import seaborn as sns
st.header("This video is brought to you by AHSAN ZAMAN ")
st.text("Bhot acha laga sikh kr")

st.header("wow, bhot acha")

df = sns.load_dataset("iris")
st.write(df[['species', 'sepal_length', 'petal_length']].head(10))

st.bar_chart(df['sepal_length'])
st.line_chart(df['sepal_length'])

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
model_path = 'C:/Users/pc/Desktop/pycharm/hi/kmeans_model.pkl'
scaler_path = 'C:/Users/pc/Desktop/pycharm/hi/scaler.pkl'

with open(model_path, 'rb') as model_file:
    kmeans = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit app
st.title("Customer Support Data Clustering")

st.write("This app allows you to predict clusters based on the provided customer support data.")

# Input fields for numerical features
item_price = st.number_input("Item Price", min_value=0.0, value=0.0)
handling_time = st.number_input("Connected Handling Time", min_value=0.0, value=0.0)
csat_score = st.number_input("CSAT Score", min_value=0, value=0)

# Collect input data into DataFrame
input_data = pd.DataFrame({
    'Item_price': [item_price],
    'connected_handling_time': [handling_time],
    'CSAT Score': [csat_score]
})

# Predict button
if st.button("Predict"):
    # Preprocess the input data
    scaled_input_data = scaler.transform(input_data)

    # Predict cluster
    cluster = kmeans.predict(scaled_input_data)

    st.write(f"The predicted cluster is: {cluster[0]}")




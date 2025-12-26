import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Loading the Model 
with open("trained_model/XGB_Model.pk1", "rb") as f:
    model = pickle.load(f)

with open("scaler.pk1", "rb") as s:
    scaler = pickle.load(s)

st.title("Advanced Credit Card Fraud Detection Software")
st.write("Enter the transaction details below and click Predict")

# Input Details

feature_values = {}

st.subheader("PCA features (V1–V28)")
for i in range(1, 29):
    feature_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

Time = st.number_input("Time", value=0.0)
Amount = st.number_input("Amount", value=0.0)

# Button
if st.button("Predict Fraud"):

    # Create input array in correct order
    input_data = np.array([[
        Time,
        *[feature_values[f"V{i}"] for i in range(1, 29)],
        Amount
    ]])

    # Scale Amount
    input_data[:, -1] = scaler.transform(input_data[:, -1].reshape(-1, 1))

    prediction = model.predict(input_data)[0]

    # Output result
    if prediction == 1:
        st.error("🚨 Fraud Detected!")
    else:
        st.success("✅ Transaction is Normal")

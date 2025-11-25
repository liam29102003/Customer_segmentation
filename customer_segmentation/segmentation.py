import streamlit as st
import pandas as pd
import numpy as np
import joblib

kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Customer Segmentation App")
st.write("Enter customer details to predict their segment.")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income ", min_value=0, max_value=200000, value=50000)
total_spending = st.number_input("Total Spending", min_value=0, max_value=100000, value=10000) 
num_of_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_of_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=5)
num_web_visits_month = st.number_input("Number of Web Visits per Month", min_value=0, max_value=100, value=20)  
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

input_data = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'Total_Spending': [total_spending],
    'NumWebPurchases': [num_of_web_purchases],
    'NumStorePurchases': [num_of_store_purchases],
    'NumWebVisitsMonth': [num_web_visits_month],
    'Recency': [recency]
})

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    segment = kmeans.predict(input_scaled)
    
    if segment[0] == 0:
        segment_label = "Low-Spending Seniors"
    elif segment[0] == 1:
        segment_label = "Bargain-Hunting Young Adults"
    elif segment[0] == 2:
        segment_label = "Active Mid-Age Shoppers"
    elif segment[0] == 3:
        segment_label = "Big Spender Premium Customers"
    elif segment[0] == 4:
        segment_label = "Loyal High-Spending Customers"
    elif segment[0] == 5:
        segment_label = "Occasional Low-Spending Customers"
    else:
        segment_label = f"Segment {segment[0]}"
    
    st.success(f"The predicted customer segment is: {segment_label}")

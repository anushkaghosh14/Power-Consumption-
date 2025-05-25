# app.py

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model and scaler
model = pickle.load(open("xgb_model/xgb_model.pkl", "rb"))
#scaler = pickle.load(open("xgb_model/scaler.pkl", "rb"))

# Streamlit App
st.title("⚡ Power Consumption Predictor")
st.markdown("Upload a CSV file to predict `global_active_power`.")

# File uploader
uploaded_file = st.file_uploader("Upload your household power CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['weekday'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    df = pd.get_dummies(df, columns=['hour'], drop_first=True)

    for h in range(1, 24):
        col = f"hour_{h}"
        if col not in df.columns:
            df[col] = 0

    continuous_cols = ['global_reactive_power', 'voltage', 'global_intensity',
                       'sub_metering_1', 'sub_metering_2', 'sub_metering_3', 'day', 'weekday', 'month']
    #df[continuous_cols] = scaler.transform(df[continuous_cols])

    X = df.drop(['datetime', 'global_active_power'], axis=1, errors='ignore')
    preds = model.predict(X)

    df['Predicted Power (kW)'] = preds
    st.write("### Predictions")
    st.dataframe(df[['datetime', 'Predicted Power (kW)']].head(10))

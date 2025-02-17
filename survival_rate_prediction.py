# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 21:19:15 2025

@author: Lenovo
"""

import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import streamlit as st

# 1. Load Dataset
cycles_df = pd.read_csv("cycles.csv")
harvests_df = pd.read_csv("harvests.csv")

# 2. Data Cleaning & Preprocessing
harvests_df['total_harvested'] = harvests_df['size'] * harvests_df['weight']
harvest_summary = harvests_df.groupby('cycle_id')['total_harvested'].sum().reset_index()
cycles_df = cycles_df.rename(columns={'id': 'cycle_id'})
sr_df = cycles_df[['cycle_id', 'total_seed', 'area', 'target_cultivation_day']].merge(harvest_summary, on='cycle_id', how='left')
sr_df['survival_rate'] = sr_df['total_harvested'] / sr_df['total_seed']
sr_df.fillna(0, inplace=True)

# 3. Model Training
X = sr_df[['total_seed', 'area', 'target_cultivation_day']]
y = sr_df['survival_rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Save Model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# 5. Streamlit App
st.title("Shrimp Survival Rate Prediction")

st.sidebar.header("Input Features", divider=True)
total_seed = st.sidebar.number_input("Total Seed", min_value=1, value=50000)
area = st.sidebar.number_input("Pond Area (m²)", min_value=1.0, value=500.0)
target_cultivation_day = st.sidebar.number_input("Target Cultivation Day", min_value=1, value=90)

if st.sidebar.button("Predict Survival Rate"):
    features = np.array([[total_seed, area, target_cultivation_day]])
    prediction = model.predict(features)
    st.write(f"Predicted Survival Rate: {prediction[0]:.2f}")

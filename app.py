# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 12:32:20 2025

@author: maila
"""

import numpy as np
import pickle
import streamlit as st
import math

xGmodel=pickle.load(open('C:/Users/maila/Documents/xG/xGmodel.sav', 'rb'))

def calculateDistance(x, y):
    return math.sqrt((120.0 - x)**2 + (40.0 - y)**2)

def calculateAngle(x, y):
    leftPostCordinate = 40.0 - 7.32 / 2
    rightPostCordinate = 40.0 + 7.32 / 2
    try:
        angle = abs(
            math.atan2(rightPostCordinate - y, 120.0 - x) -
            math.atan2(leftPostCordinate - y, 120.0 - x)
        )
        return angle
    except:
        return 0.0

st.title("Expected Goals (xG) Predictor")
st.subheader("Input shot features to calculate xG")

col1, col2=st.columns(2)
with col1:
    location_x = st.slider("Location X", 0.0, 120.0, 100.0, step=0.5)
    location_y = st.slider("Location Y", 0.0, 80.0, 40.0, step=0.5)
    numberOfDefenders = st.slider("Number of Defenders Nearby", 0, 10, 2)
    goalkeeper_x = st.slider("Goalkeeper X", 100.0, 120.0, 118.0)
    
with col2:
    bodyPart = st.selectbox("Body Part", ["Right Foot", "Left Foot", "Head", "Other"])
    shotType = st.selectbox("Shot Type", ["Open Play", "Free Kick", "Penalty", "Corner"])
    underPressure = st.checkbox("Under Pressure?")
    goalkeeper_y = st.slider("Goalkeeper Y", 0.0, 80.0, 40.0)
    
def encodeBodyPart(bodyPart):
    return 1 if bodyPart == "Head" else 0 if bodyPart in ["Left Foot", "Right Foot"] else 2

shotTypeMap = {"Open Play": 0, "Free Kick": 1, "Penalty": 2, "Corner": 3}

bodyPartEncoded = encodeBodyPart(bodyPart)
shotTypeEncoded = shotTypeMap[shotType]
underPressureEncoded = int(underPressure)

distance = calculateDistance(location_x, location_y)
angle = calculateAngle(location_x, location_y)

input_features = np.array([[ 
    location_x,
    location_y,
    bodyPartEncoded,
    shotTypeEncoded,
    underPressureEncoded,
    numberOfDefenders,
    goalkeeper_x,
    goalkeeper_y,
    distance,
    angle
]])

if st.button("Predict xG"):
    xg_prob = xGmodel.predict_proba(input_features)[0][1]
    pred_goal = xGmodel.predict(input_features)[0]
    
    st.success(f"Predicted xG: **{xg_prob:.3f}**")
    st.info(f"Predicted outcome: {'Goal' if pred_goal == 1 else 'No Goal'}")
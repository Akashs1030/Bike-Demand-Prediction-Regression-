#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pickle


# In[4]:


model=pickle.load(open('bike.pkl','rb'))


# In[5]:


st.title("BIKE DEMAND PREDICTION")


# In[9]:


#Sidebar Inputs
def user_inputs():
    season = st.sidebar.selectbox("Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)", [1,2,3,4])
     # Year input as actual year
    yr = st.sidebar.selectbox("Year (0=2011, 1=2012)", [0,1])
    month_dict = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
    }
    month_name = st.sidebar.selectbox("Month", list(month_dict.keys()))
    mnth = month_dict[month_name]    
    hr = st.sidebar.slider("Hour", 0, 23, 12)
    holiday = st.sidebar.selectbox("Holiday (0=No, 1=Yes)", [0,1])
    weekday = st.sidebar.selectbox("Weekday (0=Sun to 6=Sat)", list(range(0,7)))
    workingday = st.sidebar.selectbox("Working Day (0=No, 1=Yes)", [0,1])
    weathersit = st.sidebar.selectbox("Weather Situation (1=Clear, 2=Mist, 3=Light Snow, 4=Heavy Rain)",[1, 2, 3, 4])
    temp = st.sidebar.slider("Temperature (Normalized)", 0.0, 1.0, 0.30, 0.01)
    hum = st.sidebar.slider("Humidity (Normalized)", 0.0, 1.0, 0.50, 0.01)
    windspeed = st.sidebar.slider("Windspeed (Normalized)", 0.0, 1.0, 0.10, 0.01)
    data = {'season': season,'yr': yr,'mnth': mnth,'hr': hr,'holiday': holiday,'weekday': weekday,'workingday': workingday,'weathersit': weathersit,
        'temp': temp,'hum': hum,'windspeed': windspeed}
    return pd.DataFrame(data, index=[0])


# In[10]:


#Get User Input
df = user_inputs()
st.subheader("Input Features")
st.dataframe(df, use_container_width=True)


# In[11]:


#Prediction
if st.button("Predict Bike Rentals"):
    prediction = model.predict(df)
    st.subheader("Predicted Bike Rentals")
    st.success(f"{int(prediction[0])}")


# In[ ]:





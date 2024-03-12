import streamlit as st
import pickle
import pandas as pd
import pickle as pk
import numpy as np

pipe = pd.read_pickle(open('pipe1.pkl','rb'))
df = pd.read_pickle(open('df1.pkl','rb'))

st.title('Laptop Predictor')

# Brand
Company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('Ram(in GB)', [2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of Laptop')

# touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768',
 '1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440',
 '2304x1440'])

# cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

# hdd
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,128,256,512,1024,2048])

gpu = st.selectbox('GPU',df['Gpu Brand'].unique())

os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    ppi =  None
    if touchscreen =='yes':
       touchscreen = 1
    else:
        touchscreen = 0

    if ips =='yes':
        ips = 1
    else:
        ips = 0

    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res**2) + (y_res**2))**0.5/screen_size
    query = np.array([Company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    st.title('The predicted price of this configuration is ' + str(int(np.exp(pipe.predict(query)[0]))))



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import yfinance as yf
yf.pdr_override()

startdate = '2012-01-01'
enddate = '2022-12-31'

st.title('Stock Price Forecasting and Visualization')
user_input  = st.text_input('Enter Stock Ticker', 'AMZN')

df = pdr.get_data_yahoo(user_input,start=startdate,end=enddate)
df.head()

st.subheader('Data From 2012 - 2022')
st.write(df.describe())

#visualisation
st.subheader('Closing Price vs Time Graph')
fig = plt.figure(figsize = (15,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time with 100MA Graph')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (15,6))
plt.plot(ma100,'r')
plt.plot(df.Close,'c')
st.pyplot(fig)

st.subheader('Closing Price vs Time with 100 & 200 days MA Graph')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (15,6))
plt.plot(ma100,'r')
plt.plot(ma200,'y')
plt.plot(df.Close,'c')
st.pyplot(fig)

 # 70/30 data split in training dataset 
data_training = pd.DataFrame(df['Close'][0:(int)(len(df)*0.70)])  
data_testing = pd.DataFrame(df['Close'][(int)(len(df)*0.70):(int)(len(df))]) 

#scale down data
scaler = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)

#Load LSTM model
model = load_model('keras_model.h5')

#Testing Part
past100_days = data_training.tail(100)
final_df = past100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i - 100 : i])
  y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

#scaling up predicted value
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Data Visualisation
st.subheader('Predicted Price Vs Actual Price Graph')
final_fig = plt.figure(figsize = (12, 6))
plt.plot(y_test, 'g', label = 'Actual Price')
plt.plot(y_predicted, 'r', label = 'Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(final_fig)
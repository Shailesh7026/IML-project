import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

start = '2010-01-01'
end = '2023-10-20'

st.write("TensorFlow version:", tf.__version__)

st.title('Stock Trend Prediction')

st.write('You can find stock tickers from this site : https://finance.yahoo.com/')

# Use yfinance to get the data
user_input = st.text_input('Enter Stock Ticker:', 'AAPL')
df = yf.download(user_input, start=start, end=end)


st.subheader(f'Data for {user_input} from 2010 to 2023')
st.write(df.describe())

st.subheader("Closing Price Time chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader("Closing Price Time chart with 100MA")
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df['Close'], 'b')
st.pyplot(fig)

data_training = df['Close'].loc[start:end].iloc[:int(len(df) * 0.7)]
data_testing = df['Close'].loc[start:end].iloc[int(len(df) * 0.7):]

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i - 100:i, 0])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)


model = tf.keras.models.load_model('./models/predict_stock_price_v2.keras')

st.write("Current working directory:", os.getcwd())
st.write("Files in the current directory:", os.listdir())

past_100_days = data_training.tail(100)
data_testing = data_testing.reset_index(drop=True)

final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df.values.reshape(-1, 1))

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i, 0])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
scale_factor = 1 / 0.02123255
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader("Predictions vs Original Stock price")
plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label="Predicted Price")
plt.legend()
st.pyplot(plt)

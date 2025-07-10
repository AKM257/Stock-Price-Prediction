import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
model = load_model("Stock.h5")
# --- App Title ---
st.title("ML Stock Price Predictor")

# --- Ticker Input ---
ticker = st.text_input("Enter Stock Ticker Symbol", value="AAPL")

# --- Load Data ---
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2010-01-01", end=datetime.date.today())
    df = df[['Close']]
    df.dropna(inplace=True)
    return df

df = load_data(ticker)
st.subheader("Latest Stock Data")
st.dataframe(df.tail())

# --- Scale Data ---
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df[['Close']])

# --- Predict Tomorrow ---
last_60 = scaled_close[-60:]                # shape: (60, 1)
X_input = np.expand_dims(last_60, axis=0)   # shape: (1, 60, 1)

# --- Load Model ---

# --- Predict Next Close Price ---
pred_scaled = model.predict(X_input)
predicted_price = scaler.inverse_transform([[pred_scaled[0][0]]])[0][0]

st.subheader("Predicted Next Closing Price")
st.success(f"${predicted_price:.2f}")

# --- Plot Last 100 Actual Prices and Predicted Line ---
st.subheader("Historical Closing Prices")
fig1, ax1 = plt.subplots()
df[-100:].plot(ax=ax1, legend=True, label="Actual Close")
ax1.axhline(predicted_price, color='red', linestyle='--', label="Predicted Next Close")
ax1.legend()
st.pyplot(fig1)

# --- Prepare Data for Full Test Prediction ---
X, y = [], []
for i in range(60, len(scaled_close)):
    X.append(scaled_close[i-60:i])
    y.append(scaled_close[i])
X = np.array(X)
y = np.array(y)

# Split into test set (20%)
split = int(len(X) * 0.8)
X_test = X[split:]
y_test = y[split:]

# --- Predict on Test Set ---
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# --- Plot Real vs Predicted ---
st.subheader("Real vs Predicted Prices (Test Set)")
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(real_prices, label='Real Prices')
ax2.plot(predicted_prices, label='Predicted Prices')
ax2.set_title("Real vs Predicted Closing Prices")
ax2.set_xlabel("Time")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

# --- Evaluate MSE ---
mse = mean_squared_error(real_prices, predicted_prices)
st.write(f"Mean Squared Error (MSE): {mse:.4f}")

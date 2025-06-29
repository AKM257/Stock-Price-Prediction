
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
model = joblib.load('Stock_prediction.pkl')


st.title(" AI Stock Price Prediction Tool")

stock_symbol = st.text_input(" Enter a Stock Symbol (e.g., GOOG, AAPL, MSFT)", value="GOOG").upper()

def get_stock_data(stock):
    end_date = datetime.now()
    start_date = datetime(end_date.year - 20, 1, 1)

    st.info(f"Fetching historical data for {stock} from {start_date.year} to {end_date.year}...")
    df_data = yf.download(stock, start=start_date, end=end_date)
    if 'Close' not in df_data.columns:
        st.warning(f"No 'Close' price data found for '{stock}'. Please check the symbol.")
        return pd.DataFrame()
    df_data = df_data[['Close']].copy()
 
    df_data.dropna(inplace=True)
    return df_data
stock_df = get_stock_data(stock_symbol)

if stock_df.empty:
    st.stop()
st.subheader(f" Latest Available Historical Data for {stock_symbol}")
st.dataframe(stock_df.tail())
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_close = scaler.fit_transform(stock_df[['Close']])

last_60_days = scaled_close[-60:]

input_array = np.expand_dims(last_60_days, axis=0)
predicted_scaled_value = model.predict(input_array)

predicted_price = scaler.inverse_transform([[predicted_scaled_value[0][0]]])[0][0]
st.subheader(" Predicted Next Trading Day's Closing Price")
st.success(f"Estimated Price for {stock_symbol}: **${predicted_price:.2f}**")

st.subheader(f"{stock_symbol} Closing Prices: History & Forecast")
fig1, ax1 = plt.subplots(figsize=(10, 6))
stock_df[-100:].plot(ax=ax1, label="Actual Recent Close Prices", color='blue')
ax1.axhline(predicted_price, color='red', linestyle='--', label=f"Predicted Next Close: ${predicted_price:.2f}")

ax1.set_title(f"{stock_symbol} Recent Closing Prices and Next Day Prediction")
ax1.set_xlabel("Date")
ax1.set_ylabel("Closing Price ($)")
ax1.legend() # Display the legend to identify the lines.
st.pyplot(fig1) # Show the plot in the Streamlit app.


st.subheader("Model Performance: Real vs. Predicted (Back-Test)")
st.write("This chart shows how well the model would have predicted past stock prices based on historical data.")
X_seq, y_seq = [], []
for i in range(60, len(scaled_close)):
    X_seq.append(scaled_close[i-60:i]) 
    y_seq.append(scaled_close[i]) 
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

split_index = int(len(X_seq) * 0.7) 
X_test = X_seq[split_index:]
y_test = y_seq[split_index:] 
test_preds_scaled = model.predict(X_test)


pred_prices = scaler.inverse_transform(test_preds_scaled) 
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(actual_prices, label="Real Prices", color="blue", linewidth=1.5)
ax2.plot(pred_prices, label="Predicted Prices", color="orange", linestyle='--', alpha=0.7, linewidth=1.5)
ax2.set_title(f"{stock_symbol} Real vs. Predicted Stock Prices (Back-Test)")
ax2.set_xlabel("Time Step (Days in Test Set)")
ax2.set_ylabel("Price ($)")
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6) 
st.pyplot(fig2)

mse_val = mean_squared_error(actual_prices, pred_prices)
st.metric("Mean Squared Error (MSE)", f"{mse_val:.4f}", help="Lower values indicate a better fit between actual and predicted prices.")

st.markdown("---")
st.markdown("This tool provides predictions based on historical patterns and a pre-trained model. Stock prices are highly volatile and actual future prices may vary significantly.")
st.markdown("---")

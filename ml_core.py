import yfinance as yf
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Download data
def get_data(symbol="AAPL", period="2y"):
    df = yf.download(symbol, period=period)
    df = df.dropna()
    return df

# 2. Create features from last N closing prices
def make_supervised(df, lookback=5):
    # 1D array of closes
    closes = df["Close"].to_numpy(dtype=float)   # shape: (N,)

    X, y = [], []
    for i in range(len(closes) - lookback):
        window = closes[i:i+lookback]           # shape: (lookback,)
        X.append(window)
        y.append(closes[i+lookback])            # scalar

    X = np.array(X)                             # (n_samples, lookback) OR (n_samples, lookback, 1)
    y = np.array(y)                             # (n_samples,) OR (n_samples, 1)

    X = X.reshape(X.shape[0], lookback)         # (n_samples, lookback)
    y = y.reshape(-1)                           # (n_samples,)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y



# 3. Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # keep time order
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    mse = mean_squared_error(y_test, preds := model.predict(X_test))
    rmse = sqrt(mse)
    print(f"Test RMSE: {rmse:.2f}")

    return model

# 4. Predict next-day price from most recent data
def predict_next_price(model, df, lookback=5):
    last_closes = df["Close"].values[-lookback:]
    X_last = last_closes.reshape(1, -1)
    next_price = model.predict(X_last)[0]
    return float(next_price)

def make_signal(current_price, predicted_price, low_th=0.97, high_th=1.03):
    """
    Decide whether to BUY, SELL, or HOLD based on the ratio of
    current price to predicted next-day price.
    low_th, high_th are thresholds around 1.0 (100%).
    """
    ratio = current_price / predicted_price  # >1 => current is expensive

    if ratio < low_th:
        return "BUY", f"Price is {ratio*100:.1f}% of predicted — looks undervalued."
    elif ratio > high_th:
        return "SELL", f"Price is {ratio*100:.1f}% of predicted — looks overvalued."
    else:
        return "HOLD", f"Price is close to predicted — stay neutral."


if __name__ == "__main__":
    symbol = "AAPL"
    df = get_data(symbol)
    X, y = make_supervised(df, lookback=5)
    model = train_model(X, y)
    pred_next = predict_next_price(model, df, lookback=5)
    current_price = float(df["Close"].iloc[-1])

    print(f"Current price: {current_price:.2f}")
    print(f"Predicted next close: {pred_next:.2f}")
    signal, explanation = make_signal(current_price, pred_next)
    print(f"Signal: {signal}")
    print(explanation)

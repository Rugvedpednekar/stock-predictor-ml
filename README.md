# StockPredictor — Full-Stack Machine Learning Stock Forecasting App
A full-stack web application that predicts next-day stock prices using Machine Learning.
The project combines:

 Price history (yfinance)

 Random Forest regression model

 Time-series feature engineering

 Buy / Sell / Hold trading signal

 Modern responsive UI (Tailwind CSS)

 REST API backend (Flask)

 Run Locally
1. Install dependencies
pip install -r requirements.txt

2. Start backend
py app.py

3. Open app in browser
http://127.0.0.1:5000/

🧠 How the Model Works

The ML pipeline:

1. Download historical price data

Using yfinance (2 years by default).

2. Convert to supervised learning

Example (lookback = 5 days):

Input  (X): [p−5, p−4, p−3, p−2, p−1]
Output (y): p_today

3. Train Random Forest

200 decision trees

Learns patterns in 5-day sequences

Predicts next-day closing price

4. Evaluate model

RMSE printed in console.

5. Generate signals

Compares:

current_price vs predicted_price


Produces:

BUY

SELL

HOLD

explanation message

🔌 API Endpoints
Predict a single stock
GET /predict?symbol=AAPL


Response example:

{
  "symbol": "AAPL",
  "current_price": 173.42,
  "predicted_next_close": 169.88,
  "signal": "SELL",
  "message": "Price is 102.1% of predicted — looks overvalued.",
  "history": [
    {"date": "2025-01-12", "close": 170.12},
    ...
  ]
}

Predict multiple stocks
GET /predict_batch?symbols=AAPL,TSLA,MSFT

🏗️ Project Structure
.
├── app.py                  # Flask backend + API
├── ml_core.py              # Machine Learning logic
├── templates/
│   └── index.html          # Main frontend UI
├── static/                 # Optional assets
└── README.md

🎨 Frontend Features

Tailwind CSS (CDN)

Modern responsive design

Search bar with real-time predictions

Price history chart (last 60 days)

Color-coded prediction results

Model confidence display

Mobile-friendly interface

🔧 Tech Stack
Backend

Python

Flask

yfinance

scikit-learn (RandomForestRegressor)

Frontend

HTML

Tailwind CSS

JavaScript

Canvas-based custom chart

ML Concepts

Time-series forecasting

Rolling window supervised learning

Ensemble regression

Feature engineering

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ml_core import get_data, make_supervised, train_model, predict_next_price, make_signal

app = Flask(__name__)
CORS(app)

def build_prediction_for_symbol(symbol: str, lookback: int = 5):
    """Run the full pipeline for a single ticker and return a dict."""
    df = get_data(symbol)
    X, y = make_supervised(df, lookback=lookback)
    model = train_model(X, y)

    pred_next = predict_next_price(model, df, lookback=lookback)
    current_price = df["Close"].iloc[-1].item()
    signal, explanation = make_signal(current_price, pred_next)

    # recent history
    recent = df.tail(60)

    history = []
    for idx, row in recent.iterrows():
        history.append({
            "date": str(idx)[:10],             # '2025-11-12'
            "close": float(row["Close"]),      # scalar -> float
        })

    return {
        "symbol": symbol.upper(),
        "current_price": round(current_price, 2),
        "predicted_next_close": round(float(pred_next), 2),
        "signal": signal,
        "message": explanation,
        "history": history,
    }


# ---------------------------
# PAGE ROUTES
# ---------------------------

@app.route("/")
def index():
    # Home page -> templates/index.html
    return render_template("index.html")


@app.route("/search")
def search_page():
    # Search page -> templates/search.html
    return render_template("search.html")


@app.route("/how-it-works")
def how_page():
    # How it works page -> templates/how-it-works.html
    return render_template("how-it-works.html")


# ---------------------------
# API ROUTE
# ---------------------------

@app.route("/predict", methods=["GET"])
@app.route("/predict", methods=["GET"])
def predict():
    symbol = request.args.get("symbol", "AAPL").upper()
    lookback = 5
    period = request.args.get("period", "6mo")  # <--- NEW

    try:
        # 1. Get data for the selected period
        df = get_data(symbol, period=period)
        if df is None or df.empty:
            return jsonify({"error": "No data found for symbol"}), 400

        # 2. Build supervised dataset + train model
        X, y = make_supervised(df, lookback=lookback)
        if len(X) == 0:
            return jsonify({"error": "Not enough data to build training set"}), 400

        model = train_model(X, y)

        # 3. Predict next price
        pred_next = predict_next_price(model, df, lookback=lookback)

        last_close = df["Close"].iloc[-1]
        current_price = float(last_close)

        # 4. Signal
        signal, explanation = make_signal(current_price, pred_next)

        # 5. History for the chart (now whole period, not just 60 days)
        recent = df.copy().reset_index()
        date_col = "Date" if "Date" in recent.columns else recent.columns[0]

        history = []
        for _, row in recent.iterrows():
            date_val = row[date_col]
            if hasattr(date_val, "strftime"):
                date_str = date_val.strftime("%Y-%m-%d")
            else:
                date_str = str(date_val)[:10]

            history.append(
                {
                    "date": date_str,
                    "close": float(row["Close"]),
                }
            )

        return jsonify({
            "symbol": symbol,
            "current_price": round(float(current_price), 2),
            "predicted_next_close": round(float(pred_next), 2),
            "signal": signal,
            "message": explanation,
            "history": history,
            "period": period
        })

    except Exception as e:
        print("Error in /predict:", e)
        return jsonify({"error": "Server error while predicting"}), 500


@app.route("/predict_batch", methods=["GET"])
def predict_batch():
    """
    Example: /predict_batch?symbols=AAPL,TSLA,MSFT
    Returns: { "results": [ {..AAPL..}, {..TSLA..}, {..MSFT..} ] }
    """
    raw = request.args.get("symbols", "")
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]

    if not symbols:
        return jsonify({"error": "No symbols provided"}), 400

    results = []
    for sym in symbols:
        try:
            res = build_prediction_for_symbol(sym)
            results.append(res)
        except Exception as e:
            # if one fails, return an error record instead of crashing everything
            results.append({
                "symbol": sym,
                "error": str(e),
            })

    return jsonify({"results": results})
  
if __name__ == "__main__":
    app.run(debug=True)
    
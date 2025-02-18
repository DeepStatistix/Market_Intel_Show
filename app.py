from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import datetime
import logging
from werkzeug.exceptions import BadRequest
from tensorflow.keras.models import load_model

# Import from your forecast.py
from forecast import (
    generate_forecast,
    load_dataset,
    load_lstm_model,
    get_future_dates_for_trading_window,
    clip_trading_window_after_hist_last_date
)

app = Flask(__name__)

MARKET_MAPPING = {
    "Pulwama_Pachhar": "Pulwama_Pachhar",
    "Pulwama_Pricoo":  "Pulwama_Pricoo",
    "Shopian":         "Shopian"
}

VARIETIES = ["American", "Delicious", "Kullu Delicious", "Maharaji"]
GRADES = ["A", "B"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_inputs(market, variety, grade, days):
    logging.info(f"Validating inputs: market={market}, variety={variety}, grade={grade}, days={days}")
    if market not in MARKET_MAPPING:
        raise BadRequest(f"Unsupported market. Choose from: {', '.join(MARKET_MAPPING.keys())}")
    if variety not in VARIETIES:
        raise BadRequest(f"Unsupported variety. Choose from: {', '.join(VARIETIES)}")
    if grade not in GRADES:
        raise BadRequest(f"Unsupported grade. Choose from: {', '.join(GRADES)}")
    if not (1 <= days <= 365):
        raise BadRequest("Days must be between 1 and 365")
    return MARKET_MAPPING[market]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    """
    Only compute & return the future forecast for 'days' steps, ignoring any historical plot.
    We'll build official window => clip => slice => LSTM forecast
    """
    try:
        market = request.form.get('market')
        variety = request.form.get('variety')
        grade = request.form.get('grade')
        days_str = request.form.get('days')

        logging.info(f"Received form: market={market}, variety={variety}, grade={grade}, days={days_str}")
        try:
            days = int(days_str)
        except (TypeError, ValueError):
            raise BadRequest("Days must be an integer.")

        actual_market = validate_inputs(market, variety, grade, days)

        # Load LSTM => seq_length
        model, seq_length = load_lstm_model(actual_market, variety, grade)
        if model is None:
            raise RuntimeError("Model loading failed. No forecasting possible.")

        # For the official trading window, we'll assume the next "year" = current year + 1
        current_year = datetime.date.today().year
        next_year = current_year

        # Build entire official window
        full_window = get_future_dates_for_trading_window(actual_market, variety, grade, next_year)
        if not full_window:
            raise RuntimeError("No official trading window or it's empty.")

        # Suppose we want to ensure the forecast doesn't start before 'today' (or a specific date).
        # Let's clamp it so it starts after today's date.
        # We'll do: hist_last_date = pd.Timestamp(datetime.date.today())
        today_date = datetime.date.today()
        hist_last_date_ts = pd.Timestamp(today_date)  # a Timestamp version of today

        clipped_window = clip_trading_window_after_hist_last_date(
            hist_last_date_ts,  # pass a Timestamp
            full_window[0],
            full_window[-1]
        )
        if not clipped_window:
            raise RuntimeError("Clipped window is empty after ensuring it starts after 'today'.")

        # Slice to 'days'
        clipped_window = clipped_window[:days]
        future_days = len(clipped_window)

        # Now we just do the forecast. We load the entire dataset to feed the LSTM's final seq_length
        df = load_dataset(actual_market, variety, grade)
        if df is None:
            raise RuntimeError("Dataset not found or invalid.")
        # We'll forecast 'future_days' steps
        forecasted_prices = generate_forecast(actual_market, variety, grade, forecast_days=future_days, seq_length=seq_length)
        if forecasted_prices is None:
            raise RuntimeError("Forecast generation returned None.")

        forecast_list = forecasted_prices.tolist()
        # Convert our clipped_window dates to strings
        future_dates = [d.strftime("%Y-%m-%d") for d in clipped_window]

        logging.info("✅ Only future forecast returned, no historical data.")
        return jsonify({
            "status": "success",
            "market": actual_market,
            "variety": variety,
            "grade": grade,
            "forecast_days": future_days,
            # We do NOT return historical data at all
            "forecasted_prices": forecast_list,
            "future_dates": future_dates
        })

    except Exception as e:
        logging.error(f"Error in /forecast route: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400
# inside app.py
from qr_code import generate_qr_code_base64

@app.route("/qr_app")
def qr_app():
    # 1) Decide the link you want your QR code to open
    # If your app is on your local network, you might eventually replace 127.0.0.1 with your LAN IP 
    link_url = "https://market-intel-show.onrender.com/"  # or wherever your “future only” forecast is served

    # 2) Generate the QR code as base64
    qr_data = generate_qr_code_base64(link_url)

    # 3) Render a template that displays this base64 as an <img>
    return render_template("qr_app.html", qr_data=qr_data)

if __name__ == "__main__":
    app.run(debug=True)
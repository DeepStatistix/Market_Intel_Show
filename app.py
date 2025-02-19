# file: app.py
import os
import matplotlib
matplotlib.use('Agg')
import logging
import datetime
import json

import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Flask and local imports
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import BadRequest
from tensorflow.keras.models import load_model

import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

# Local modules
from data_collect import collect_all_data
from forecast import (
    generate_forecast,
    load_dataset,
    load_lstm_model,
    get_future_dates_for_trading_window,
    clip_trading_window_after_hist_last_date
)
from qr_code import generate_qr_code_base64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Mappings and constants
MARKET_MAPPING = {
    "Pulwama_Pachhar": "Pulwama_Pachhar",
    "Pulwama_Pricoo":  "Pulwama_Pricoo",
    "Shopian":         "Shopian"
}
VARIETIES = ["American", "Delicious", "Kullu Delicious", "Maharaji"]
GRADES = ["A", "B"]

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

# -------------------- ROUTES --------------------

@app.route('/')
def index():
    return render_template('index.html')

# Ensure 'static' directory exists for storing plots
if not os.path.exists("static"):
    os.makedirs("static")

# Load the dataset
DATA_FOLDER = "data"
df = collect_all_data(DATA_FOLDER)  # Already filtered where Mask == 1

# Validate dataset load
if df.empty:
    logging.error("No valid data found after filtering Mask == 1.")
else:
    logging.info(f"Loaded dataset with {len(df)} rows.")

# Dropdown options
MARKETS = df["Market"].unique().tolist()
VARIETIES = df["Variety"].unique().tolist()
GRADES = df["Grade"].unique().tolist()


@app.route('/plot', methods=['GET', 'POST'])
def plot():
    """
    Generates both individual price trends and comparative plots.
    """
    if request.method == 'GET':
        return render_template('plot.html', markets=MARKETS, varieties=VARIETIES, grades=GRADES)

    try:
        # Get user inputs
        market = request.form.get("market")
        variety = request.form.get("variety")
        grade = request.form.get("grade")

        logging.info(f"Generating plots for Market={market}, Variety={variety}, Grade={grade}")

        # ------------------ INDIVIDUAL PLOT (FILTERED BY USER SELECTION) ------------------
        filtered_df = df[
            (df["Market"] == market) &
            (df["Variety"] == variety) &
            (df["Grade"] == grade)
        ].copy()

        if filtered_df.empty:
            logging.warning(f"No data found for Market={market}, Variety={variety}, Grade={grade}")
            return render_template("plot.html", error="No data available.", markets=MARKETS, varieties=VARIETIES, grades=GRADES)

        # Convert 'Date' column to datetime format and sort
        filtered_df["Date"] = pd.to_datetime(filtered_df["Date"], errors="coerce")
        filtered_df.sort_values("Date", inplace=True)

        # **🔹 Fix for Missing Date Gaps: Plot Only Available Data**
        plt.figure(figsize=(15, 8))
        plt.title(f"Price Trends for {variety} ({grade}) in {market}", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price (₹/kg)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)

        # Only plot available data points (skipping missing dates)
        plt.plot(filtered_df["Date"], filtered_df["Min Price (per kg)"], label="Min Price", 
                 color="blue", linestyle="dotted", marker="o", markersize=4)
        plt.plot(filtered_df["Date"], filtered_df["Max Price (per kg)"], label="Max Price", 
                 color="red", linestyle="dotted", marker="s", markersize=4)
        plt.plot(filtered_df["Date"], filtered_df["Avg Price (per kg)"], label="Avg Price", 
                 color="green", linewidth=2, marker="D", markersize=4)

        plt.legend(loc="upper left", fontsize=10)
        plt.xticks(rotation=45)
        plt.ylim(20, 100)  # Ensures correct price scale
        plt.tight_layout()

        # Save the plot
        plot_path = "static/individual_plot.png"
        plt.savefig(plot_path)
        plt.close()

        logging.info(f"Individual plot saved at {plot_path}")

        # ------------------ COMPARATIVE PLOTS ------------------

        ## 📌 1️⃣ **Comparison Across Markets for the Selected Variety & Grade**
        market_comp_df = df[(df["Variety"] == variety) & (df["Grade"] == grade)][["Market", "Avg Price (per kg)"]]
        market_avg_prices = market_comp_df.groupby("Market").mean().reset_index()

        plt.figure(figsize=(12, 8))
        plt.bar(market_avg_prices["Market"], market_avg_prices["Avg Price (per kg)"], color="skyblue")
        plt.title(f"Avg Price Comparison Across Markets for {variety} ({grade})", fontsize=14)
        plt.xlabel("Market", fontsize=12)
        plt.ylabel("Avg Price (₹/kg)", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        market_plot_path = "static/market_comparison.png"
        plt.tight_layout()
        plt.savefig(market_plot_path)
        plt.close()

        ## 📌 2️⃣ **Comparison Across Varieties for the Selected Market & Grade**
        variety_comp_df = df[(df["Market"] == market) & (df["Grade"] == grade)][["Variety", "Avg Price (per kg)"]]
        variety_avg_prices = variety_comp_df.groupby("Variety").mean().reset_index()

        plt.figure(figsize=(12, 8))
        plt.bar(variety_avg_prices["Variety"], variety_avg_prices["Avg Price (per kg)"], color="lightcoral")
        plt.title(f"Avg Price Comparison Across Varieties in {market} ({grade})", fontsize=14)
        plt.xlabel("Variety", fontsize=12)
        plt.ylabel("Avg Price (₹/kg)", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        variety_plot_path = "static/variety_comparison.png"
        plt.tight_layout()
        plt.savefig(variety_plot_path)
        plt.close()

        logging.info("Comparative plots saved successfully.")

        return render_template("plot.html", 
                               individual_plot=plot_path, 
                               market_plot=market_plot_path, 
                               variety_plot=variety_plot_path, 
                               markets=MARKETS, varieties=VARIETIES, grades=GRADES)

    except Exception as e:
        logging.error(f"Error generating plots: {str(e)}", exc_info=True)
        return render_template("plot.html", error="Error generating plots.", markets=MARKETS, varieties=VARIETIES, grades=GRADES)

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

        # We assume the next "year" = current year
        current_year = datetime.date.today().year
        next_year = current_year

        # Build entire official window
        full_window = get_future_dates_for_trading_window(actual_market, variety, grade, next_year)
        if not full_window:
            raise RuntimeError("No official trading window or it's empty.")

        # We'll clamp the window so it starts after 'today'
        today_date = datetime.date.today()
        hist_last_date_ts = pd.Timestamp(today_date)

        clipped_window = clip_trading_window_after_hist_last_date(
            hist_last_date_ts,
            full_window[0],
            full_window[-1]
        )
        if not clipped_window:
            raise RuntimeError("Clipped window is empty after ensuring it starts after 'today'.")

        # Slice to 'days'
        clipped_window = clipped_window[:days]
        future_days = len(clipped_window)

        # Load dataset => forecast
        df = load_dataset(actual_market, variety, grade)
        if df is None:
            raise RuntimeError("Dataset not found or invalid.")
        
        forecasted_prices = generate_forecast(actual_market, variety, grade,
                                              forecast_days=future_days,
                                              seq_length=seq_length)
        if forecasted_prices is None:
            raise RuntimeError("Forecast generation returned None.")

        forecast_list = forecasted_prices.tolist()
        future_dates = [d.strftime("%Y-%m-%d") for d in clipped_window]

        logging.info("✅ Only future forecast returned, no historical data.")
        return jsonify({
            "status": "success",
            "market": actual_market,
            "variety": variety,
            "grade": grade,
            "forecast_days": future_days,
            "forecasted_prices": forecast_list,
            "future_dates": future_dates
        })

    except Exception as e:
        logging.error(f"Error in /forecast route: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/qr_app")
def qr_app():
    # Link to your deployed Render URL or whichever domain
    link_url = "https://market-intel-show.onrender.com/"
    qr_data = generate_qr_code_base64(link_url)
    return render_template("qr_app.html", qr_data=qr_data)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

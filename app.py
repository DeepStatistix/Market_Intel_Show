# file: app.py
import os
import logging
import datetime
import json

import pandas as pd
import numpy as np

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


@app.route("/trend", methods=["GET","POST"])
def trend():
    """
    A route that displays a form letting you pick Market/Variety/Grade
    (or All) then shows subplots with Min/Max/Avg lines for each group.
    """
    big_df = collect_all_data("data")
    if big_df.empty:
        return "No data found or Mask==1 rows are empty."

    # Gather unique sets for the form
    all_markets   = sorted(big_df["Market"].dropna().unique())
    all_varieties = sorted(big_df["Variety"].dropna().unique())
    all_grades    = sorted(big_df["Grade"].dropna().unique())

    selected_market  = None
    selected_variety = None
    selected_grade   = None
    plot_html        = None

    if request.method == "POST":
        selected_market  = request.form.get("market")
        selected_variety = request.form.get("variety")
        selected_grade   = request.form.get("grade")

        df_filtered = big_df.copy()
        if selected_market and selected_market != "All":
            df_filtered = df_filtered[df_filtered["Market"] == selected_market]
        if selected_variety and selected_variety != "All":
            df_filtered = df_filtered[df_filtered["Variety"] == selected_variety]
        if selected_grade and selected_grade != "All":
            df_filtered = df_filtered[df_filtered["Grade"] == selected_grade]

        print("\n[DEBUG] After user filter. shape=", df_filtered.shape)
        # 1) Print largest 5 'Max Price (per kg)' to see any outlier
        if not df_filtered.empty:
            largest_5 = df_filtered.nlargest(5, "Max Price (per kg)")
            print("[DEBUG] largest 5 Max Price:\n", largest_5.to_string(index=False))

        if df_filtered.empty:
            plot_html = "<h3>No data for that combination. Try different inputs.</h3>"
        else:
            # Group -> subplots
            combos = df_filtered.groupby(["Market","Variety","Grade"])
            n_plots= len(combos)
            fig = make_subplots(
                rows=n_plots, cols=1,
                shared_xaxes=True,
                shared_yaxes=False,
                subplot_titles=[f"{m}-{v}-{g}" for (m,v,g) in combos.groups.keys()]
            )

            row_i = 1
            for (mkt, var, grd), group in combos:
                group = group.sort_values("Date")

                # Add 3 lines
                fig.add_trace(go.Scatter(
                    x=group["Date"],
                    y=group["Min Price (per kg)"],
                    mode='lines',
                    name=f'Min ({grd})',
                    line=dict(color='blue', width=2),
                    showlegend=True),
                    row=row_i, col=1)


                fig.add_trace(go.Scatter(
                    x=group["Date"],
                    y=group["Max Price (per kg)"],
                    mode='lines',
                    name=f'Max ({grd})',
                    line=dict(color='red', width=2),
                    showlegend=True),
                    row=row_i, col=1)


                fig.add_trace(go.Scatter(
                    x=group["Date"],
                    y=group["Avg Price (per kg)"],
                    mode='lines',
                    name=f'Avg ({grd})',
                    line=dict(color='green', width=2),
                    showlegend=True),
                    row=row_i, col=1)



                row_i += 1

            fig.update_layout(
                title="Min/Max/Avg Price Trends",
                hovermode="x unified",
                height=400*n_plots,
                xaxis=dict(
                    tickformat='%Y-%m-%d',
                    tickangle=45
                ),
                yaxis=dict(
                    tickprefix='₹',
                    tickformat=',.0f'
                )
            )
            fig.update_xaxes(title="Date", type='date')
            fig.update_yaxes(title="Price (₹/kg)", separatethousands=True)


            plot_html = pyo.plot(fig, include_plotlyjs=False, output_type='div')

    return render_template(
        "trend.html",
        all_markets=all_markets,
        all_varieties=all_varieties,
        all_grades=all_grades,
        selected_market=selected_market,
        selected_variety=selected_variety,
        selected_grade=selected_grade,
        plot_html=plot_html
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

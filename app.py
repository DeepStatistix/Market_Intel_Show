# file: app.py
import os
import matplotlib
matplotlib.use('Agg')
import logging
import datetime
import json
import seaborn as sns
from io import BytesIO
import base64
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
    "Shopian":         "Shopian",
    "Narwal":          "Narwal"
}
# Allowed varieties for other markets vs. Narwal can differ. For simplicity, we let the collected dataset decide.
VARIETIES = [["American", "Delicious", "Kullu Delicious", "Maharaji","Hazratbali","Condition","Razakwadi"]]  # Will be set based on collected data.
GRADES = ["A","B"]     # Will be set based on collected data.

def validate_inputs(market, variety, grade, days):
    logging.info(f"Validating inputs: market={market}, variety={variety}, grade={grade}, days={days}")
    if market not in MARKET_MAPPING:
        raise BadRequest(f"Unsupported market. Choose from: {', '.join(MARKET_MAPPING.keys())}")
    
    # For Narwal market, ignore the grade.
    if market == "Narwal":
        grade = None
    else:
        if grade not in GRADES:
            raise BadRequest(f"Unsupported grade. Choose from: {', '.join(GRADES)}")
    
    if not (1 <= days <= 365):
        raise BadRequest("Days must be between 1 and 365")
    
    return MARKET_MAPPING[market], grade

# -------------------- ROUTES --------------------

@app.route('/')
def index():
    return render_template('index.html')

# Ensure 'static' directory exists for storing plots
if not os.path.exists("static"):
    os.makedirs("static")

# Load the dataset (already filtered where Mask == 1)
DATA_FOLDER = "data"
df = collect_all_data(DATA_FOLDER)
if df.empty:
    logging.error("No valid data found after filtering Mask == 1.")
else:
    logging.info(f"Loaded dataset with {len(df)} rows.")

# Set dropdown options based on collected data
MARKETS = df["Market"].unique().tolist()
VARIETIES = df["Variety"].unique().tolist()
# For markets with grades, use unique grades; for Narwal, it might be "Unknown" or missing.
GRADES = df["Grade"].dropna().unique().tolist()

@app.route('/plot', methods=['GET', 'POST'])
def plot():
    """
    Generates both individual price trends and comparative plots.
    """
    if request.method == 'GET':
        return render_template('plot.html', markets=MARKETS, varieties=VARIETIES, grades=GRADES)

    try:
        market = request.form.get("market")
        variety = request.form.get("variety")
        grade = request.form.get("grade")

        # For Narwal, ignore the grade
        if market == "Narwal":
            grade = None

        logging.info(f"Generating plots for Market={market}, Variety={variety}, Grade={grade}")

        # Filter dataset: if Narwal, ignore grade filter
        if market == "Narwal":
            filtered_df = df[(df["Market"] == market) & (df["Variety"] == variety)].copy()
        else:
            filtered_df = df[(df["Market"] == market) & (df["Variety"] == variety) & (df["Grade"] == grade)].copy()

        if filtered_df.empty:
            logging.warning(f"No data found for Market={market}, Variety={variety}, Grade={grade}")
            return render_template("plot.html", error="No data available.", markets=MARKETS, varieties=VARIETIES, grades=GRADES)

        filtered_df["Date"] = pd.to_datetime(filtered_df["Date"], errors="coerce")
        filtered_df.sort_values("Date", inplace=True)

        plt.figure(figsize=(15, 8))
        plt.title(f"Price Trends for {variety}" + (f" ({grade})" if grade else ""), fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price (₹/kg)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.plot(filtered_df["Date"], filtered_df["Min Price (per kg)"], label="Min Price", 
                 color="blue", linestyle="dotted", marker="o", markersize=4)
        plt.plot(filtered_df["Date"], filtered_df["Max Price (per kg)"], label="Max Price", 
                 color="red", linestyle="dotted", marker="s", markersize=4)
        plt.plot(filtered_df["Date"], filtered_df["Avg Price (per kg)"], label="Avg Price", 
                 color="green", linewidth=2, marker="D", markersize=4)

        plt.legend(loc="upper left", fontsize=10)
        plt.xticks(rotation=45)
        plt.ylim(5, 100)
        plt.tight_layout()

        plot_path = "static/individual_plot.png"
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Individual plot saved at {plot_path}")

        ## Comparative Plots remain unchanged (they filter by grade for non-Narwal markets)
        if market != "Narwal":
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
        else:
            market_plot_path = None

        if market != "Narwal":
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
        else:
            variety_plot_path = None

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
    Only compute & return the future forecast for 'days' steps.
    For Narwal market, since there is no grade, grade is ignored.
    For other markets, we build the official trading window.
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

        # Validate inputs; for Narwal, grade will be set to None.
        actual_market, grade = validate_inputs(market, variety, grade, days)

        # Load LSTM model => seq_length
        model, seq_length = load_lstm_model(actual_market, variety, grade)
        if model is None:
            raise RuntimeError("Model loading failed. No forecasting possible.")

        current_year = datetime.date.today().year
        next_year = current_year

        # For Narwal, we skip using a trading window and simply forecast days ahead from last historical date.
        if actual_market == "Narwal":
            df = load_dataset(actual_market, variety, grade)
            if df is None:
                raise RuntimeError("Dataset not found or invalid.")
            df.sort_values(by="Date", inplace=True)
            last_date = df["Date"].max()
            clipped_window = [last_date + datetime.timedelta(days=i) for i in range(1, days+1)]
            future_days = days
        else:
            full_window = get_future_dates_for_trading_window(actual_market, variety, grade, next_year)
            if not full_window:
                raise RuntimeError("No official trading window or it's empty.")
            today_date = datetime.date.today()
            hist_last_date_ts = pd.Timestamp(today_date)
            clipped_window = clip_trading_window_after_hist_last_date(
                hist_last_date_ts,
                full_window[0],
                full_window[-1]
            )
            if not clipped_window:
                raise RuntimeError("Clipped window is empty after ensuring it starts after 'today'.")
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
            "grade": grade if grade else "",
            "forecast_days": future_days,
            "forecasted_prices": forecast_list,
            "future_dates": future_dates
        })

    except Exception as e:
        logging.error(f"Error in /forecast route: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400
# Register the blueprint

CSV_PATH = "data/scraped/apple_data.csv"  # Path to your CSV

@app.route('/')
def home():
    """
    Renders a simple home page or directly the form. 
    We'll redirect to /analysis_form for the actual user input.
    """
    return '<h2>Welcome! Go to <a href="/analysis_form">Analysis Form</a>.</h2>'

@app.route('/analysis_form', methods=['GET'])
def analysis_form():
    """
    Renders a form where the user can choose District, Market, Variety, and Grade.
    On submission, it sends a GET request to /analysis with query parameters.
    """
    # We can load some unique values from the CSV to populate dropdowns:
    if not os.path.exists(CSV_PATH):
        return f"CSV file not found at {CSV_PATH}", 500
    
    df = pd.read_csv(CSV_PATH)

    # Convert date columns, numeric columns if needed
    # (Only necessary if you want to handle them here.)
    # For the sake of generating dropdown options, we just need the string columns.

    districts = sorted(df["District"].dropna().unique().tolist())
    markets   = sorted(df["Market"].dropna().unique().tolist())
    varieties = sorted(df["Variety"].dropna().unique().tolist())
    grades    = sorted(df["Grade"].dropna().unique().tolist())

    return render_template("analysis_form.html",
                           districts=districts,
                           markets=markets,
                           varieties=varieties,
                           grades=grades)

@app.route('/analysis', methods=['GET'])
def analysis():
    """
    Loads the scraped apple data from a CSV file, filters for specific state-district-market combos,
    creates a table and a separate Matplotlib chart for each (with hue for Variety and Grade),
    and displays them on one page.
    """
    csv_path = "data/scraped/apple_data.csv"
    if not os.path.exists(csv_path):
        return f"CSV file not found at {csv_path}", 500

    # 1. Load the CSV
    df = pd.read_csv(csv_path)
    
    # 2. Ensure price columns are numeric (remove commas then convert)
    for col in ['Min Price', 'Max Price', 'Model Price']:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d %b %Y', errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.sort_values(by="Date", inplace=True)
    
    # Debug: Print dtypes and sample values
    print("Dtypes after conversion:\n", df.dtypes)
    print("Sample Model Price values:\n", df['Model Price'].head())

    # 3. Define default combos as (State, District, Market)
    combos = [
        ("Jammu and Kashmir", "Jammu", "Narwal Jammu (F&V)"),
        ("NCT of Delhi", "Delhi", "Azadpur"),
        ("Karnataka", "Bangalore", "Binny Mill (F&V), Bangalore"),
        ("West Bengal", "Kolkata", "Mechua"),
        ("Uttar Pradesh", "Amethi", "Sultanpur"),
        ("Srinagar", "Srinagar", "Parimpore")
    ]

    analysis_items = []

    for (state_val, district_val, market_val) in combos:
        # 4. Filter the DataFrame for each state-district-market combo
        subdf = df[(df['State'] == state_val) & (df['District'] == district_val) & (df['Market'] == market_val)].copy()
        if subdf.empty:
            logging.info(f"No data found for State={state_val}, District={district_val}, Market={market_val}")
            continue

        # 5. Build a table of the latest data (last 10 rows sorted by Date descending)
        subdf_latest = subdf.sort_values(by="Date", ascending=False).head(10)
        table_html = subdf_latest.to_html(
            classes="table table-striped",
            index=False,
            border=0
        )

        # 6. Create a Matplotlib figure for this combo
        # Create a new column combining Variety and Grade for use as hue
        subdf["Variety_Grade"] = subdf["Variety"] + " (" + subdf["Grade"] + ")"

        plt.figure(figsize=(8, 5))
        sns.lineplot(data=subdf, x="Date", y="Model Price", hue="Variety_Grade", marker="o")
        plt.title(f"{state_val} - {district_val} - {market_val}")
        plt.xlabel("Date")
        plt.ylabel("Price (₹/quintal)")  # Adjust if your data is in ₹/kg; you might multiply by 100
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure to a BytesIO buffer
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf8")
        plt.close()  # Close the figure to free memory
        plot_img = f"data:image/png;base64,{image_base64}"
        
        # 7. Append the combo's results
        analysis_items.append({
            "combo_title": f"{state_val} - {district_val} - {market_val}",
            "table_html": table_html,
            "plot_img": plot_img
        })

    return render_template("analysis.html", analysis_items=analysis_items)

@app.route("/qr_app")
def qr_app():
    # Link to your deployed Render URL or whichever domain
    link_url = "https://market-intel-show.onrender.com/"
    qr_data = generate_qr_code_base64(link_url)
    return render_template("qr_app.html", qr_data=qr_data)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

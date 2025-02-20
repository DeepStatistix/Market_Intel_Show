import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.metrics import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import logging
import matplotlib.pyplot as plt
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mapping for markets
MARKET_DATA_PATHS = {
    "Pulwama_Pachhar": "data/Pulwama/Pachhar/",
    "Pulwama_Pricoo":  "data/Pulwama/Prichoo/",
    "Shopian":         "data/Shopian/",
    "Narwal":          "data/Narwal/"
}

MODEL_PATHS = {
    "Pulwama_Pachhar": "models/Pulwama/Pachhar/{variety}/{grade}/",
    "Pulwama_Pricoo":  "models/Pulwama/Prichoo/{variety}/{grade}/",
    "Shopian":         "models/Shopian/",
    "Narwal":          "models/Narwal/"
}

def load_dataset(market, variety, grade):
    """Loads CSV, keeps only Date + Avg Price for Mask == 1."""
    base_dir = MARKET_DATA_PATHS.get(market)
    if base_dir is None:
        logging.error(f"❌ Market '{market}' not found in MARKET_DATA_PATHS.")
        return None

    # For Narwal market, datasets do not have a grade column.
    if market == "Narwal":
        file_path = os.path.join(base_dir, f"{variety}_dataset.csv")
    else:
        file_path = os.path.join(base_dir, f"{variety}_{grade}_dataset.csv")
    logging.info(f"✅ Checking dataset path: {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"❌ Dataset file not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    logging.info(f"Dataset Columns: {df.columns.tolist()}")
    required_cols = {"Date", "Mask", "Avg Price (per kg)"}
    if not required_cols.issubset(df.columns):
        logging.error("❌ Required columns missing in dataset.")
        return None

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)
    df.sort_values(by="Date", inplace=True)
    df = df[df["Mask"] == 1]  # only keep masked rows
    if df.empty:
        logging.error("❌ No valid data after Mask==1.")
        return None
    if df["Avg Price (per kg)"].isnull().any():
        logging.error("❌ 'Avg Price (per kg)' has NaNs.")
        return None

    logging.info("✅ Dataset loaded successfully")
    return df[["Date", "Avg Price (per kg)"]]

def load_lstm_model(market, variety, grade):
    """Load LSTM model, return model + seq_length."""
    if market == "Narwal":
        # For Narwal, models are stored without a grade folder/filename.
        base_path = MODEL_PATHS.get(market, "")
        model_path = os.path.join(base_path, f"lstm_{variety}.h5")
    else:
        base_path = MODEL_PATHS.get(market, "")
        base_path = base_path.format(variety=variety, grade=grade)
        model_path = f"{base_path}lstm_{variety}_grade_{grade}.h5"
    logging.info(f"Checking model path: {model_path}")

    if not os.path.exists(model_path):
        logging.error(f"❌ Model file not found: {model_path}")
        return None, None

    custom_objects = {"mse": MeanSquaredError()}
    model = load_model(model_path, custom_objects=custom_objects)
    seq_length = model.input_shape[1]
    logging.info(f"✅ Model loaded. seq_length={seq_length}")
    return model, seq_length

def prepare_data(df, seq_length):
    """Scale 'Avg Price' & build sequences."""
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[["Avg Price (per kg)"]].values)

    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i : i + seq_length])
        y.append(data_scaled[i + seq_length])

    return np.array(X), np.array(y), scaler

def forecast_future(model, scaler, df, seq_length, forecast_days):
    """
    Step-by-step LSTM forecast from last seq_length points in df.
    Returns a NumPy array of length forecast_days.
    """
    data_scaled = scaler.transform(df[["Avg Price (per kg)"]].values)
    input_seq = data_scaled[-seq_length:].reshape(1, seq_length, 1)

    preds = []
    for _ in range(forecast_days):
        next_price = model.predict(input_seq, verbose=0)[0, 0]
        preds.append(next_price)
        input_seq = np.append(input_seq[:, 1:, :], [[[next_price]]], axis=1)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds

def generate_forecast(market, variety, grade, forecast_days=10, seq_length=None):
    """Generate future forecast for 'forecast_days' steps."""
    if seq_length is None:
        logging.error("❌ seq_length must be provided.")
        raise ValueError("seq_length must be provided from the model input_shape.")

    # For Narwal, grade is not used; so we pass an empty string or None.
    grade_display = grade if market != "Narwal" else ""
    logging.info(f"✅ Generating forecast for {market}, {variety} {grade_display}, days={forecast_days}")
    
    df = load_dataset(market, variety, grade)
    if df is None:
        return None

    model, seq_length = load_lstm_model(market, variety, grade)
    if model is None:
        return None

    logging.info("✅ Model & dataset loaded. Preparing data.")
    X, y, scaler = prepare_data(df, seq_length)
    if X.shape[0] == 0:
        logging.error("❌ Not enough data for the given seq_length.")
        return None

    logging.info(f"✅ Forecasting {forecast_days} days ahead...")
    preds = forecast_future(model, scaler, df, seq_length, forecast_days)
    if preds is None or len(preds) == 0:
        logging.error("❌ Forecasting returned no predictions.")
        return None

    logging.info(f"✅ Forecast done. Predictions: {preds}")
    return preds

# Trading window functions remain unchanged, assuming they apply only to markets that use grade.
def clip_trading_window_after_hist_last_date(hist_last_date, start_date, end_date):
    """
    hist_last_date: pd.Timestamp or datetime.date
    start_date, end_date: datetime.date
    Shifts start_date if it falls before hist_last_date + 1 day.
    """
    if isinstance(hist_last_date, pd.Timestamp):
        hist_last_date = hist_last_date.date()

    if start_date <= hist_last_date:
        start_date = hist_last_date + datetime.timedelta(days=1)
    
    if start_date > end_date:
        return []

    future_dates = []
    current = start_date
    while current <= end_date:
        future_dates.append(current)
        current += datetime.timedelta(days=1)
    return future_dates

MARKET_TRADE_WINDOWS = {
    "Pulwama_Pachhar": {
        "American": {
            "A":  ((9, 15), (11, 15)),  
            "B":  ((9, 15), (11, 15))
        },
        "Delicious": {
            "A":  ((9, 15), (12, 15)),  
            "B":  ((9, 15), (12, 15))
        },
        "Kullu Delicious": {
            "A":  ((9, 1),  (11, 15)),
            "B":  ((9, 1),  (11, 15))
        }
    },
    "Pulwama_Pricoo": {
        "American": {
            "A":  ((9, 15), (11, 15)),
            "B":  ((9, 15), (11, 15))
        },
        "Delicious": {
            "A":  ((9, 15), (12, 15)),
            "B":  ((9, 15), (12, 15))
        },
        "Kullu Delicious": {
            "A":  ((9, 1),  (11, 15)),
            "B":  ((9, 1),  (11, 15))
        }
    },
    "Shopian": {
        "American": {
            "A":  ((10, 1), (11, 30)),
            "B":  ((10, 1), (11, 30))
        },
        "Delicious": {
            "A":  ((9, 15), (12, 31)),
            "B":  ((9, 15), (12, 31))
        },
        "Kullu Delicious": {
            "A":  ((9, 1), (12, 15)),
            "B":  ((9, 1), (12, 15))
        },
        "Maharaji": {
            "A":  ((10, 1), (11, 30)),
            "B":  ((10, 1), (11, 30))
        }
    }
}

def get_future_dates_for_trading_window(market, variety, grade, year):
    """Return list of daily dates for the specified year within the known trading window."""
    market_entry = MARKET_TRADE_WINDOWS.get(market)
    if not market_entry:
        raise ValueError(f"No trading window for market={market}")
    
    variety_entry = market_entry.get(variety)
    if not variety_entry:
        raise ValueError(f"No trading window for variety={variety} in {market}")
    
    grade_entry = variety_entry.get(grade)
    if not grade_entry:
        raise ValueError(f"No trading window for grade={grade} of {variety} in {market}")
    
    (start_month_day, end_month_day) = grade_entry
    start_month, start_day = start_month_day
    end_month, end_day     = end_month_day
    
    start_date = datetime.date(year, start_month, start_day)
    end_date   = datetime.date(year, end_month, end_day)
    if end_date < start_date:
        raise ValueError("End date is before start date—check your config.")
    
    future_dates = []
    current = start_date
    while current <= end_date:
        future_dates.append(current)
        current += datetime.timedelta(days=1)
    
    return future_dates

# Standalone debug
if __name__ == "__main__":
    # Example usage for a non-Narwal market:
    market = "Pulwama_Pachhar"
    variety = "American"
    grade = "A"
    days = 10

    logging.info(f"Debug run. Market={market}, variety={variety}, grade={grade}, days={days}")
    model, seq_len = load_lstm_model(market, variety, grade)
    if model is None:
        logging.error("No model found. Exiting.")
    else:
        df = load_dataset(market, variety, grade)
        if df is None:
            logging.error("No dataset found or invalid. Exiting.")
        else:
            df.sort_values(by="Date", inplace=True)
            hist_df = df.tail(30)
            hist_prices = hist_df["Avg Price (per kg)"].tolist()
            hist_dates = hist_df["Date"].dt.strftime("%Y-%m-%d").tolist()

            preds = generate_forecast(market, variety, grade, forecast_days=days, seq_length=seq_len)
            if preds is not None:
                last_date = df["Date"].max()
                future_dates = [
                    (last_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
                    for i in range(1, days+1)
                ]
                logging.info(f"Forecast: {preds}")

                plt.figure(figsize=(10,5))
                plt.plot(future_dates, preds, marker='o', color='orange', label="Forecast")
                plt.title(f"{variety} Grade {grade} in {market}")
                plt.xlabel("Date")
                plt.ylabel("Price (₹/kg)")
                plt.xticks(rotation=45)
                plt.legend()
                plt.grid()
                plt.show()

    # Example usage for Narwal market (which has no grade):
    market = "Narwal"
    variety = "American"
    grade = None  # Grade is not applicable
    days = 10

    logging.info(f"Debug run. Market={market}, variety={variety}, no grade, days={days}")
    model, seq_len = load_lstm_model(market, variety, grade)
    if model is None:
        logging.error("No model found for Narwal. Exiting.")
    else:
        df = load_dataset(market, variety, grade)
        if df is None:
            logging.error("No dataset found or invalid for Narwal. Exiting.")
        else:
            df.sort_values(by="Date", inplace=True)
            preds = generate_forecast(market, variety, grade, forecast_days=days, seq_length=seq_len)
            if preds is not None:
                last_date = df["Date"].max()
                future_dates = [
                    (last_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
                    for i in range(1, days+1)
                ]
                logging.info(f"Forecast for Narwal: {preds}")

                plt.figure(figsize=(10,5))
                plt.plot(future_dates, preds, marker='o', color='orange', label="Forecast")
                plt.title(f"{variety} in {market}")
                plt.xlabel("Date")
                plt.ylabel("Price (₹/kg)")
                plt.xticks(rotation=45)
                plt.legend()
                plt.grid()
                plt.show()

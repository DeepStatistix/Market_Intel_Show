# file: collect_all_data.py
import os
import pandas as pd

def collect_all_data(data_root="data"):
    """
    Recursively walks data_root subfolders (Pulwama/Pachhar, Shopian, etc.)
    Loads every CSV that has 'Date','Mask','Min Price (per kg)','Max Price (per kg)','Avg Price (per kg)'.
    Merges into one DataFrame, returns it sorted by Date.
    """
    all_rows = []

    for root, dirs, files in os.walk(data_root):
        for filename in files:
            if filename.endswith(".csv"):
                filepath = os.path.join(root, filename)
                df = pd.read_csv(filepath)

                # Required columns
                req_cols = {"Date", "Mask", "Min Price (per kg)",
                            "Max Price (per kg)", "Avg Price (per kg)"}
                if not req_cols.issubset(df.columns):
                    continue  # skip if missing required

                # Filter Mask==1
                df = df[df["Mask"] == 1].copy()
                if df.empty:
                    continue

                # Attempt to parse Market, Variety, Grade from either the CSV
                # or from the filename. For simplicity, let's just do:
                market = df["Market"].iloc[0] if "Market" in df.columns else "Unknown"
                variety = df["Variety"].iloc[0] if "Variety" in df.columns else "Unknown"
                grade = df["Grade"].iloc[0] if "Grade" in df.columns else "Unknown"

                # Keep relevant columns
                tmp = df[["Date","Min Price (per kg)","Max Price (per kg)","Avg Price (per kg)"]].copy()
                tmp["Market"] = market
                tmp["Variety"] = variety
                tmp["Grade"] = grade
                all_rows.append(tmp)

    if not all_rows:
        return pd.DataFrame()  # empty if none

    big_df = pd.concat(all_rows, ignore_index=True)
    # Convert 'Date' to datetime
    big_df["Date"] = pd.to_datetime(big_df["Date"], errors="coerce")
    big_df.dropna(subset=["Date"], inplace=True)
    big_df.sort_values(by="Date", inplace=True)
    return big_df
# ------------------- MAIN BLOCK FOR TESTING -------------------

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt  # for quick local plotting

    data_folder = sys.argv[1] if len(sys.argv) > 1 else "data"
    print(f"Collecting CSV data from: {data_folder}")
    big_df = collect_all_data(data_folder)

    if big_df.empty:
        print("No valid data found or all are empty after Mask==1.")
    else:
        print("Number of rows:", len(big_df))
        print("Columns:", list(big_df.columns))

        print("\nDataFrame info:")
        big_df.info()

        # Print unique Varieties and Grades across the entire dataset
        all_varieties = big_df["Variety"].dropna().unique()
        all_grades = big_df["Grade"].dropna().unique()
        print("\nAll unique Varieties in dataset:\n", list(all_varieties))
        print("\nAll unique Grades in dataset:\n", list(all_grades))

        # Show unique Varieties and Grades per Market
        print("\nUnique Varieties per Market:")
        for market_name, subset in big_df.groupby("Market"):
            unique_vars = subset["Variety"].dropna().unique()
            print(f"  Market={market_name}: {list(unique_vars)}")

        print("\nUnique Grades per Market:")
        for market_name, subset in big_df.groupby("Market"):
            unique_grades = subset["Grade"].dropna().unique()
            print(f"  Market={market_name}: {list(unique_grades)}")

        print("\nSample rows (head 10):")
        print(big_df.head(10).to_string(index=False))

        # Print a statistical summary per (Market,Variety)
        grouped_mv = big_df.groupby(["Market", "Variety"])
        for (mkt, var), group in grouped_mv:
            print(f"\n--- Statistical Summary for Market={mkt}, Variety={var} ---")
            print(group.describe())

        # ---------- QUICK LINE PLOTS FOR EACH (Market,Variety) ----------
        # This can produce many windows if you have a large dataset or many combos.
        for (mkt, var), group in grouped_mv:
            if group.empty:
                continue
            group = group.sort_values("Date")

            # We'll do a small figure
            plt.figure(figsize=(8, 4))
            plt.title(f"{mkt} - {var}")
            plt.plot(group["Date"], group["Min Price (per kg)"], label="Min", color="blue")
            plt.plot(group["Date"], group["Max Price (per kg)"], label="Max", color="red")
            plt.plot(group["Date"], group["Avg Price (per kg)"], label="Avg", color="green")

            plt.xlabel("Date")
            plt.ylabel("Price (₹/kg)")
            plt.legend()
            plt.tight_layout()
            plt.show()

    
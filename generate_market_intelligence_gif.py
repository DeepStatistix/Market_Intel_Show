import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# Ensure output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Generate 100 days of sample dates (10 seconds at 10 fps = 100 frames)
dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

# Simulate apple prices with seasonal fluctuations (centered around ₹55)
price_delicious = 55 + 3 * np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(0, 1, size=100)
price_american = 55 + 2 * np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(0, 1, size=100)
price_maharaji = 55 + 1.5 * np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(0, 1, size=100)

# Simulated demand predictions (inversely related to seasonal trend)
demand_prediction = 70 - 2 * np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(0, 2, size=100)

# Create DataFrame
df = pd.DataFrame({
    "Date": dates,
    "Delicious": price_delicious,
    "American": price_american,
    "Maharaji": price_maharaji,
    "Demand": demand_prediction
})

# Set up figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(df["Date"].min(), df["Date"].max())
ax.set_ylim(40, 80)  # Cover price & demand ranges
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Price (₹/kg) & Demand Index", fontsize=12)
ax.set_title("Market Intelligence for Apple Forecasting in J&K", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.5)

# Initialize line objects for apple prices and demand
line_del, = ax.plot([], [], label="Delicious", color="red", linewidth=2)
line_ame, = ax.plot([], [], label="American", color="blue", linewidth=2)
line_mah, = ax.plot([], [], label="Maharaji", color="green", linewidth=2)
line_demand, = ax.plot([], [], label="Demand", color="purple", linestyle="--", linewidth=2)

# Create a text object for overlays; we'll update its content in the animation.
info_text = ax.text(0.5, 0.9, "", transform=ax.transAxes, fontsize=12, ha="center", bbox=dict(facecolor='white', alpha=0.7))

# Add legend outside the plot area
ax.legend(loc="upper left", bbox_to_anchor=(0, 1.05), ncol=2, fontsize=10)

# Define update function for the animation
def update(frame):
    # Update the data for each line based on frame number
    line_del.set_data(df["Date"][:frame], df["Delicious"][:frame])
    line_ame.set_data(df["Date"][:frame], df["American"][:frame])
    line_mah.set_data(df["Date"][:frame], df["Maharaji"][:frame])
    line_demand.set_data(df["Date"][:frame], df["Demand"][:frame])
    
    # Determine which segment of the story we are in:
    # 0-24 frames: Price Trends
    # 25-49 frames: Seasonal Forecasts
    # 50-74 frames: Demand Predictions
    # 75-100 frames: Forecasting for Farmers (Storytelling)
    if frame < 25:
        info_text.set_text("Price Trends for Key Apple Varieties\n(₹50–₹60/kg)")
    elif frame < 50:
        info_text.set_text("Seasonal Fluctuations & Forecasts\nAutumn Dip, Winter Recovery, Summer Peak")
    elif frame < 75:
        info_text.set_text("Demand Predictions\nLower Prices → Higher Demand\nGrowing Yearly")
    else:
        info_text.set_text("Forecasting Empowers Farmers\nSell at the Right Time & Plan Supply")
    
    return line_del, line_ame, line_mah, line_demand, info_text

# Create animation: 100 frames, 10 fps means a 10-second GIF.
ani = animation.FuncAnimation(fig, update, frames=len(df), interval=100, blit=True, repeat=True)

# Save the animation as a GIF
gif_path = os.path.join(output_dir, "apple_market_intelligence.gif")
ani.save(gif_path, writer="pillow", fps=10)

plt.close(fig)
print(f"GIF saved at: {gif_path}")

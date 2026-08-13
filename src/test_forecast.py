import pandas as pd

from forecasting.predictor import forecast_next_hours


# Load historical data
df = pd.read_csv(
    "data/raw/ed_arrivals.csv"
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)


# Use the latest historical data
history = df.tail(100).copy()


# Forecast next 6 hours
forecast = forecast_next_hours(
    history=history,
    hours=12,
)

print("\n12-Hour ED Forecast")
print("==================")
print(history["timestamp"].iloc[-1])
print("Number of predictions:", len(forecast))

print(forecast)
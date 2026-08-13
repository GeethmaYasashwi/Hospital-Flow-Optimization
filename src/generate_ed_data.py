import numpy as np
import pandas as pd


# Make results reproducible
np.random.seed(42)

#This is synthetic data generation for a hospital emergency department (ED) arrivals dataset. 
#The goal is to create a realistic dataset that can be used for analysis, modeling, and forecasting of ED arrivals.
# This is solely for educational purposes and should not be used for any real-world medical or operational decisions.

# 1. Create one year of hourly timestamps

timestamps = pd.date_range(
    start="2026-01-01 00:00:00",
    end="2026-12-31 23:00:00",
    freq="h",
)

df = pd.DataFrame({
    "timestamp": timestamps
})


# 2. Create time-based features

df["hour"] = df["timestamp"].dt.hour

df["day_of_week"] = df["timestamp"].dt.day_name()

df["is_weekend"] = (
    df["timestamp"].dt.dayofweek >= 5
).astype(int)


# 3. Simulate holidays

# These are fictional hospital holidays for our simulation.
holiday_dates = [
    "2026-01-01",
    "2026-04-14",
    "2026-05-01",
    "2026-12-25",
]

df["is_holiday"] = (
    df["timestamp"].dt.strftime("%Y-%m-%d").isin(holiday_dates)
).astype(int)

# 4. Create realistic arrival patterns

# Basic number of arrivals
base = 3


# More arrivals during daytime
day_effect = np.where(
    (df["hour"] >= 7) & (df["hour"] <= 21),
    4,
    0,
)


# Evening peak
evening_peak = np.where(
    (df["hour"] >= 16) & (df["hour"] <= 19),
    3,
    0,
)


# Slightly higher weekend demand
weekend_effect = np.where(
    df["is_weekend"] == 1,
    2,
    0,
)

# Slightly lower demand on holidays
holiday_effect = np.where(
    df["is_holiday"] == 1,
    -1,
    0,
)


# Random variation
random_effect = np.random.normal(
    loc=0,
    scale=1.5,
    size=len(df),
)


# 5. Calculate simulated arrivals

arrivals = (
    base
    + day_effect
    + evening_peak
    + weekend_effect
    + holiday_effect
    + random_effect
)


# Admissions cannot be negative
df["arrivals"] = np.maximum(
    np.round(arrivals),
    0,
).astype(int)

# 6. Select final columns

df = df[
    [
        "timestamp",
        "arrivals",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "hour",
    ]
]

# 7. Save raw data

output_path = "data/raw/ed_arrivals.csv"

df.to_csv(
    output_path,
    index=False,
)

print("ED dataset created successfully!")
print(f"Rows: {len(df):,}")
print(f"Saved to: {output_path}")
print("\nFirst 10 rows:")
print(df.head(10))
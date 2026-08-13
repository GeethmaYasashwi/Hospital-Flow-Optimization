from pathlib import Path

import joblib
import pandas as pd


# --------------------------------------------------
# Load the trained model
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "ed_forecasting_random_forest_model.joblib"


def load_model():
    """
    Load the saved ED forecasting model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    # If we saved a dictionary containing the model
    if isinstance(artifact, dict):
        return artifact["model"]

    # If we saved only the model
    return artifact


# --------------------------------------------------
# Create features for the next hour
# --------------------------------------------------

def create_next_hour_features(history: pd.DataFrame):
    """
    Create the feature values required to predict
    the next hour.
    """

    history = history.copy()

    # Make sure timestamps are datetime
    history["timestamp"] = pd.to_datetime(
        history["timestamp"]
    )

    latest_timestamp = history["timestamp"].iloc[-1]

    next_timestamp = (
        latest_timestamp + pd.Timedelta(hours=1)
    )

    # Future time features
    hour = next_timestamp.hour
    day_of_week = next_timestamp.dayofweek

    is_weekend = int(day_of_week >= 5)

    # Historical values
    lag_1 = history["arrivals"].iloc[-1]
    lag_2 = history["arrivals"].iloc[-2]
    lag_3 = history["arrivals"].iloc[-3]
    lag_6 = history["arrivals"].iloc[-6]
    lag_24 = history["arrivals"].iloc[-24]

    # Rolling features
    rolling_mean_6 = (
        history["arrivals"]
        .iloc[-6:]
        .mean()
    )

    rolling_mean_24 = (
        history["arrivals"]
        .iloc[-24:]
        .mean()
    )

    features = pd.DataFrame(
        [{
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_holiday": 0,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_6": lag_6,
            "lag_24": lag_24,
            "rolling_mean_6": rolling_mean_6,
            "rolling_mean_24": rolling_mean_24,
        }]
    )

    return next_timestamp, features


# --------------------------------------------------
# Forecast next N hours
# --------------------------------------------------

def forecast_next_hours(
    history: pd.DataFrame,
    hours: int = 6,
):
    """
    Forecast ED arrivals for the next N hours.

    Parameters
    ----------
    history:
        Historical ED arrival data containing
        timestamp and arrivals columns.

    hours:
        Number of future hours to forecast.

    Returns
    -------
    pandas.DataFrame
        Future timestamps and predicted arrivals.
    """

    if hours < 1:
        raise ValueError(
            "hours must be greater than 0"
        )

    if hours > 12:
        raise ValueError(
            "Maximum forecasting horizon is 12 hours"
        )

    required_columns = {
        "timestamp",
        "arrivals",
    }

    missing_columns = (
        required_columns - set(history.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Need at least 24 hours of history
    if len(history) < 24:
        raise ValueError(
            "At least 24 hours of historical data "
            "are required."
        )

    history = history.copy()

    history["timestamp"] = pd.to_datetime(
        history["timestamp"]
    )

    history = history.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    model = load_model()

    predictions = []

    # Recursive forecasting
    for _ in range(hours):

        next_timestamp, features = (
            create_next_hour_features(history)
        )

        prediction = model.predict(features)[0]

        # Patient arrivals cannot be negative
        prediction = max(0, prediction)

        predictions.append({
            "timestamp": next_timestamp,
            "predicted_arrivals": prediction,
        })

        # Add prediction to history
        # so that it can be used to predict
        # the following hour.
        new_row = pd.DataFrame(
            [{
                "timestamp": next_timestamp,
                "arrivals": prediction,
            }]
        )

        history = pd.concat(
            [
                history,
                new_row,
            ],
            ignore_index=True,
        )

    return pd.DataFrame(predictions)
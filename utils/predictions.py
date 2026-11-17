import joblib
import xgboost as xgb
import pandas as pd
import numpy as np

from pathlib import Path


# ==========================================
# 1. LOAD TRAINED MODELS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "saved_models"

# Model A
model_A = joblib.load(
    MODEL_DIR / "wattwise_model_A_best.joblib"
)

# Model B metadata
model_B_metadata = joblib.load(
    MODEL_DIR / "wattwise_model_B_metadata.joblib"
)

# Model B XGBoost
model_B = xgb.Booster()

model_B.load_model(
    str(MODEL_DIR / "wattwise_model_B_xgboost.json")
)


# ==========================================
# 2. EXTRACT FEATURES AND CLASSIFIERS
# ==========================================

MODEL_A_FEATURES = model_A["features"]

MODEL_B_FEATURES = model_B_metadata["features"]

MODEL_A_CLASSIFIER = model_A["classification_model"]

MODEL_B_CLASSIFIER = model_B_metadata["classification_model"]

HISTORICAL_DAILY = model_A["historical_daily"].copy()

PEAK_HOUR = model_A["peak_hour"]

TARIFF = model_A["tariff"]

TARIFF_NOTE = TARIFF["note"]


# ==========================================
# 3. ELECTRICITY CHARGE CALCULATOR
# ==========================================

def calculate_charge_inr(kwh):

    remaining = max(float(kwh), 0.0)
    charge = 0.0

    for slab in TARIFF["slabs"]:

        limit = slab["up_to_units"]

        if limit is None:
            units = remaining
        else:
            units = min(remaining, limit)

        charge += units * slab["rate_inr_per_unit"]

        remaining -= units

        if remaining <= 0:
            break

    return round(charge + TARIFF["fixed_charge_inr"], 2)


# ==========================================
# 4. MODEL A - FUTURE FEATURE GENERATION
# ==========================================

def build_A_future_row(n_days):

    n_days = int(n_days)

    if not 1 <= n_days <= 30:
        raise ValueError("Forecast horizon must be between 1 and 30 days.")

    daily = HISTORICAL_DAILY.copy()

    daily["date"] = pd.to_datetime(daily["date"])

    daily = daily.sort_values("date")

    daily = daily.dropna(
        subset=[
            "daily_energy_kwh",
            "lag_1",
            "lag_7",
            "lag_14",
            "lag_30",
            "rolling_mean_7",
            "rolling_mean_14",
            "rolling_mean_30"
        ]
    )

    if daily.empty:
        raise ValueError("Insufficient historical data for forecasting.")

    last = daily.iloc[-1]

    base_date = pd.Timestamp(last["date"])

    future_date = base_date + pd.Timedelta(days=n_days)

    row = last.copy()

    row["future_days"] = n_days
    row["future_date"] = future_date

    row["future_dow"] = future_date.dayofweek
    row["future_day"] = future_date.day
    row["future_month"] = future_date.month
    row["future_dayofyear"] = future_date.dayofyear

    row["future_is_weekend"] = int(
        future_date.dayofweek >= 5
    )

    row["future_dow_sin"] = np.sin(
        2 * np.pi * future_date.dayofweek / 7
    )

    row["future_dow_cos"] = np.cos(
        2 * np.pi * future_date.dayofweek / 7
    )

    row["future_month_sin"] = np.sin(
        2 * np.pi * future_date.month / 12
    )

    row["future_month_cos"] = np.cos(
        2 * np.pi * future_date.month / 12
    )

    X_future = pd.DataFrame(
        [row]
    ).reindex(columns=MODEL_A_FEATURES)

    return X_future, future_date


# ==========================================
# 5. MODEL A - ENERGY FORECAST
# ==========================================

def predict_model_A(n_days):

    X_future, future_date = build_A_future_row(n_days)

    predicted_kwh = max(
        float(model_A["model"].predict(X_future)[0]),
        0.0
    )

    consumption_level = str(
        MODEL_A_CLASSIFIER.predict(X_future)[0]
    )

    monthly_kwh = predicted_kwh * 30

    return {
        "model": "Model A - Historical Forecasting",
        "future_days": int(n_days),
        "future_date": str(future_date.date()),
        "predicted_daily_kwh": round(predicted_kwh, 3),
        "consumption_level": consumption_level,
        "estimated_30_day_kwh": round(monthly_kwh, 2),
        "estimated_30_day_charge": calculate_charge_inr(monthly_kwh),
        "historical_peak_hour": PEAK_HOUR,
        "bill_note": TARIFF_NOTE
    }


# ==========================================
# 6. MODEL B - HOUSEHOLD PREDICTION
# ==========================================

def predict_model_B(household_input):

    missing = [
        feature
        for feature in MODEL_B_FEATURES
        if feature not in household_input
    ]

    if missing:
        raise ValueError(
            "Missing household inputs: " + ", ".join(missing)
        )

    X_new = pd.DataFrame(
        [
            {
                feature: household_input[feature]
                for feature in MODEL_B_FEATURES
            }
        ],
        columns=MODEL_B_FEATURES
    )

    for feature in MODEL_B_FEATURES:

        X_new[feature] = pd.to_numeric(
            X_new[feature],
            errors="coerce"
        )

    if X_new.isna().all(axis=1).iloc[0]:
        raise ValueError("Enter at least one valid numerical input.")

    if np.isinf(X_new.to_numpy(dtype=float)).any():
        raise ValueError("Inputs must contain finite numerical values.")

    # XGBoost prediction
    dmatrix = xgb.DMatrix(
        X_new,
        feature_names=MODEL_B_FEATURES
    )

    predicted_kwh = max(
        float(model_B.predict(dmatrix)[0]),
        0.0
    )

    # Consumption-level classification
    consumption_level = str(
        MODEL_B_CLASSIFIER.predict(X_new)[0]
    )

    monthly_kwh = predicted_kwh * 30

    return {
        "model": "Model B - New Household Prediction",
        "predicted_daily_kwh": round(predicted_kwh, 3),
        "consumption_level": consumption_level,
        "estimated_monthly_kwh": round(monthly_kwh, 2),
        "estimated_monthly_charge": calculate_charge_inr(monthly_kwh),
        "historical_peak_hour": PEAK_HOUR,
        "bill_note": TARIFF_NOTE,
        "data_limitation": model_B_metadata["target_description"]
    }


print("WATTWISE PREDICTION ENGINE INITIALIZED!")
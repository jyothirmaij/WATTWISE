"""
WattWise — Model artifacts and result exports

Contribution 5 of 5: Joblib model artifacts, tariff metadata, model summary JSON and output-folder packaging for deployment.

Execution note: run these contribution files in the order listed in README.md
using `%run -i` in Google Colab so shared variables remain in the notebook namespace.
"""

# =========================
# 20. SAVE FINAL MODELS AS JOBLIB
# =========================

import sklearn
import lightgbm
import xgboost

# Illustrative placeholders only. Verify the current applicable tariff
# before presenting an official-style bill estimate.
TARIFF_METADATA = {
    "type": "illustrative_marginal_energy_charge_only",
    "slabs": [
        {"up_to_units": 50, "rate_inr_per_unit": 3.15},
        {"up_to_units": 50, "rate_inr_per_unit": 4.25},
        {"up_to_units": 100, "rate_inr_per_unit": 5.50},
        {"up_to_units": 100, "rate_inr_per_unit": 6.75},
        {"up_to_units": None, "rate_inr_per_unit": 7.60}
    ],
    "fixed_charge_inr": 0,
    "note": TARIFF_NOTE
}

A_ARTIFACT = {
    "model": BEST_A_REG,
    "model_name": BEST_A_REG_NAME,
    "features": A_FEATURES,
    "classification_model": BEST_A_CLS,
    "classification_model_name": BEST_A_CLS_NAME,
    "classification_features": A_FEATURES,
    "level_q1": float(q1_A),
    "level_q2": float(q2_A),
    "historical_daily": A_DAILY,
    "peak_hour": int(peak_hour),
    "target_name": "estimated_average_household_daily_kwh",
    "target_description": (
        "Sampling-adjusted estimate of average household daily consumption "
        "from the pooled 20-household dataset; not an individual household forecast."
    ),
    "max_forecast_horizon_days": 30,
    "tariff": TARIFF_METADATA
}

B_ARTIFACT = {
    "model": BEST_B_REG,
    "model_name": BEST_B_REG_NAME,
    "features": B_FEATURES,
    "classification_model": BEST_B_CLS,
    "classification_model_name": BEST_B_CLS_NAME,
    "classification_features": B_FEATURES,
    "level_q1": float(q1_B),
    "level_q2": float(q2_B),
    "peak_hour_reference": int(peak_hour),
    "target_name": B_TARGET,
    "target_description": (
        "Daily household consumption predicted from synthetic household data; "
        "real-world generalisation has not been established."
    ),
    "tariff": TARIFF_METADATA
}

joblib.dump(A_ARTIFACT, MODEL_DIR / "wattwise_model_A_best.joblib")
joblib.dump(B_ARTIFACT, MODEL_DIR / "wattwise_model_B_best.joblib")

# Export evaluation outputs.
A_RESULTS_DF.to_csv(OUTPUT_DIR / "model_A_regression_comparison.csv", index=False)
A_CLS_RESULTS_DF.to_csv(OUTPUT_DIR / "model_A_classification_comparison.csv", index=False)
B_RESULTS_DF.to_csv(OUTPUT_DIR / "model_B_regression_comparison.csv", index=False)
B_CLS_RESULTS_DF.to_csv(OUTPUT_DIR / "model_B_classification_comparison.csv", index=False)
A_HORIZON_RESULTS.to_csv(OUTPUT_DIR / "model_A_horizon_metrics.csv", index=False)
A_DAILY.to_csv(OUTPUT_DIR / "model_A_daily_features.csv", index=False)

print("Saved model artifacts:")
print(MODEL_DIR / "wattwise_model_A_best.joblib")
print(MODEL_DIR / "wattwise_model_B_best.joblib")
print("Saved evaluation tables to:", OUTPUT_DIR)



# # Flask Web App Architecture
#
# Use the two `.joblib` artifacts from the previous cell.
#
# ### Model A page
# 1. Select existing historical household/fleet.
# 2. Enter `N days` from 1–30.
# 3. Predict exactly that future day.
# 4. Show:
#    - predicted kWh
#    - Low / Medium / High
#    - estimated charge
#    - historical peak hour
#    - energy-saving recommendations
#
# ### Model B page
# 1. Enter household details.
# 2. Predict daily kWh.
# 3. Calculate 30-day kWh and charge.
# 4. Show Low / Medium / High.
# 5. Show peak-hour information and recommendations.
#
# **Important:** the Flask app should load the saved joblib artifacts rather than retraining models for every request.



# =========================
# 21. EXPORT MODEL SUMMARY JSON
# =========================

next_day_row = A_HORIZON_RESULTS[
    A_HORIZON_RESULTS["future_days"] == 1
]

summary = {
    "project": "WattWise",
    "model_A": {
        "best_regressor": BEST_A_REG_NAME,
        "best_classifier": BEST_A_CLS_NAME,
        "overall_holdout_r2": float(A_RESULTS_DF.iloc[0]["R2"]),
        "overall_holdout_mae": float(A_RESULTS_DF.iloc[0]["MAE"]),
        "overall_holdout_rmse": float(A_RESULTS_DF.iloc[0]["RMSE"]),
        "next_day_metrics": (
            next_day_row.iloc[0][["MAE", "RMSE", "R2"]].to_dict()
            if not next_day_row.empty else None
        ),
        "target_definition": (
            "Sampling-adjusted estimated average household daily consumption "
            "from pooled observations, not individual household consumption."
        ),
        "target_date_split_cutoff": str(cutoff.date())
    },
    "model_B": {
        "best_regressor": BEST_B_REG_NAME,
        "best_classifier": BEST_B_CLS_NAME,
        "holdout_r2": float(B_RESULTS_DF.iloc[0]["R2"]),
        "holdout_mae": float(B_RESULTS_DF.iloc[0]["MAE"]),
        "holdout_rmse": float(B_RESULTS_DF.iloc[0]["RMSE"]),
        "data_note": (
            "Metrics are from a synthetic household dataset and do not prove "
            "real-world generalisation."
        )
    },
    "historical_pool_peak_hour": int(peak_hour),
    "tariff_note": TARIFF_NOTE,
    "software_versions": {
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "xgboost": xgboost.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__
    }
}

with open(OUTPUT_DIR / "wattwise_model_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(json.dumps(summary, indent=2, default=str))



import os

OUTPUT_DIR = "/content/drive/MyDrive/FINAL"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("WattWise output folder is ready:", OUTPUT_DIR)



import os

for root, dirs, files in os.walk(OUTPUT_DIR):
    for file in files:
        print(os.path.join(root, file))



import shutil
from google.colab import files

PROJECT_DIR = "/content/drive/MyDrive/FINAL"

shutil.make_archive(
    "/content/WattWise_New_Outputs",
    "zip",
    PROJECT_DIR
)

files.download("/content/WattWise_New_Outputs.zip")

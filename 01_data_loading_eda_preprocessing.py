"""
WattWise — Data loading, EDA and preprocessing

Contribution 1 of 5: Drive setup, library imports, dataset loading, data-quality inspection, outlier summaries and leakage-safe cleaning for both datasets.

Execution note: run these contribution files in the order listed in README.md
using `%run -i` in Google Colab so shared variables remain in the notebook namespace.
"""

from google.colab import drive
drive.mount('/content/drive')



# =========================
# 1. INSTALL / IMPORT
# =========================
!pip -q install lightgbm xgboost joblib flask

import os, warnings, json, joblib, math
import sklearn
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from IPython.display import display

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier

from google.colab import drive

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# CHANGE ONLY THIS FOLDER IF NEEDED
DATA_DIR = Path("/content/drive/MyDrive/FINAL")
MODEL_DIR = DATA_DIR / "saved_models"
OUTPUT_DIR = DATA_DIR / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Ready.")
print("DATA_DIR:", DATA_DIR)



# =========================
# 2. LOCATE DATA FILES
# =========================

# Google Drive is mounted in Cell 1.
# Keep both CSV files inside this folder:
# MyDrive/FINAL/

MODEL_A_PATH = DATA_DIR / "/content/drive/MyDrive/project/WattWise_synthetic_household_energy_100k_Model A.csv"
MODEL_B_PATH = DATA_DIR / "/content/drive/MyDrive/project/WattWise_Model_B_New_Household_100K.csv"

print("Model A exists:", MODEL_A_PATH.exists(), MODEL_A_PATH)
print("Model B exists:", MODEL_B_PATH.exists(), MODEL_B_PATH)

if not MODEL_A_PATH.exists() or not MODEL_B_PATH.exists():
    raise FileNotFoundError(
        "One or both CSV files were not found. "
        "Upload them to MyDrive/FINAL or update MODEL_A_PATH and MODEL_B_PATH."
    )



# =========================
# 3. LOAD DATA
# =========================
A_RAW = pd.read_csv(MODEL_A_PATH, parse_dates=["timestamp"])
B_RAW = pd.read_csv(MODEL_B_PATH)

print("Model A:", A_RAW.shape)
print("Model B:", B_RAW.shape)

display(A_RAW.head())
display(B_RAW.head())



# =========================
# 4. EDA — MISSING VALUES + DUPLICATES
# =========================
def missing_summary(df, name):
    out = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_%": (df.isna().mean().values * 100).round(3),
        "dtype": df.dtypes.astype(str).values
    })
    print(f"\\n{name} shape={df.shape}")
    print("Total missing cells:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    display(out[out["missing_count"] > 0].sort_values("missing_count", ascending=False))
    return out

A_MISS = missing_summary(A_RAW, "MODEL A")
B_MISS = missing_summary(B_RAW, "MODEL B")

# Simple missing-value visualization
def plot_missing(df, title):
    s = df.isna().sum()
    s = s[s > 0].sort_values(ascending=False)
    plt.figure(figsize=(10,4))
    if len(s):
        plt.bar(s.index.astype(str), s.values)
        plt.xticks(rotation=60, ha="right")
        plt.ylabel("Missing cells")
    else:
        plt.text(0.5, 0.5, "No missing values", ha="center", va="center")
        plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_missing(A_RAW, "Model A — Missing Values")
plot_missing(B_RAW, "Model B — Missing Values")



# =========================
# 5. REUSABLE OUTLIER FUNCTIONS
# =========================
def outlier_summary_iqr(df, cols):
    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(x) == 0:
            continue
        q1, q3 = x.quantile(.25), x.quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        n = ((x < lo) | (x > hi)).sum()
        rows.append([c, n, n/len(x)*100, lo, hi])
    return pd.DataFrame(rows, columns=["column","outlier_count","outlier_%","lower","upper"])

A_NUM = A_RAW.select_dtypes(include=np.number).columns.tolist()
B_NUM = B_RAW.select_dtypes(include=np.number).columns.tolist()

A_OUT = outlier_summary_iqr(A_RAW, A_NUM)
B_OUT = outlier_summary_iqr(B_RAW, B_NUM)

print("MODEL A outliers by IQR:")
display(A_OUT.sort_values("outlier_count", ascending=False).head(15))

print("MODEL B outliers by IQR:")
display(B_OUT.sort_values("outlier_count", ascending=False).head(15))

def plot_outlier_counts(summary, title):
    s = summary.sort_values("outlier_count", ascending=False).head(12)
    plt.figure(figsize=(10,4))
    plt.bar(s["column"].astype(str), s["outlier_count"])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("IQR outlier count")
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_outlier_counts(A_OUT, "Model A — IQR Outlier Counts")
plot_outlier_counts(B_OUT, "Model B — IQR Outlier Counts")



# =========================
# 6. DATA CLEANING — LEAKAGE-SAFE
# =========================

# Important:
# - Do not interpolate randomly sampled household observations.
# - Do not impute using full-dataset medians before train/test splitting.
# - Do not IQR-clip the target or predictors using the complete dataset.
# - Missing predictor values are handled inside each sklearn Pipeline.

# -------------------------
# MODEL A raw cleaning
# -------------------------
A = A_RAW.copy()

A["timestamp"] = pd.to_datetime(A["timestamp"], errors="coerce")
A = A.dropna(subset=["timestamp", "household_id"])
A = A.drop_duplicates().sort_values("timestamp").reset_index(drop=True)

A_NUMERIC = [
    "voltage", "current", "power_factor", "active_power_w",
    "reactive_power_var", "energy_consumption_kwh",
    "temperature_c", "humidity_percent", "electricity_charge_inr"
]

for c in A_NUMERIC:
    A[c] = pd.to_numeric(A[c], errors="coerce")

# Physically impossible values are treated as missing.
for c in [
    "voltage", "current", "active_power_w",
    "reactive_power_var", "energy_consumption_kwh",
    "electricity_charge_inr"
]:
    A.loc[A[c] < 0, c] = np.nan

A.loc[~A["power_factor"].between(0, 1), "power_factor"] = np.nan
A.loc[~A["humidity_percent"].between(0, 100), "humidity_percent"] = np.nan
A.loc[~A["temperature_c"].between(-20, 60), "temperature_c"] = np.nan

# Target observations must exist; do not invent target energy values.
A = A.dropna(subset=["energy_consumption_kwh"]).copy()

# -------------------------
# MODEL B raw cleaning
# -------------------------
B = B_RAW.copy().drop_duplicates().copy()

B_TARGET = "daily_energy_consumption_kwh"

for c in B.columns:
    if c not in ["household_id"]:
        B[c] = pd.to_numeric(B[c], errors="coerce")

# Remove invalid target rows.
B = B.dropna(subset=[B_TARGET])
B = B[B[B_TARGET] >= 0].copy()

# Household counts and physical quantities cannot be negative.
B_NONNEGATIVE = [
    "num_people", "house_size_sqft", "num_ac", "num_fans",
    "num_refrigerators", "num_tvs", "num_washing_machines",
    "num_lights", "num_computers", "avg_usage_hours"
]

for c in B_NONNEGATIVE:
    if c in B.columns:
        B.loc[B[c] < 0, c] = np.nan

if "humidity_percent" in B.columns:
    B.loc[~B["humidity_percent"].between(0, 100), "humidity_percent"] = np.nan

if "temperature_c" in B.columns:
    B.loc[~B["temperature_c"].between(-20, 60), "temperature_c"] = np.nan

print("CLEANING COMPLETED")
print("Model A shape:", A.shape)
print("Model B shape:", B.shape)
print("Model A missing values:", int(A.isna().sum().sum()))
print("Model B missing values:", int(B.isna().sum().sum()))
print("Model A duplicate rows:", int(A.duplicated().sum()))
print("Model B duplicate rows:", int(B.duplicated().sum()))




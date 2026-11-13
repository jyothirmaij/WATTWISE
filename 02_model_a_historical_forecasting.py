"""
WattWise — Model A historical forecasting

Contribution 2 of 5: Daily aggregation, sampling-adjusted target construction, calendar/cyclical/lag/rolling features, 1–30 day horizon dataset, regression/classification comparison, evaluation visuals and future-day prediction logic.

Execution note: run these contribution files in the order listed in README.md
using `%run -i` in Google Colab so shared variables remain in the notebook namespace.
"""

# ## Model A — Historical Energy Forecasting
#
# The supplied Model-A dataset contains 100,000 randomly sampled 15-minute observations from 20 households. There is approximately one observation per timestamp across the household pool, rather than a complete 96-reading daily series for every household.
#
# The daily target is therefore an **estimated average household daily consumption**, calculated as:
#
# `estimated_average_household_daily_kWh = sum(observed_interval_kWh) × 96 / valid_observed_intervals`
#
# This is a sampling-based estimate, not a directly measured complete household-day total. The website must describe it accordingly.
#
# The model uses:
# - lag 1 / 7 / 14 / 30
# - rolling mean 7 / 14 / 30
# - calendar and future-horizon features
# - direct prediction for a requested horizon from 1–30 days
#
# The validation split is based on the **target date**, not the source date. This prevents training samples whose future target falls in the holdout period.



# =========================
# 7. MODEL A — DAILY AGGREGATION
# =========================

A["date"] = A["timestamp"].dt.floor("D")
A["hour"] = A["timestamp"].dt.hour

# The data has roughly one sampled household observation per 15-minute timestamp.
# Estimate average household daily usage by scaling the observed interval mean to 96 intervals.
A_DAILY = (
    A.groupby("date")
     .agg(
         observed_energy_kwh=("energy_consumption_kwh", "sum"),
         observed_intervals=("energy_consumption_kwh", "count"),
         households_observed=("household_id", "nunique"),
         avg_active_power_w=("active_power_w", "mean"),
         avg_voltage=("voltage", "mean"),
         avg_current=("current", "mean"),
         avg_power_factor=("power_factor", "mean"),
         avg_temperature_c=("temperature_c", "mean"),
         avg_humidity_percent=("humidity_percent", "mean"),
     )
     .reset_index()
)

A_DAILY["daily_energy_kwh"] = (
    A_DAILY["observed_energy_kwh"] * 96
    / A_DAILY["observed_intervals"].clip(lower=1)
)

# Verify date continuity before using row-based lag features.
full_dates = pd.date_range(
    A_DAILY["date"].min(),
    A_DAILY["date"].max(),
    freq="D"
)

A_DAILY = (
    A_DAILY.set_index("date")
           .reindex(full_dates)
           .rename_axis("date")
           .reset_index()
)

# Recompute the sampling estimate for reindexed dates.
# Missing dates remain missing; they are not filled with fabricated targets.
A_DAILY["daily_energy_kwh"] = (
    A_DAILY["observed_energy_kwh"] * 96
    / A_DAILY["observed_intervals"].clip(lower=1)
)

# Historical peak hour from the cleaned observations.
hourly_power = A.groupby("hour")["active_power_w"].mean()
peak_hour = int(hourly_power.idxmax())
peak_mean_power = float(hourly_power.max())

# Calendar features
A_DAILY["dow"] = A_DAILY["date"].dt.dayofweek
A_DAILY["day"] = A_DAILY["date"].dt.day
A_DAILY["month"] = A_DAILY["date"].dt.month
A_DAILY["dayofyear"] = A_DAILY["date"].dt.dayofyear
A_DAILY["is_weekend"] = (A_DAILY["dow"] >= 5).astype(int)

A_DAILY["dow_sin"] = np.sin(2 * np.pi * A_DAILY["dow"] / 7)
A_DAILY["dow_cos"] = np.cos(2 * np.pi * A_DAILY["dow"] / 7)
A_DAILY["month_sin"] = np.sin(2 * np.pi * A_DAILY["month"] / 12)
A_DAILY["month_cos"] = np.cos(2 * np.pi * A_DAILY["month"] / 12)
A_DAILY["doy_sin"] = np.sin(2 * np.pi * A_DAILY["dayofyear"] / 365.25)
A_DAILY["doy_cos"] = np.cos(2 * np.pi * A_DAILY["dayofyear"] / 365.25)

# Past-only lag and rolling features.
for lag in [1, 7, 14, 30]:
    A_DAILY[f"lag_{lag}"] = A_DAILY["daily_energy_kwh"].shift(lag)

for window in [7, 14, 30]:
    A_DAILY[f"rolling_mean_{window}"] = (
        A_DAILY["daily_energy_kwh"].shift(1).rolling(window, min_periods=window).mean()
    )

A_DAILY = A_DAILY.sort_values("date").reset_index(drop=True)

display(A_DAILY.head(10))
print("Daily rows:", len(A_DAILY))
print("Date gaps:", int(A_DAILY["date"].diff().dt.days.gt(1).sum()))
print("Days with missing estimated target:", int(A_DAILY["daily_energy_kwh"].isna().sum()))
print("Historical peak hour:", peak_hour, "| mean power:", round(peak_mean_power, 2), "W")



# =========================
# 8. MODEL A — BUILD 1..30 DAY HORIZON DATASET
# =========================

HORIZONS = range(1, 31)
A_SUP = []

for h in HORIZONS:
    tmp = A_DAILY.copy()
    tmp["future_days"] = h
    tmp["future_date"] = tmp["date"] + pd.to_timedelta(h, unit="D")
    tmp["target_date"] = tmp["future_date"]

    tmp["future_dow"] = tmp["future_date"].dt.dayofweek
    tmp["future_day"] = tmp["future_date"].dt.day
    tmp["future_month"] = tmp["future_date"].dt.month
    tmp["future_dayofyear"] = tmp["future_date"].dt.dayofyear
    tmp["future_is_weekend"] = (tmp["future_dow"] >= 5).astype(int)

    tmp["future_dow_sin"] = np.sin(2 * np.pi * tmp["future_dow"] / 7)
    tmp["future_dow_cos"] = np.cos(2 * np.pi * tmp["future_dow"] / 7)
    tmp["future_month_sin"] = np.sin(2 * np.pi * tmp["future_month"] / 12)
    tmp["future_month_cos"] = np.cos(2 * np.pi * tmp["future_month"] / 12)

    # Shift by calendar days because A_DAILY has been reindexed to daily frequency.
    tmp["target_kwh"] = A_DAILY["daily_energy_kwh"].shift(-h)
    A_SUP.append(tmp)

A_SUP = pd.concat(A_SUP, ignore_index=True)

required = [
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
    "target_kwh", "target_date"
]
A_SUP = A_SUP.dropna(subset=required).reset_index(drop=True)

# Split by target date, not source date.
unique_target_dates = np.sort(A_SUP["target_date"].unique())
cutoff_position = int(np.floor(len(unique_target_dates) * 0.80))
cutoff_position = min(max(cutoff_position, 1), len(unique_target_dates) - 1)
cutoff = pd.Timestamp(unique_target_dates[cutoff_position - 1])

A_TRAIN = A_SUP[A_SUP["target_date"] <= cutoff].copy()
A_TEST = A_SUP[A_SUP["target_date"] > cutoff].copy()

A_DROP = [
    "date", "future_date", "target_date", "target_kwh",
    "daily_energy_kwh", "observed_energy_kwh", "observed_intervals", "households_observed"
]
A_FEATURES = [c for c in A_SUP.columns if c not in A_DROP]

X_A_TRAIN = A_TRAIN[A_FEATURES]
y_A_TRAIN = A_TRAIN["target_kwh"]
X_A_TEST = A_TEST[A_FEATURES]
y_A_TEST = A_TEST["target_kwh"]

print("A train:", X_A_TRAIN.shape)
print("A test :", X_A_TEST.shape)
print("Target-date cutoff:", cutoff)
print("Latest training target:", A_TRAIN["target_date"].max())
print("Earliest testing target:", A_TEST["target_date"].min())
print("Feature count:", len(A_FEATURES))

assert A_TRAIN["target_date"].max() < A_TEST["target_date"].min(), (
    "Temporal leakage: training and testing target dates overlap."
)
print("Target-date chronological split verified.")



# =========================
# 9. MODEL A — REGRESSION MODEL COMPARISON
# =========================
A_REG_MODELS = {
    "LinearRegression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "RandomForest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            max_features=0.8, n_jobs=-1, random_state=RANDOM_STATE
        ))
    ]),
    "ExtraTrees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2,
            max_features=0.9, n_jobs=-1, random_state=RANDOM_STATE
        ))
    ]),
    "HistGradientBoosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=0.2, random_state=RANDOM_STATE
        ))
    ]),
    "XGBoost": XGBRegressor(
        n_estimators=800, max_depth=6, learning_rate=0.035,
        subsample=0.9, colsample_bytree=0.9,
        objective="reg:squarederror", reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=1200, learning_rate=0.03,
        num_leaves=31, max_depth=-1,
        min_child_samples=20, subsample=0.9,
        colsample_bytree=0.9, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
    )
}

A_RESULTS = []
A_FITTED = {}

for name, model in A_REG_MODELS.items():
    model.fit(X_A_TRAIN, y_A_TRAIN)
    pred = model.predict(X_A_TEST)
    A_FITTED[name] = model

    A_RESULTS.append({
        "Model": name,
        "MAE": mean_absolute_error(y_A_TEST, pred),
        "RMSE": np.sqrt(mean_squared_error(y_A_TEST, pred)),
        "R2": r2_score(y_A_TEST, pred)
    })

A_RESULTS_DF = pd.DataFrame(A_RESULTS).sort_values(["MAE", "RMSE"], ascending=[True, True]).reset_index(drop=True)
display(A_RESULTS_DF)

BEST_A_REG_NAME = A_RESULTS_DF.iloc[0]["Model"]
BEST_A_REG = A_FITTED[BEST_A_REG_NAME]
print("BEST MODEL A REGRESSOR:", BEST_A_REG_NAME)



# =========================
# 10. MODEL A — CLASSIFICATION
# =========================
# Levels are created from TRAINING target quantiles only.
q1_A, q2_A = y_A_TRAIN.quantile([1/3, 2/3])

def make_level(y):
    return pd.cut(
        y,
        bins=[-np.inf, q1_A, q2_A, np.inf],
        labels=["Low","Medium","High"]
    )

y_A_TRAIN_cls = make_level(y_A_TRAIN)
y_A_TEST_cls = make_level(y_A_TEST)

A_CLS_MODELS = {
    "LogisticRegression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]),
    "RandomForestClassifier": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2,
            max_features=0.8, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE
        ))
    ]),
    "ExtraTreesClassifier": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2,
            max_features=0.9, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE
        ))
    ]),
    "HistGradientBoostingClassifier": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05,
            max_leaf_nodes=31, random_state=RANDOM_STATE
        ))
    ]),
    "LightGBMClassifier": LGBMClassifier(
        n_estimators=700, learning_rate=0.035, num_leaves=31,
        max_depth=-1, subsample=0.9, colsample_bytree=0.9,
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=-1
    )
}

A_CLS_RESULTS = []
A_CLS_FITTED = {}

for name, model in A_CLS_MODELS.items():
    model.fit(X_A_TRAIN, y_A_TRAIN_cls)
    pred_cls = model.predict(X_A_TEST)
    A_CLS_FITTED[name] = model

    A_CLS_RESULTS.append({
        "Model": name,
        "Accuracy": accuracy_score(y_A_TEST_cls, pred_cls),
        "Precision_weighted": precision_score(y_A_TEST_cls, pred_cls, average="weighted", zero_division=0),
        "Recall_weighted": recall_score(y_A_TEST_cls, pred_cls, average="weighted", zero_division=0),
        "F1_weighted": f1_score(y_A_TEST_cls, pred_cls, average="weighted", zero_division=0)
    })

A_CLS_RESULTS_DF = pd.DataFrame(A_CLS_RESULTS).sort_values(["F1_weighted", "Accuracy"], ascending=[False, False]).reset_index(drop=True)
display(A_CLS_RESULTS_DF)

BEST_A_CLS_NAME = A_CLS_RESULTS_DF.iloc[0]["Model"]
BEST_A_CLS = A_CLS_FITTED[BEST_A_CLS_NAME]
print("BEST MODEL A CLASSIFIER:", BEST_A_CLS_NAME)
print("TRAINING LEVEL CUTS:", q1_A, q2_A)



# =========================
# 11. MODEL A — TEST VISUALS
# =========================
best_A_pred = BEST_A_REG.predict(X_A_TEST)

plt.figure(figsize=(6,5))
plt.scatter(y_A_TEST, best_A_pred, alpha=0.25)
lims=[min(y_A_TEST.min(), best_A_pred.min()), max(y_A_TEST.max(), best_A_pred.max())]
plt.plot(lims, lims, linestyle="--")
plt.xlabel("Actual kWh")
plt.ylabel("Predicted kWh")
plt.title(f"Model A — Actual vs Predicted ({BEST_A_REG_NAME})")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
resid = y_A_TEST.values - best_A_pred
plt.hist(resid, bins=40)
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Count")
plt.title("Model A — Residual Distribution")
plt.tight_layout()
plt.show()

print(classification_report(
    y_A_TEST_cls,
    BEST_A_CLS.predict(X_A_TEST),
    digits=4
))


# Horizon-specific metrics: especially inspect horizon=1 for next-day claims.
A_TEST_EVAL = A_TEST[["future_days", "target_date", "target_kwh"]].copy()
A_TEST_EVAL["predicted_kwh"] = best_A_pred

A_HORIZON_RESULTS = (
    A_TEST_EVAL.groupby("future_days")
    .apply(
        lambda g: pd.Series({
            "MAE": mean_absolute_error(g["target_kwh"], g["predicted_kwh"]),
            "RMSE": np.sqrt(mean_squared_error(g["target_kwh"], g["predicted_kwh"])),
            "R2": r2_score(g["target_kwh"], g["predicted_kwh"])
                if len(g) > 1 else np.nan,
            "rows": len(g)
        }),
        include_groups=False
    )
    .reset_index()
)

print("Model A performance by forecast horizon:")
display(A_HORIZON_RESULTS)

print("Next-day (horizon=1) metrics:")
display(A_HORIZON_RESULTS[A_HORIZON_RESULTS["future_days"] == 1])


# Horizon-specific metrics: especially inspect horizon=1 for next-day claims.
A_TEST_EVAL = A_TEST[["future_days", "target_date", "target_kwh"]].copy()
A_TEST_EVAL["predicted_kwh"] = best_A_pred

horizon_rows = []
for horizon, group in A_TEST_EVAL.groupby("future_days"):
    horizon_rows.append({
        "future_days": int(horizon),
        "MAE": mean_absolute_error(group["target_kwh"], group["predicted_kwh"]),
        "RMSE": np.sqrt(mean_squared_error(group["target_kwh"], group["predicted_kwh"])),
        "R2": r2_score(group["target_kwh"], group["predicted_kwh"])
              if len(group) > 1 else np.nan,
        "rows": len(group)
    })

A_HORIZON_RESULTS = pd.DataFrame(horizon_rows)
print("Model A performance by forecast horizon:")
display(A_HORIZON_RESULTS)
print("Next-day (horizon=1) metrics:")
display(A_HORIZON_RESULTS[A_HORIZON_RESULTS["future_days"] == 1])



# ## Model A prediction logic
#
# The web app asks for **N = 1–30 days**.
#
# It does **not** generate 30 predictions when the user asks for one day. It creates exactly one future row with `future_days=N`, uses the latest available lag/rolling features, and predicts the consumption for that requested future day.



# =========================
# 12. MODEL A — SINGLE FUTURE-DAY PREDICTION
# =========================

# Illustrative marginal energy rates only. Verify current tariff and charges
# before presenting this as an actual KSEB bill.
TARIFF_SLABS = [
    (50, 3.15), (50, 4.25), (100, 5.50), (100, 6.75), (float("inf"), 7.60)
]
TARIFF_NOTE = (
    "Illustrative energy-charge estimate only; not an official electricity bill."
)

def calculate_charge_inr(kwh, fixed_charge=0.0, tariff_slabs=TARIFF_SLABS):
    remaining = max(float(kwh), 0.0)
    energy_charge = 0.0
    for slab_limit, rate in tariff_slabs:
        units = min(remaining, slab_limit)
        energy_charge += units * rate
        remaining -= units
        if remaining <= 0:
            break
    return round(energy_charge + float(fixed_charge), 2)


def build_A_future_row(n_days, daily_df=A_DAILY):
    n_days = int(n_days)
    if not 1 <= n_days <= 30:
        raise ValueError("n_days must be between 1 and 30.")

    d = daily_df.sort_values("date").copy()
    d = d.dropna(subset=[
        "daily_energy_kwh", "lag_1", "lag_7", "lag_14", "lag_30",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_30"
    ])
    if d.empty:
        raise ValueError("Insufficient valid historical data to create a forecast.")

    last = d.iloc[-1]
    base_date = pd.Timestamp(last["date"])
    future_date = base_date + pd.Timedelta(days=n_days)

    row = last.copy()
    row["future_days"] = n_days
    row["future_date"] = future_date
    row["future_dow"] = future_date.dayofweek
    row["future_day"] = future_date.day
    row["future_month"] = future_date.month
    row["future_dayofyear"] = future_date.dayofyear
    row["future_is_weekend"] = int(future_date.dayofweek >= 5)
    row["future_dow_sin"] = np.sin(2 * np.pi * future_date.dayofweek / 7)
    row["future_dow_cos"] = np.cos(2 * np.pi * future_date.dayofweek / 7)
    row["future_month_sin"] = np.sin(2 * np.pi * future_date.month / 12)
    row["future_month_cos"] = np.cos(2 * np.pi * future_date.month / 12)

    X_future = pd.DataFrame([row]).reindex(columns=A_FEATURES)
    return X_future, future_date


def predict_model_A(n_days):
    X_future, future_date = build_A_future_row(n_days)
    pred_kwh = max(float(BEST_A_REG.predict(X_future)[0]), 0.0)
    predicted_level = str(BEST_A_CLS.predict(X_future)[0])

    # Scenario only: assumes this daily estimate repeats for 30 days.
    estimated_30_day_kwh = pred_kwh * 30
    estimated_30_day_charge = calculate_charge_inr(estimated_30_day_kwh)

    return {
        "requested_future_day": int(n_days),
        "future_date": str(future_date.date()),
        "predicted_daily_kwh": round(pred_kwh, 3),
        "consumption_level": predicted_level,
        "estimated_30_day_kwh_if_repeated": round(estimated_30_day_kwh, 2),
        "estimated_30_day_energy_charge_inr": estimated_30_day_charge,
        "historical_pool_peak_hour": int(peak_hour),
        "bill_estimate_note": TARIFF_NOTE
    }


display(pd.DataFrame([
    predict_model_A(1), predict_model_A(7), predict_model_A(30)
]))

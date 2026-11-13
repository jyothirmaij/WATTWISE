"""
WattWise — Model B new-household prediction

Contribution 3 of 5: Model B feature selection, train/test split, regression and classification model comparisons, evaluation plots, and prediction function for a new household.

Execution note: run these contribution files in the order listed in README.md
using `%run -i` in Google Colab so shared variables remain in the notebook namespace.
"""

# ## Model B — New Household
#
# Model B predicts daily household consumption from household and appliance characteristics.
#
# The following columns are excluded:
# - `household_id`: identifier, not a predictive household characteristic.
# - `estimated_monthly_energy_kwh`: derived from the target.
# - `estimated_monthly_charge_inr`: derived from the target.
# - `peak_hour`: not a practical required input for a new household user.
#
# The target is synthetic. Its held-out metrics measure performance on this synthetic dataset and must not be presented as proven real-world accuracy. The web form should ask only for features that a user can reasonably provide.



# =========================
# 13. MODEL B — FEATURES + TARGET
# =========================

B_TARGET = "daily_energy_consumption_kwh"

B_DROP = [
    "household_id",
    B_TARGET,
    "estimated_monthly_energy_kwh",
    "estimated_monthly_charge_inr",
    "peak_hour"
]

B_FEATURES = [c for c in B.columns if c not in B_DROP]

X_B = B[B_FEATURES].copy()
y_B = B[B_TARGET].copy()

# The supplied data has one row per household ID, so a reproducible
# random holdout is appropriate for this synthetic tabular dataset.
from sklearn.model_selection import train_test_split

X_B_TRAIN, X_B_TEST, y_B_TRAIN, y_B_TEST = train_test_split(
    X_B, y_B, test_size=0.20, random_state=RANDOM_STATE
)

print("Model B features:", B_FEATURES)
print("Training:", X_B_TRAIN.shape)
print("Testing:", X_B_TEST.shape)
print("Unique household IDs:", B["household_id"].nunique())



# =========================
# 14. MODEL B — REGRESSION COMPARISON
# =========================
B_REG_MODELS = {
    "LinearRegression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "RandomForest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            max_features=0.9, n_jobs=-1, random_state=RANDOM_STATE
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
            max_iter=500, learning_rate=0.04, max_leaf_nodes=31,
            l2_regularization=0.2, random_state=RANDOM_STATE
        ))
    ]),
    "XGBoost": XGBRegressor(
        n_estimators=900, max_depth=6, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9,
        objective="reg:squarederror", reg_lambda=1,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=1200, learning_rate=0.03, num_leaves=31,
        max_depth=-1, min_child_samples=20,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=-1
    )
}


B_RESULTS = []
B_FITTED = {}

for name, model in B_REG_MODELS.items():
    model.fit(X_B_TRAIN, y_B_TRAIN)
    pred = model.predict(X_B_TEST)
    B_FITTED[name] = model

    B_RESULTS.append({
        "Model": name,
        "MAE": mean_absolute_error(y_B_TEST, pred),
        "RMSE": np.sqrt(mean_squared_error(y_B_TEST, pred)),
        "R2": r2_score(y_B_TEST, pred)
    })

B_RESULTS_DF = pd.DataFrame(B_RESULTS).sort_values(["MAE", "RMSE"], ascending=[True, True]).reset_index(drop=True)
display(B_RESULTS_DF)

BEST_B_REG_NAME = B_RESULTS_DF.iloc[0]["Model"]
BEST_B_REG = B_FITTED[BEST_B_REG_NAME]
print("BEST MODEL B REGRESSOR:", BEST_B_REG_NAME)



# =========================
# 15. MODEL B — CLASSIFICATION
# =========================
q1_B, q2_B = y_B_TRAIN.quantile([1/3, 2/3])

def make_B_level(y):
    return pd.cut(
        y,
        bins=[-np.inf, q1_B, q2_B, np.inf],
        labels=["Low","Medium","High"]
    )

y_B_TRAIN_cls = make_B_level(y_B_TRAIN)
y_B_TEST_cls = make_B_level(y_B_TEST)

B_CLS_MODELS = {
    "LogisticRegression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]),
    "RandomForestClassifier": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2,
            max_features=0.9, class_weight="balanced",
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
        subsample=0.9, colsample_bytree=0.9,
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=-1
    )
}

B_CLS_RESULTS = []
B_CLS_FITTED = {}

for name, model in B_CLS_MODELS.items():
    model.fit(X_B_TRAIN, y_B_TRAIN_cls)
    pred_cls = model.predict(X_B_TEST)
    B_CLS_FITTED[name] = model

    B_CLS_RESULTS.append({
        "Model": name,
        "Accuracy": accuracy_score(y_B_TEST_cls, pred_cls),
        "Precision_weighted": precision_score(y_B_TEST_cls, pred_cls, average="weighted", zero_division=0),
        "Recall_weighted": recall_score(y_B_TEST_cls, pred_cls, average="weighted", zero_division=0),
        "F1_weighted": f1_score(y_B_TEST_cls, pred_cls, average="weighted", zero_division=0)
    })

B_CLS_RESULTS_DF = pd.DataFrame(B_CLS_RESULTS).sort_values(["F1_weighted", "Accuracy"], ascending=[False, False]).reset_index(drop=True)
display(B_CLS_RESULTS_DF)

BEST_B_CLS_NAME = B_CLS_RESULTS_DF.iloc[0]["Model"]
BEST_B_CLS = B_CLS_FITTED[BEST_B_CLS_NAME]

print("BEST MODEL B CLASSIFIER:", BEST_B_CLS_NAME)
print("TRAINING LEVEL CUTS:", q1_B, q2_B)



# =========================
# 16. MODEL B — VISUALS
# =========================
best_B_pred = BEST_B_REG.predict(X_B_TEST)

plt.figure(figsize=(6,5))
plt.scatter(y_B_TEST, best_B_pred, alpha=0.25)
lims=[min(y_B_TEST.min(), best_B_pred.min()), max(y_B_TEST.max(), best_B_pred.max())]
plt.plot(lims, lims, linestyle="--")
plt.xlabel("Actual daily kWh")
plt.ylabel("Predicted daily kWh")
plt.title(f"Model B — Actual vs Predicted ({BEST_B_REG_NAME})")
plt.tight_layout()
plt.show()

print(classification_report(
    y_B_TEST_cls,
    BEST_B_CLS.predict(X_B_TEST),
    digits=4
))



# =========================
# 17. MODEL B — NEW HOUSEHOLD PREDICTION
# =========================

def calculate_monthly_cost_from_daily(daily_kwh):
    return calculate_charge_inr(float(daily_kwh) * 30)


def predict_model_B(household_input: dict):
    missing_features = [c for c in B_FEATURES if c not in household_input]
    if missing_features:
        raise ValueError(
            "Missing required household inputs: " + ", ".join(missing_features)
        )

    # Ignore extra keys and preserve the exact training feature order.
    X_new = pd.DataFrame(
        [{c: household_input[c] for c in B_FEATURES}],
        columns=B_FEATURES
    )

    for c in B_FEATURES:
        X_new[c] = pd.to_numeric(X_new[c], errors="coerce")

    if X_new.isna().all(axis=1).iloc[0]:
        raise ValueError("At least one valid household input is required.")

    daily_kwh = max(float(BEST_B_REG.predict(X_new)[0]), 0.0)
    predicted_level = str(BEST_B_CLS.predict(X_new)[0])

    monthly_kwh = daily_kwh * 30
    monthly_cost = calculate_charge_inr(monthly_kwh)

    return {
        "predicted_daily_kwh": round(daily_kwh, 3),
        "estimated_monthly_kwh_if_repeated": round(monthly_kwh, 2),
        "estimated_monthly_energy_charge_inr": monthly_cost,
        "consumption_level": predicted_level,
        "historical_pool_peak_hour": int(peak_hour),
        "bill_estimate_note": TARIFF_NOTE
    }


# Example household — peak_hour is intentionally not required.
example_household = {
    "num_people": 4,
    "house_size_sqft": 1500,
    "num_ac": 1,
    "num_fans": 4,
    "num_refrigerators": 1,
    "num_tvs": 2,
    "num_washing_machines": 1,
    "num_lights": 10,
    "num_computers": 2,
    "avg_usage_hours": 5.0,
    "temperature_c": 28.0,
    "humidity_percent": 70.0
}

display(pd.DataFrame([predict_model_B(example_household)]))

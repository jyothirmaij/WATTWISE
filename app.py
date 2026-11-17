from flask import Flask, render_template, request
from pathlib import Path
import pandas as pd
import json
import plotly.express as px

from utils.predictions import predict_model_A, predict_model_B


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# MODEL A - HISTORICAL FORECASTING
# ==========================================

@app.route("/forecast", methods=["GET", "POST"])
def forecast():

    result = None
    error = None

    if request.method == "POST":
        try:
            days = int(request.form.get("days", 1))

            if not 1 <= days <= 30:
                raise ValueError("Select a forecast period between 1 and 30 days.")

            result = predict_model_A(days)

        except Exception as e:
            error = str(e)

    return render_template(
        "forecast.html",
        result=result,
        error=error
    )


# ==========================================
# MODEL B - HOUSEHOLD PREDICTION
# ==========================================

@app.route("/household", methods=["GET", "POST"])
def household():

    result = None
    error = None
    household_input = {}

    if request.method == "POST":

        try:

            features = [
                "num_people",
                "house_size_sqft",
                "num_ac",
                "num_fans",
                "num_refrigerators",
                "num_tvs",
                "num_washing_machines",
                "num_lights",
                "num_computers",
                "avg_usage_hours",
                "temperature_c",
                "humidity_percent"
            ]

            for feature in features:

                value = request.form.get(feature)

                if value is None or value.strip() == "":
                    raise ValueError(
                        f"Please enter {feature.replace('_', ' ')}."
                    )

                household_input[feature] = float(value)

            result = predict_model_B(household_input)

        except Exception as e:
            error = str(e)

    return render_template(
        "household.html",
        result=result,
        error=error,
        form_data=household_input
    )


# ==========================================
# ANALYTICS DASHBOARD
# ==========================================

@app.route("/dashboard")
def dashboard():

    # Load existing output files

    daily_df = pd.read_csv(
        OUTPUT_DIR / "model_A_daily_features.csv"
    )

    model_a_reg = pd.read_csv(
        OUTPUT_DIR / "model_A_regression_comparison.csv"
    )

    model_b_reg = pd.read_csv(
        OUTPUT_DIR / "model_B_regression_comparison.csv"
    )

    model_a_clf = pd.read_csv(
        OUTPUT_DIR / "model_A_classification_comparison.csv"
    )

    model_b_clf = pd.read_csv(
        OUTPUT_DIR / "model_B_classification_comparison.csv"
    )

    horizon_df = pd.read_csv(
        OUTPUT_DIR / "model_A_horizon_metrics.csv"
    )

    with open(
        OUTPUT_DIR / "wattwise_model_summary.json",
        "r"
    ) as file:
        summary = json.load(file)

    # Prepare historical data

    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.sort_values("date")

    daily_df["month"] = (
        daily_df["date"].dt.to_period("M").astype(str)
    )

    daily_df["day_type"] = daily_df["date"].dt.dayofweek.map(
        lambda x: "Weekend" if x >= 5 else "Weekday"
    )

    # KPI calculations

    total_records = len(daily_df)

    avg_consumption = round(
        daily_df["daily_energy_kwh"].mean(), 2
    )

    peak_consumption = round(
        daily_df["daily_energy_kwh"].max(), 2
    )

    latest_date = daily_df["date"].max().strftime("%d %b %Y")

    # Monthly consumption

    monthly_df = (
        daily_df.groupby("month", as_index=False)["daily_energy_kwh"]
        .mean()
    )

    # Weekday vs Weekend

    day_df = (
        daily_df.groupby("day_type", as_index=False)["daily_energy_kwh"]
        .mean()
    )

    # Historical consumption trend

    trend_fig = px.line(
        daily_df,
        x="date",
        y="daily_energy_kwh",
        title="Historical Daily Energy Consumption",
        labels={
            "date": "Date",
            "daily_energy_kwh": "Energy Consumption (kWh)"
        }
    )

    trend_fig.update_layout(template="plotly_dark")

    # Monthly consumption chart

    monthly_fig = px.bar(
        monthly_df,
        x="month",
        y="daily_energy_kwh",
        title="Average Daily Consumption by Month",
        labels={
            "month": "Month",
            "daily_energy_kwh": "Average Consumption (kWh)"
        }
    )

    monthly_fig.update_layout(template="plotly_dark")

    # Weekday and weekend chart

    day_fig = px.bar(
        day_df,
        x="day_type",
        y="daily_energy_kwh",
        title="Weekday vs Weekend Consumption",
        labels={
            "day_type": "Day Type",
            "daily_energy_kwh": "Average Consumption (kWh)"
        }
    )

    day_fig.update_layout(template="plotly_dark")

    # Model A regression comparison

    a_reg_fig = px.bar(
        model_a_reg,
        x="Model",
        y="R2",
        title="Model A – Regression R² Comparison",
        labels={"R2": "R² Score"}
    )

    a_reg_fig.update_layout(template="plotly_dark")

    # Model B regression comparison

    b_reg_fig = px.bar(
        model_b_reg,
        x="Model",
        y="R2",
        title="Model B – Regression R² Comparison",
        labels={"R2": "R² Score"}
    )

    b_reg_fig.update_layout(template="plotly_dark")

    # Model A classification comparison

    a_clf_fig = px.bar(
        model_a_clf,
        x="Model",
        y="F1_weighted",
        title="Model A – Classification F1 Comparison",
        labels={"F1_weighted": "Weighted F1 Score"}
    )

    a_clf_fig.update_layout(template="plotly_dark")

    # Model B classification comparison

    b_clf_fig = px.bar(
        model_b_clf,
        x="Model",
        y="F1_weighted",
        title="Model B – Classification F1 Comparison",
        labels={"F1_weighted": "Weighted F1 Score"}
    )

    b_clf_fig.update_layout(template="plotly_dark")

    # Forecast horizon analysis

    horizon_fig = px.line(
        horizon_df,
        x="future_days",
        y="MAE",
        markers=True,
        title="Model A – Forecast Horizon vs MAE",
        labels={
            "future_days": "Forecast Horizon (Days)",
            "MAE": "Mean Absolute Error"
        }
    )

    horizon_fig.update_layout(template="plotly_dark")

    # Convert charts to HTML

    trend_chart = trend_fig.to_html(
        full_html=False,
        include_plotlyjs="cdn"
    )

    monthly_chart = monthly_fig.to_html(
        full_html=False,
        include_plotlyjs=False
    )

    day_chart = day_fig.to_html(
        full_html=False,
        include_plotlyjs=False
    )

    a_reg_chart = a_reg_fig.to_html(
        full_html=False,
        include_plotlyjs=False
    )

    b_reg_chart = b_reg_fig.to_html(
        full_html=False,
        include_plotlyjs=False
    )

    a_clf_chart = a_clf_fig.to_html(
        full_html=False,
        include_plotlyjs=False
    )

    b_clf_chart = b_clf_fig.to_html(
        full_html=False,
        include_plotlyjs=False
    )

    horizon_chart = horizon_fig.to_html(
        full_html=False,
        include_plotlyjs=False
    )

    # Model performance summary

    model_a_r2 = round(
        summary["model_A"]["overall_holdout_r2"], 4
    )

    model_b_r2 = round(
        summary["model_B"]["holdout_r2"], 4
    )

    return render_template(
        "dashboard.html",

        total_records=total_records,
        avg_consumption=avg_consumption,
        peak_consumption=peak_consumption,
        latest_date=latest_date,

        model_a_r2=model_a_r2,
        model_b_r2=model_b_r2,

        trend_chart=trend_chart,
        monthly_chart=monthly_chart,
        day_chart=day_chart,

        a_reg_chart=a_reg_chart,
        b_reg_chart=b_reg_chart,

        a_clf_chart=a_clf_chart,
        b_clf_chart=b_clf_chart,

        horizon_chart=horizon_chart
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
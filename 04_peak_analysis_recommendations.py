"""  
WattWise — Peak-hour analysis and energy recommendations

Contribution 4 of 5: Historical pooled peak-hour analysis and rule-based energy-saving recommendation generation.

Execution note: run these contribution files in the order listed in README.md
using `%run -i` in Google Colab so shared variables remain in the notebook namespace.
"""

# =========================
# 18. PEAK-TIME ANALYSIS
# =========================
A_HOURLY = (
    A.assign(hour=A["timestamp"].dt.hour)
     .groupby("hour")
     .agg(
         avg_power_w=("active_power_w","mean"),
         total_energy_kwh=("energy_consumption_kwh","sum")
     )
     .reset_index()
)

A_HOURLY["is_peak"] = A_HOURLY["avg_power_w"] == A_HOURLY["avg_power_w"].max()

plt.figure(figsize=(10,4))
plt.plot(A_HOURLY["hour"], A_HOURLY["avg_power_w"], marker="o")
plt.xlabel("Hour of day")
plt.ylabel("Average active power (W)")
plt.title("WattWise — Average Power by Hour")
plt.xticks(range(24))
plt.grid(alpha=.2)
plt.show()

print("Peak hour:", int(A_HOURLY.loc[A_HOURLY["avg_power_w"].idxmax(),"hour"]))
print("Peak average power:", round(A_HOURLY["avg_power_w"].max(),2), "W")



# =========================
# 19. ENERGY-SAVING RECOMMENDATION ENGINE
# =========================

def recommendations(predicted_kwh, peak_hour, model_type="B"):
    if model_type.upper() == "A":
        low_cut, high_cut = q1_A, q2_A
    else:
        low_cut, high_cut = q1_B, q2_B

    rec = []

    if predicted_kwh > high_cut:
        rec.append(
            "High estimated consumption: review AC, water-heater and other "
            "high-load appliance usage; check for unnecessary standby loads."
        )
    elif predicted_kwh < low_cut:
        rec.append(
            "Estimated consumption is relatively low compared with the "
            "model's training distribution. Continue monitoring usage."
        )
    else:
        rec.append(
            "Moderate estimated consumption: focus on efficient appliance "
            "operation and avoid overlapping high-load usage."
        )

    rec.append(
        f"The pooled historical dataset shows its highest average power "
        f"around {int(peak_hour):02d}:00. This is a historical pattern, "
        "not a personalised household peak forecast."
    )
    rec.append(
        "Use energy-efficient LED lighting and switch appliances off when "
        "they are not required."
    )
    rec.append(
        "Where practical, shift flexible appliance use away from the "
        "historically observed peak period."
    )
    return rec


example_daily_kwh = float(
    predict_model_B(example_household)["predicted_daily_kwh"]
)

print("\n".join(
    "• " + item
    for item in recommendations(
        example_daily_kwh, peak_hour, model_type="B"
    )
))

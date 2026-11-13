# WattWise — Five GitHub Contribution Files

This repository split organizes the original WattWise Colab pipeline into five code files so the five team members can own and commit separate contributions.

## File ownership suggestion

| Team member | File | Contribution |
|---|---|---|
| Member 1 | `01_data_loading_eda_preprocessing.py` | Google Drive setup, imports, loading, EDA, outlier inspection and cleaning |
| Member 2 | `02_model_a_historical_forecasting.py` | Model A daily aggregation, feature engineering, horizon dataset, training, evaluation and forecast logic |
| Member 3 | `03_model_b_household_prediction.py` | Model B feature preparation, training, classification, evaluation and new-household prediction |
| Member 4 | `04_peak_analysis_recommendations.py` | Historical peak-hour analysis and energy-saving recommendation rules |
| Member 5 | `05_model_artifacts_and_exports.py` | Model serialization, metadata, summary JSON and output packaging |

Replace Member 1–5 with your actual names.

## Important: execution order

These files are separated by project responsibility, but they intentionally preserve the shared variables used by the original Colab notebook. They are **not independent standalone scripts**.

1. Open a Google Colab notebook.
2. Upload/clone this folder into the Colab working directory.
3. Run each file in order using IPython's interactive namespace:

```python
%run -i 01_data_loading_eda_preprocessing.py
%run -i 02_model_a_historical_forecasting.py
%run -i 03_model_b_household_prediction.py
%run -i 04_peak_analysis_recommendations.py
%run -i 05_model_artifacts_and_exports.py
```

The `-i` option keeps variables such as `A`, `B`, `A_DAILY`, `BEST_A_REG`, and `BEST_B_REG` in the shared notebook namespace.

## Dependencies

The notebook installs/uses packages including:
- numpy
- pandas
- matplotlib
- scikit-learn
- lightgbm
- xgboost
- joblib

The first file also uses Google Colab Drive integration. The pipeline expects both CSV datasets at the paths configured in the original code. Update those paths if your Drive folder differs.

## How to show genuine team contributions on GitHub

Each member should:
1. Sign in with their own GitHub account.
2. Add or edit their assigned file.
3. Commit their own work with a meaningful commit message, for example:
   - `Add data loading and preprocessing pipeline`
   - `Implement Model A forecasting workflow`
   - `Add Model B household prediction`
   - `Add peak-hour analysis and energy recommendations`
   - `Export trained models and evaluation metadata`
4. Push the commit to the shared repository (or open a pull request if using branches).

Do not create commits under another person's account or claim work they did not perform. The commit history should reflect the work each member actually completed.

## Project limitations to retain in the README/presentation

- Model A produces a sampling-adjusted **average-household daily consumption estimate** from pooled observations; it is not an individually measured household forecast.
- Model B uses a synthetic household dataset. Its holdout metrics do not prove real-world generalization.
- Tariff values are illustrative energy-charge assumptions, not an official electricity bill.
- Peak hour is a pooled historical reference, not a personalised household peak forecast.

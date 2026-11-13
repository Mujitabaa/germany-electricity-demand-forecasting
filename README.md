# German Electricity Demand Forecasting

### Machine Learning | Time-Series Forecasting | Python | scikit-learn

A machine learning project predicting Germany's electricity demand
one hour ahead using official historical grid-load data from SMARD.

## 1. Project Overview

Electricity demand forecasting supports power-system planning and
operational decision-making.

This project develops and evaluates machine learning models for
rolling one-hour-ahead electricity demand forecasting.

## 2. Dataset

**Source:** Bundesnetzagentur | SMARD.de
https://www.smard.de/

- Period: January 2023 – December 2025
- Observations: 26,304 hourly records
- Region: Germany
- Target: Actual grid load (MWh)
- Missing target values: 0 after conversion

The data is reused with attribution under CC BY 4.0.

## 3. Methodology

### Data Preparation
- Loaded and validated original SMARD CSV data
- Corrected numerical formatting
- Converted local timestamps to UTC
- Handled daylight-saving-time transitions
- Verified continuous hourly observations

### Feature Engineering
- Hour, day of week, month and weekend indicator
- Previous-hour electricity demand (lag_1)
- Previous-day demand (lag_24)
- Previous-week demand (lag_168)

All lag features use previously observed demand.

### Chronological Data Split
- Training: 2023
- Validation: 2024
- Testing: 2025

Final model training used 2023–2024.

### Models Evaluated
- Persistence baseline
- Ridge Regression
- Random Forest Regression
- Histogram-based Gradient Boosting

TimeSeriesSplit and RandomizedSearchCV were used for
cross-validation and hyperparameter tuning.

## 4. Model Evaluation

### Validation Results (2024)

| Model | MAE (MWh) | RMSE (MWh) |
|---|---:|---:|
| Persistence | 1,757.88 | 2,319.90 |
| Ridge Regression | 1,398.80 | 1,842.12 |
| Random Forest | 604.18 | 848.43 |
| Gradient Boosting | 514.83 | 697.04 |
| Tuned Gradient Boosting | 495.31 | 671.64 |

### Final Test Results (2025)

| Model | MAE (MWh) | RMSE (MWh) |
|---|---:|---:|
| Persistence | 1,779.00 | 2,342.67 |
| Tuned Gradient Boosting | 500.46 | 666.45 |

**MAE reduction versus persistence: 71.87%.**

The final model was evaluated on 8,760 test observations.

## 5. Key Findings

- Gradient Boosting outperformed the other evaluated approaches
  on the 2024 validation dataset.
- Previous-hour demand was the most important predictor
  according to permutation importance.
- Larger forecasting errors occurred around several holidays.
- Average errors were relatively high during some morning hours.
- The model showed similar MAE on validation and test data.

## 6. Limitations

- Forecasts assume recent actual grid-load observations are available.
- This is rolling one-hour-ahead forecasting, not a full-year
  forecast generated without new observations.
- Holiday and weather information were not included.
- Results cover the historical period studied and may not
  generalize to changing market conditions.
- Permutation importance is not a causal explanation.
- Hyperparameter search was limited to selected configurations.

## 7. Project Structure

```text
germany-electricity-demand-forecasting/
├── data/
│   └── raw/
│       └── electricity_data.csv
├── notebooks/
│   └── 01_data_inspection.ipynb
├── models/
│   ├── electricity_forecasting_model.joblib
│   └── feature_names.joblib
├── requirements.txt
├── .gitignore
└── README.md
```

## 8. Reproducibility

1. Clone this repository.
2. Create a Python environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Launch Jupyter from the repository root.
5. Open notebooks/01_data_inspection.ipynb.
6. Verify that the notebook's working directory is the
   repository root before running its cells.

The saved model was created using scikit-learn 1.8.0.
Only load model files from trusted sources.

## 9. Technologies

Python, Pandas, NumPy, Matplotlib, scikit-learn,
Jupyter Notebook, Joblib.

## 10. Author

Muhammad Mujtaba
B.Sc. Computer Science and Digitalisation
Berlin School of Business and Innovation, Germany

GitHub: https://github.com/Mujitabaa

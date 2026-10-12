# Air Quality Index Prediction

An interactive AI-powered web application that predicts the **Air Quality Index (AQI)** from environmental sensor readings using **Gradient Boosting (XGBoost-style)**.

Built with Python, Streamlit, and scikit-learn.

---

## What it does

This app takes real-world environmental sensor inputs — PM2.5, PM10, NO2, CO, O3, SO2, Temperature, Humidity, and Wind Speed — and predicts the AQI level in real time. It also explains how the model works, which features matter most, and how Gradient Boosting learns.

---

## ML Model

| Property | Value |
|---|---|
| Algorithm | Gradient Boosting Regressor |
| Type | Supervised — Regression |
| Library | scikit-learn |
| Evaluation | R², MAE, RMSE, 5-Fold Cross-Validation |

**How Gradient Boosting works:**  
It builds an ensemble of decision trees sequentially. Each new tree corrects the errors of the previous ones. The final prediction is the weighted sum of all trees — the same principle behind XGBoost.

---

## AQI Categories

| AQI Range | Category | Health Impact |
|---|---|---|
| 0 – 50 | Good | Safe for all |
| 51 – 100 | Moderate | Minor concern for sensitive people |
| 101 – 150 | Unhealthy for Sensitive Groups | Sensitive groups affected |
| 151 – 200 | Unhealthy | Health effects for everyone |
| 201 – 300 | Very Unhealthy | Serious health effects |
| 301 – 500 | Hazardous | Emergency conditions |

---

## Features

- **Predictions vs Reality** — scatter plot of actual vs predicted AQI with error distribution
- **Feature Importance** — impurity-based and permutation importance side by side
- **How Boosting Works** — MSE curve as trees are added, cross-validation results
- **Data Explorer** — scatter plots, hourly AQI trends, category breakdown, correlation heatmap
- **Live AQI Predictor** — enter sensor readings and get an instant prediction with a risk gauge

---

## Installation

```bash
pip install streamlit scikit-learn pandas numpy plotly
```

## Run

```bash
streamlit run FAI.py
```

Then open your browser at `http://localhost:8501`

---

## Project Structure

```
FAI/
├── FAI.py          # Main application
└── README.md       # This file
```

---

## Case Study

**Application:** Environmental monitoring and public health  
**Problem:** Predict AQI from sensor data to issue early warnings before pollution levels become dangerous  
**Dataset:** Synthetic data generated to mirror real-world air quality distributions across all 6 AQI categories

---

## Tech Stack

- Python 3.7+
- Streamlit — interactive web UI
- scikit-learn — Gradient Boosting model
- Plotly — interactive charts
- Pandas / NumPy — data processing

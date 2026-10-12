# Air Quality Index Prediction

An interactive AI-powered web application that predicts the **Air Quality Index (AQI)** from environmental sensor readings using **Gradient Boosting (XGBoost-style)**. Built for real-time environmental monitoring and public health awareness.

---

## Overview

Air pollution is one of the leading environmental health risks globally. This application uses machine learning to predict how polluted the air is at any given moment based on sensor measurements. By predicting AQI in advance, authorities and individuals can take preventive action before pollution reaches dangerous levels.

This project demonstrates a full end-to-end ML pipeline — from raw sensor data to a live interactive prediction dashboard — built entirely in Python.

---

## Demo

| Tab | Description |
|---|---|
| Predictions vs Reality | How well the model predicts on unseen data |
| Feature Importance | Which pollutants drive AQI the most |
| How Boosting Works | Visual explanation of the algorithm |
| Data Explorer | Explore the dataset interactively |
| Live AQI Predictor | Enter sensor values and get instant prediction |

---

## ML Model — Gradient Boosting Regressor

### What is Gradient Boosting?

Gradient Boosting is an ensemble machine learning algorithm that builds a series of decision trees **sequentially**. Unlike Random Forest which builds trees in parallel, Gradient Boosting builds each tree to correct the mistakes of the previous one.

```
Tree 1  →  makes a rough prediction
Tree 2  →  learns to fix errors from Tree 1
Tree 3  →  learns to fix errors from Tree 1 + Tree 2
...
Final prediction = weighted sum of all trees
```

This makes it one of the most powerful and accurate algorithms for structured/tabular data — the same core idea behind **XGBoost**, **LightGBM**, and **CatBoost** used by top data scientists worldwide.

### Why Gradient Boosting for AQI?

- AQI depends on multiple interacting features (PM2.5, wind, humidity, etc.) — tree ensembles capture these non-linear interactions naturally
- It handles feature scale differences without needing normalization
- It gives built-in feature importance scores
- It is robust to outliers, which are common in sensor data

### Model Configuration

| Hyperparameter | Default | Description |
|---|---|---|
| `n_estimators` | 200 | Number of trees in the ensemble |
| `max_depth` | 4 | Maximum depth of each tree |
| `learning_rate` | 0.1 | Step size for each tree's contribution |
| `subsample` | 0.8 | Fraction of data used per tree (prevents overfitting) |
| `min_samples_leaf` | 5 | Minimum samples required at a leaf node |

All hyperparameters are adjustable live from the sidebar — the model retrains instantly on every change.

### Evaluation Metrics

| Metric | What it measures |
|---|---|
| **R² Score** | Proportion of variance explained (1.0 = perfect) |
| **MAE** | Mean Absolute Error in AQI units |
| **RMSE** | Root Mean Squared Error (penalises large errors more) |
| **CV R² (5-fold)** | Cross-validation score — checks for overfitting |

---

## AQI Categories

The Air Quality Index is a standardised scale used by environmental agencies worldwide.

| AQI Range | Category | Health Impact | Recommended Action |
|---|---|---|---|
| 0 – 50 | Good | Air quality is satisfactory | No restrictions |
| 51 – 100 | Moderate | Acceptable for most people | Very sensitive individuals should limit outdoor time |
| 101 – 150 | Unhealthy for Sensitive Groups | Sensitive groups may experience effects | Children, elderly, and asthma patients should reduce exertion |
| 151 – 200 | Unhealthy | Everyone may begin to experience effects | Avoid prolonged outdoor activity |
| 201 – 300 | Very Unhealthy | Serious health effects for everyone | Stay indoors where possible |
| 301 – 500 | Hazardous | Emergency conditions — health warnings issued | Remain indoors, seal windows, wear masks if outside |

---

## Input Features

The model takes 9 environmental sensor readings as input:

| Feature | Unit | Description |
|---|---|---|
| **PM2.5** | ug/m³ | Fine particulate matter (diameter < 2.5 microns) — most harmful to lungs |
| **PM10** | ug/m³ | Coarse particulate matter (diameter < 10 microns) |
| **NO2** | ppb | Nitrogen dioxide — from vehicle and industrial emissions |
| **CO** | ppm | Carbon monoxide — from combustion |
| **O3** | ppb | Ground-level ozone — forms from sunlight reacting with pollutants |
| **SO2** | ppb | Sulphur dioxide — from burning fossil fuels |
| **Temperature** | °C | Ambient air temperature |
| **Humidity** | % | Relative humidity |
| **Wind Speed** | m/s | Wind speed — higher wind disperses pollutants |

---

## Application Tabs Explained

### 1. Predictions vs Reality
Scatter plot comparing the model's predicted AQI against the true AQI on the held-out test set. Each point is coloured by AQI category. A perfect model would have all points on the diagonal line. Also shows the error distribution histogram to confirm the model is unbiased (centred at zero).

### 2. Feature Importance
Two complementary importance measures are shown side by side:
- **Impurity-based importance** — how much each feature reduces prediction error across all trees
- **Permutation importance** — how much accuracy drops when each feature's values are randomly shuffled (more reliable, model-agnostic)

Also includes a **Partial Dependence Plot** for the top feature, showing how that single feature affects the predicted AQI while all others are held constant.

### 3. How Boosting Works
Shows the MSE (Mean Squared Error) curve on both training and test sets as more trees are added. This visualises exactly how the model improves with each boosting round and where it starts to plateau. Also shows 5-fold cross-validation scores to confirm the model generalises well.

### 4. Data Explorer
- Interactive scatter plot with any two features on the axes, coloured by AQI
- Average AQI across 24 hours of the day (shows rush-hour pollution peaks)
- AQI category frequency bar chart
- Full correlation heatmap of all features

### 5. Live AQI Predictor
Enter sensor readings manually using sliders and number inputs. The model predicts AQI instantly and displays:
- A **gauge chart** with colour-coded risk zones
- A **health advice card** with recommended actions
- A **feature contribution bar chart** showing which sensor readings pushed the prediction up or down for that specific input
- A reference table of all AQI categories

---

## Installation

Make sure you have Python 3.7 or higher installed.

```bash
pip install streamlit scikit-learn pandas numpy plotly
```

## Run the App

```bash
streamlit run FAI.py
```

Open your browser at `http://localhost:8501`

---

## Project Structure

```
FAI/
├── FAI.py          # Full application — data generation, model, UI
└── README.md       # Project documentation
```

---

## Dataset

This project uses a **synthetic dataset** generated to mirror the statistical properties of real-world air quality sensor data. The data is balanced across all 6 AQI categories (approximately equal samples per category) to ensure the model learns all pollution levels equally — not just the common "Good" cases.

- Default size: 3,000 readings
- Configurable up to 6,000 rows from the sidebar
- No external download required — everything is generated at runtime

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.7+ | Core language |
| Streamlit | Interactive web dashboard |
| scikit-learn | Gradient Boosting model, metrics, cross-validation |
| Plotly | Interactive charts and gauges |
| Pandas | Data manipulation |
| NumPy | Numerical computations |

---

## Use Cases

- Environmental agencies monitoring city air quality in real time
- Smart city infrastructure with IoT sensor networks
- Public health departments issuing early pollution warnings
- Research into the relationship between weather patterns and pollution levels
- Educational demonstrations of gradient boosting on real-world data

---

## Author

GitHub: [Darknite0707](https://github.com/Darknite0707)

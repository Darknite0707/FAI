"""
Air Quality Index Prediction — Gradient Boosting (XGBoost-style)
=================================================================
Case Study : Predict the Air Quality Index (AQI) from environmental
             sensor readings (PM2.5, PM10, NO2, CO, O3, SO2, Temp,
             Humidity, Wind Speed).

Algorithm  : Gradient Boosting Regressor (sklearn) — same principle
             as XGBoost: an ensemble of decision trees built
             sequentially, each correcting the previous one's errors.

Run:
    pip install streamlit scikit-learn pandas numpy plotly
    streamlit run FAI.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Air Quality AI · Gradient Boosting",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
    border-left: 6px solid #00c9ff;
}
.hero h1 { margin: 0; font-size: 2rem; letter-spacing: -.5px; }
.hero p  { margin: .4rem 0 0; opacity: .85; font-size: 1rem; }

.badge {
    display: inline-block;
    padding: .28rem .9rem;
    border-radius: 20px;
    font-size: .76rem;
    font-weight: 700;
    letter-spacing: .06em;
    margin-bottom: .9rem;
    color: white;
    background: #203a43;
}
[data-testid="metric-container"] {
    background: #1e3a5f;
    border: 1px solid #00c9ff;
    border-radius: 12px;
    padding: .75rem 1rem;
}
[data-testid="metric-container"] * {
    color: #ffffff !important;
}

/* Remove excess vertical whitespace */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}
div[data-testid="stVerticalBlock"] > div:has(hr) {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
hr {
    margin: 0.3rem 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero — rendered after sidebar so source_label is available ───────────────
HERO_PLACEHOLDER = st.empty()


# ────────────────────────────────────────────────────────────────────────────
# Data
# ────────────────────────────────────────────────────────────────────────────
FEATURES = ["PM2.5", "PM10", "NO2", "CO", "O3", "SO2",
            "Temperature", "Humidity", "WindSpeed"]
UNITS    = ["ug/m3", "ug/m3", "ppb", "ppm", "ppb", "ppb", "C", "%", "m/s"]

AQI_CATEGORIES = [
    (0,   50,  "Good",            "#2ecc71"),
    (51,  100, "Moderate",        "#f1c40f"),
    (101, 150, "Unhealthy (Sens.)","#e67e22"),
    (151, 200, "Unhealthy",       "#e74c3c"),
    (201, 300, "Very Unhealthy",  "#8e44ad"),
    (301, 500, "Hazardous",       "#7f0000"),
]

def aqi_label(val):
    for lo, hi, label, _ in AQI_CATEGORIES:
        if lo <= val <= hi:
            return label
    return "Hazardous"

def aqi_color(val):
    for lo, hi, _, color in AQI_CATEGORIES:
        if lo <= val <= hi:
            return color
    return "#7f0000"


@st.cache_data
def make_data(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic hourly air-quality readings spread evenly across all
    6 AQI categories so every category has meaningful sample counts.
    """
    rng = np.random.default_rng(seed)

    # Target AQI midpoints for each category (Good → Hazardous)
    # Each segment gets n // 6 rows; remainder fills the last bucket
    seg_size = n // 6
    segments = [
        # (aqi_center, aqi_std, pm25_base, no2_base, co_base, o3_base, so2_base)
        (25,  8,  12,  20, 0.35, 45, 8),    # Good        0-50
        (75,  10, 35,  45, 0.65, 70, 18),   # Moderate    51-100
        (125, 10, 65,  75, 1.00, 95, 30),   # Unhealthy S 101-150
        (175, 10, 95, 105, 1.50, 115, 45),  # Unhealthy   151-200
        (250, 20, 140, 140, 2.20, 145, 60), # Very Unhlth 201-300
        (380, 30, 210, 175, 3.50, 175, 75), # Hazardous   301-500
    ]

    rows = []
    for i, (aqi_c, aqi_s, pm25_b, no2_b, co_b, o3_b, so2_b) in enumerate(segments):
        m = seg_size if i < 5 else n - seg_size * 5
        hours_seg = rng.integers(0, 24, m)
        day_cycle = np.sin(2 * np.pi * hours_seg / 24)

        pm25 = (pm25_b  + 10 * (-day_cycle) + rng.normal(0, 6, m)).clip(5,  350)
        pm10 = (pm25 * rng.uniform(1.3, 1.8, m) + rng.normal(0, 8, m)).clip(5, 500)
        no2  = (no2_b   + 8  * (-day_cycle) + rng.normal(0, 8, m)).clip(5,  200)
        co   = (co_b    + 0.2 * (-day_cycle) + rng.normal(0, 0.1, m)).clip(0.1, 5)
        o3   = (o3_b    + 15 * day_cycle     + rng.normal(0, 10, m)).clip(5,  200)
        so2  = (so2_b   + rng.normal(0, 4, m)).clip(1, 80)
        temp = (22      + 8  * day_cycle     + rng.normal(0, 3, m)).clip(-5, 45)
        hum  = (60      - 10 * day_cycle     + rng.normal(0, 8, m)).clip(10, 100)
        wind = (3.5     + rng.exponential(1.5, m)).clip(0.1, 20)

        aqi  = (
            0.40 * pm25 +
            0.20 * (no2 / 2) +
            0.15 * (pm10 / 3) +
            0.10 * (co * 20) +
            0.10 * (o3 / 2) +
            0.05 * so2
        ).clip(0, 500)
        aqi += rng.normal(0, aqi_s * 0.3, m)

        seg_df = pd.DataFrame({
            f: v for f, v in zip(
                FEATURES,
                [pm25, pm10, no2, co, o3, so2, temp, hum, wind]
            )
        })
        seg_df["AQI"]  = aqi.clip(0, 500).round(1)
        seg_df["Hour"] = hours_seg
        rows.append(seg_df)

    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed)
    df["Category"] = df["AQI"].apply(aqi_label)
    return df.round(2).reset_index(drop=True)


@st.cache_data
def load_kaggle_csv(file) -> pd.DataFrame:
    """
    Accepts the Kaggle 'Air Quality Data in India' city_day.csv.
    Auto-maps its columns to the FEATURES expected by the model.
    Drops rows with too many nulls, fills remaining with column medians.
    """
    raw = pd.read_csv(file)

    # Column name mapping: Kaggle name → our FEATURES name
    rename_map = {
        "PM2.5":    "PM2.5",
        "PM10":     "PM10",
        "NO":       "NO2",   # fallback if NO2 missing
        "NO2":      "NO2",
        "NOx":      "NO2",   # another fallback
        "CO":       "CO",
        "O3":       "O3",
        "Ozone":    "O3",
        "SO2":      "SO2",
        "Toluene":  "SO2",   # rough proxy if SO2 missing
    }
    raw = raw.rename(columns=rename_map)

    # Keep only the columns we need; add missing ones as NaN
    for feat in FEATURES:
        if feat not in raw.columns:
            raw[feat] = np.nan

    # Temperature, Humidity, WindSpeed are not in this Kaggle dataset
    # Fill with realistic city averages
    if raw["Temperature"].isna().all():
        raw["Temperature"] = 25.0
    if raw["Humidity"].isna().all():
        raw["Humidity"] = 60.0
    if raw["WindSpeed"].isna().all():
        raw["WindSpeed"] = 3.0

    df_out = raw[FEATURES].copy()

    # Drop rows where more than half the pollutant features are NaN
    pollutants = ["PM2.5", "PM10", "NO2", "CO", "O3", "SO2"]
    df_out = df_out[df_out[pollutants].isna().sum(axis=1) <= 3]

    # Fill remaining NaNs with median
    df_out = df_out.fillna(df_out.median(numeric_only=True))
    df_out = df_out.dropna().reset_index(drop=True)

    # Compute AQI
    df_out["AQI"] = (
        0.40 * df_out["PM2.5"] +
        0.20 * (df_out["NO2"] / 2) +
        0.15 * (df_out["PM10"] / 3) +
        0.10 * (df_out["CO"] * 20) +
        0.10 * (df_out["O3"] / 2) +
        0.05 * df_out["SO2"]
    ).clip(0, 500).round(1)

    df_out["Hour"]     = 12  # Kaggle data is daily — use noon as placeholder
    df_out["Category"] = df_out["AQI"].apply(aqi_label)

    return df_out.round(2)


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Configuration")

    st.subheader("Gradient Boosting")
    n_estimators  = st.slider("Number of trees",       50,  500, 200, step=50)
    max_depth     = st.slider("Max tree depth",         2,    8,   4)
    learning_rate = st.select_slider("Learning rate",
                        options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
    subsample     = st.slider("Row subsample",        0.5,  1.0, 0.8, step=0.1)
    min_samples   = st.slider("Min samples per leaf",   1,   20,   5)

    st.subheader("Training")
    test_size   = st.slider("Test split",  0.10, 0.35, 0.20, step=0.05)
    random_seed = st.number_input("Random seed", value=42, step=1)

    st.subheader("Data")
    data_source = st.radio("Data source", ["Synthetic (built-in)", "Kaggle CSV upload"])

    kaggle_df = None
    if data_source == "Kaggle CSV upload":
        uploaded = st.file_uploader(
            "Upload city_day.csv from Kaggle",
            type="csv",
            help="Download from: kaggle.com/datasets/rohanrao/air-quality-data-in-india",
        )
        if uploaded:
            try:
                kaggle_df = load_kaggle_csv(uploaded)
                st.success(f"Loaded {len(kaggle_df):,} rows from Kaggle dataset")
            except Exception as e:
                st.error(f"Could not parse file: {e}")
                kaggle_df = None
        else:
            st.info("Upload city_day.csv to use real data.\nUsing synthetic data until then.")

    if kaggle_df is None:
        n_rows = st.slider("Synthetic dataset size", 600, 6000, 3000, step=300)

    st.divider()
    st.caption("Change any setting — model retrains instantly.")


# ────────────────────────────────────────────────────────────────────────────
# Train
# ────────────────────────────────────────────────────────────────────────────
df = kaggle_df if kaggle_df is not None else make_data(n_rows, int(random_seed))

source_label = "Real Kaggle Data" if kaggle_df is not None else "Synthetic Data"

HERO_PLACEHOLDER.markdown(f"""
<div class="hero">
  <h1>Air Quality Index Prediction</h1>
  <p>
    <strong>Model: Gradient Boosting (XGBoost-style)</strong> &nbsp;·&nbsp;
    Predict pollution levels from environmental sensor readings in real time
    &nbsp;·&nbsp; <span style="background:#00c9ff33;padding:.15rem .5rem;border-radius:6px;font-size:.85rem">{source_label}</span>
  </p>
</div>
""", unsafe_allow_html=True)

X = df[FEATURES].values
y = df["AQI"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_seed)
)

@st.cache_resource(show_spinner="Training Gradient Boosting model …")
def train_model(n_est, depth, lr, sub, min_s, seed):
    model = GradientBoostingRegressor(
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=lr,
        subsample=sub,
        min_samples_leaf=min_s,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model

model = train_model(n_estimators, max_depth, learning_rate,
                    subsample, min_samples, int(random_seed))

y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)

cv     = cross_val_score(
    GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=subsample,
        min_samples_leaf=min_samples, random_state=int(random_seed)
    ),
    X, y, cv=KFold(5, shuffle=True, random_state=int(random_seed)),
    scoring="r2"
)

# ────────────────────────────────────────────────────────────────────────────
# Top metrics
# ────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Readings",    f"{len(df):,}")
c2.metric("R² Score",    f"{r2:.3f}",   help="1.0 = perfect prediction")
c3.metric("MAE",         f"{mae:.2f}",  help="Mean Absolute Error (AQI units)")
c4.metric("RMSE",        f"{rmse:.2f}", help="Root Mean Squared Error")
c5.metric("CV R² (5-fold)", f"{cv.mean():.3f} ± {cv.std():.3f}")

# ────────────────────────────────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────────────────────────────────
tab_pred, tab_feat, tab_boost, tab_data, tab_live = st.tabs([
    "📊 Predictions vs Reality",
    "🔍 Feature Importance",
    "🌳 How Boosting Works",
    "🌆 Data Explorer",
    "📡 Live AQI Predictor",
])


# ── Tab 1 : Predictions vs Reality ───────────────────────────────────────────
with tab_pred:
    st.markdown('<div class="badge">GRADIENT BOOSTING — TEST SET RESULTS</div>',
                unsafe_allow_html=True)

    res_df = pd.DataFrame({
        "Actual AQI":    y_test,
        "Predicted AQI": y_pred.round(1),
        "Error":         (y_pred - y_test).round(1),
        "Category":      [aqi_label(v) for v in y_test],
    })

    col_l, col_r = st.columns(2)

    with col_l:
        fig_scatter = px.scatter(
            res_df, x="Actual AQI", y="Predicted AQI",
            color="Category",
            color_discrete_map={cat: col for _, _, cat, col in AQI_CATEGORIES},
            opacity=0.7,
            title="<b>Actual vs Predicted AQI</b>  —  perfect model = diagonal line",
            template="plotly_white",
        )
        # Perfect prediction line
        mn, mx = float(y_test.min()), float(y_test.max())
        fig_scatter.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color="black", dash="dash", width=1.5),
            name="Perfect prediction",
        ))
        fig_scatter.update_layout(margin=dict(t=50, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_r:
        fig_err = px.histogram(
            res_df, x="Error", nbins=40,
            color_discrete_sequence=["#2c5364"],
            title="<b>Prediction Error Distribution</b>  —  centred at 0 = unbiased",
            template="plotly_white",
            labels={"Error": "Predicted − Actual (AQI)"},
        )
        fig_err.add_vline(x=0, line_dash="dash", line_color="#e74c3c",
                          annotation_text="  Zero error")
        fig_err.update_layout(margin=dict(t=50, b=20))
        st.plotly_chart(fig_err, use_container_width=True)

    # Metrics by category
    st.subheader("Performance per AQI Category")
    cat_rows = []
    for _, _, cat, color in AQI_CATEGORIES:
        subset = res_df[res_df["Category"] == cat]
        if len(subset) == 0:
            continue
        cat_rows.append({
            "Category": cat,
            "Samples":  len(subset),
            "MAE":      round(subset["Error"].abs().mean(), 2),
            "Max Error":round(subset["Error"].abs().max(), 2),
        })
    cat_df = pd.DataFrame(cat_rows)
    st.dataframe(
        cat_df.style.background_gradient(cmap="Oranges", subset=["MAE","Max Error"]),
        use_container_width=True, hide_index=True,
    )


# ── Tab 2 : Feature Importance ────────────────────────────────────────────────
with tab_feat:
    st.markdown('<div class="badge">WHAT DRIVES AIR QUALITY?</div>',
                unsafe_allow_html=True)

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        # Built-in impurity importance
        fi = pd.DataFrame({
            "Feature":   FEATURES,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)

        fig_fi = px.bar(
            fi, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Blues",
            title="<b>Feature Importance</b> (impurity-based)",
            template="plotly_white", text_auto=".3f",
        )
        fig_fi.update_layout(
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_f2:
        # Permutation importance (model-agnostic)
        perm = permutation_importance(
            model, X_test, y_test, n_repeats=10,
            random_state=int(random_seed), scoring="r2"
        )
        perm_df = pd.DataFrame({
            "Feature":     FEATURES,
            "Importance":  perm.importances_mean,
            "Std":         perm.importances_std,
        }).sort_values("Importance", ascending=False)

        fig_perm = px.bar(
            perm_df, x="Importance", y="Feature",
            error_x="Std", orientation="h",
            color="Importance", color_continuous_scale="Reds",
            title="<b>Permutation Importance</b> (shuffle-based, more reliable)",
            template="plotly_white", text_auto=".3f",
        )
        fig_perm.update_layout(
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig_perm, use_container_width=True)

    # Partial dependence — top feature
    top_feat = fi.iloc[0]["Feature"]
    top_idx  = FEATURES.index(top_feat)
    st.subheader(f"Partial Dependence — {top_feat} (most important feature)")

    grid_vals = np.linspace(X[:, top_idx].min(), X[:, top_idx].max(), 60)
    X_mean    = X_train.mean(axis=0)
    pd_preds  = []
    for v in grid_vals:
        row = X_mean.copy(); row[top_idx] = v
        pd_preds.append(model.predict(row.reshape(1, -1))[0])

    fig_pd = px.line(
        x=grid_vals, y=pd_preds,
        labels={"x": top_feat, "y": "Predicted AQI"},
        title=f"<b>How {top_feat} alone affects predicted AQI</b>  (all other features held at mean)",
        template="plotly_white",
        color_discrete_sequence=["#00c9ff"],
    )
    fig_pd.update_traces(line=dict(width=3))
    fig_pd.update_layout(margin=dict(t=50, b=20))
    st.plotly_chart(fig_pd, use_container_width=True)


# ── Tab 3 : How Boosting Works ────────────────────────────────────────────────
with tab_boost:
    st.markdown('<div class="badge">HOW GRADIENT BOOSTING LEARNS</div>',
                unsafe_allow_html=True)

    st.markdown("""
**Gradient Boosting builds trees one at a time.**  
Each new tree focuses on the *residual errors* left by all previous trees — like a team where each
member fixes what the previous one got wrong.

> Tree 1 → makes a rough prediction  
> Tree 2 → predicts *error of Tree 1*  
> Tree 3 → predicts *error of Tree 1 + Tree 2*  
> … repeated for all trees  
> Final answer = sum of all trees × learning rate
""")

    # Training deviance (loss per stage)
    train_scores = np.zeros(n_estimators)
    for i, y_stage in enumerate(model.staged_predict(X_train)):
        train_scores[i] = mean_squared_error(y_train, y_stage)

    test_scores = np.zeros(n_estimators)
    for i, y_stage in enumerate(model.staged_predict(X_test)):
        test_scores[i] = mean_squared_error(y_test, y_stage)

    stages_df = pd.DataFrame({
        "Tree": np.arange(1, n_estimators + 1),
        "Train MSE": train_scores,
        "Test MSE":  test_scores,
    })

    fig_stages = go.Figure()
    fig_stages.add_trace(go.Scatter(
        x=stages_df["Tree"], y=stages_df["Train MSE"],
        mode="lines", name="Train MSE",
        line=dict(color="#00c9ff", width=2),
    ))
    fig_stages.add_trace(go.Scatter(
        x=stages_df["Tree"], y=stages_df["Test MSE"],
        mode="lines", name="Test MSE",
        line=dict(color="#e74c3c", width=2),
    ))
    fig_stages.update_layout(
        title="<b>MSE drops as more trees are added</b> — watch the model improve",
        xaxis_title="Number of trees",
        yaxis_title="Mean Squared Error",
        template="plotly_white",
        legend=dict(x=0.75, y=0.95),
        margin=dict(t=50, b=20),
    )
    st.plotly_chart(fig_stages, use_container_width=True)

    best_tree = int(np.argmin(test_scores)) + 1
    st.info(f"📌 Test MSE is lowest at **{best_tree} trees**. "
            f"Trees after this point start overfitting the training data.")

    # Cross-validation R² across folds
    st.subheader("5-Fold Cross-Validation R²")
    cv_df = pd.DataFrame({
        "Fold":    [f"Fold {i+1}" for i in range(5)],
        "R² Score": cv,
    })
    fig_cv = px.bar(
        cv_df, x="Fold", y="R² Score",
        color="R² Score", color_continuous_scale="Blues",
        text_auto=".3f", range_y=[0, 1],
        title="<b>Cross-Validation</b> — consistent scores = no overfitting",
        template="plotly_white",
    )
    fig_cv.add_hline(y=cv.mean(), line_dash="dash", line_color="#e74c3c",
                     annotation_text=f"  Mean R² = {cv.mean():.3f}")
    fig_cv.update_layout(coloraxis_showscale=False, margin=dict(t=50, b=20))
    st.plotly_chart(fig_cv, use_container_width=True)


# ── Tab 4 : Data Explorer ─────────────────────────────────────────────────────
with tab_data:
    st.markdown('<div class="badge">DATASET EXPLORER</div>', unsafe_allow_html=True)

    col_x, col_y = st.columns(2)
    x_feat = col_x.selectbox("X axis", FEATURES, index=0)
    y_feat = col_y.selectbox("Y axis", FEATURES, index=1)

    fig_exp = px.scatter(
        df, x=x_feat, y=y_feat, color="AQI",
        color_continuous_scale="RdYlGn_r",
        hover_data=["AQI", "Category"],
        opacity=0.65, size_max=8,
        title=f"<b>{x_feat}</b> vs <b>{y_feat}</b>  — colour = AQI level",
        template="plotly_white",
    )
    fig_exp.update_layout(margin=dict(t=50, b=20))
    st.plotly_chart(fig_exp, use_container_width=True)

    # AQI over time (by hour)
    st.subheader("Average AQI by Hour of Day")
    hourly = df.groupby("Hour")["AQI"].mean().reset_index()
    fig_hour = px.line(
        hourly, x="Hour", y="AQI", markers=True,
        title="<b>AQI pattern across 24 hours</b>  — rush-hour peaks visible",
        template="plotly_white",
        color_discrete_sequence=["#e74c3c"],
    )
    fig_hour.update_traces(line=dict(width=2.5), marker=dict(size=7))
    fig_hour.update_layout(margin=dict(t=50, b=20))
    st.plotly_chart(fig_hour, use_container_width=True)

    # Category breakdown
    st.subheader("AQI Category Breakdown")
    cat_counts = df["Category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    cat_color_map = {cat: col for _, _, cat, col in AQI_CATEGORIES}
    fig_cat = px.bar(
        cat_counts, x="Category", y="Count",
        color="Category", color_discrete_map=cat_color_map,
        title="<b>How often each AQI category occurs</b>",
        template="plotly_white", text_auto=True,
    )
    fig_cat.update_layout(showlegend=False, margin=dict(t=50, b=20))
    st.plotly_chart(fig_cat, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df[FEATURES + ["AQI"]].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="<b>Feature Correlation Matrix</b>",
        template="plotly_white", aspect="auto",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("Raw data preview (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download dataset as CSV",
        df.to_csv(index=False).encode(),
        "air_quality_data.csv", "text/csv",
    )


# ── Tab 5 : Live AQI Predictor ────────────────────────────────────────────────
with tab_live:
    st.markdown('<div class="badge">LIVE AQI PREDICTOR</div>', unsafe_allow_html=True)
    st.write("Enter current sensor readings to get an instant AQI prediction.")

    col1, col2, col3 = st.columns(3)
    sensor_vals = {}

    DEFAULTS = {
        "PM2.5": 35.0, "PM10": 60.0, "NO2": 40.0,
        "CO": 0.6,     "O3": 60.0,   "SO2": 15.0,
        "Temperature": 22.0, "Humidity": 60.0, "WindSpeed": 3.5,
    }
    RANGES = {
        "PM2.5": (0.0, 250.0), "PM10":  (0.0, 500.0),
        "NO2":   (0.0, 200.0), "CO":    (0.0,   5.0),
        "O3":    (0.0, 200.0), "SO2":   (0.0,  80.0),
        "Temperature": (-5.0, 45.0), "Humidity": (10.0, 100.0),
        "WindSpeed": (0.0, 20.0),
    }

    cols_cycle = [col1, col2, col3]
    for i, (feat, unit) in enumerate(zip(FEATURES, UNITS)):
        lo, hi = RANGES[feat]
        sensor_vals[feat] = cols_cycle[i % 3].number_input(
            f"{feat}  ({unit})",
            min_value=float(lo), max_value=float(hi),
            value=float(DEFAULTS[feat]),
            step=0.1 if hi <= 10 else 1.0,
        )

    input_arr = np.array([[sensor_vals[f] for f in FEATURES]])
    predicted_aqi = float(model.predict(input_arr)[0])
    predicted_aqi = np.clip(predicted_aqi, 0, 500)
    cat   = aqi_label(predicted_aqi)
    color = aqi_color(predicted_aqi)

    st.divider()

    # Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_aqi,
        number=dict(font=dict(size=40, color=color), suffix=" AQI"),
        title=dict(text=f"Predicted AQI  —  <b>{cat}</b>", font=dict(size=18)),
        gauge=dict(
            axis=dict(range=[0, 300], tickwidth=2),
            bar=dict(color=color, thickness=0.28),
            bgcolor="white",
            borderwidth=2, bordercolor="#ccc",
            steps=[
                dict(range=[0,   50],  color="#d5f5e3"),
                dict(range=[50,  100], color="#fef9e7"),
                dict(range=[100, 150], color="#fdebd0"),
                dict(range=[150, 200], color="#fadbd8"),
                dict(range=[200, 300], color="#e8daef"),
            ],
            threshold=dict(line=dict(color="#333", width=4),
                           thickness=0.75, value=100),
        ),
    ))
    fig_gauge.update_layout(
        height=320, template="plotly_white",
        margin=dict(t=60, b=10, l=30, r=30),
    )

    g_col, info_col = st.columns([2, 1])
    with g_col:
        st.plotly_chart(fig_gauge, use_container_width=True)
    with info_col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        health_tips = {
            "Good":              "✅ Air quality is satisfactory. Enjoy outdoor activities.",
            "Moderate":          "🟡 Acceptable. Very sensitive people should limit prolonged outdoor exertion.",
            "Unhealthy (Sens.)": "🟠 Sensitive groups should reduce outdoor activity.",
            "Unhealthy":         "🔴 Everyone may begin to experience health effects.",
            "Very Unhealthy":    "🟣 Health alert — everyone should avoid outdoor activity.",
            "Hazardous":         "🚨 Emergency conditions. Stay indoors, seal windows.",
        }
        st.markdown(
            f"<div style='background:{color}22;border-left:5px solid {color};"
            f"padding:1rem;border-radius:8px'>"
            f"<b style='color:{color}'>{cat}</b><br>{health_tips.get(cat,'')}</div>",
            unsafe_allow_html=True,
        )

    # AQI category reference table
    st.subheader("AQI Reference Scale")
    ref_df = pd.DataFrame([
        {"Range": f"{lo}–{hi}", "Category": cat, "Meaning": tip}
        for (lo, hi, cat, _), tip in zip(
            AQI_CATEGORIES,
            ["Safe for all", "Minor concern for very sensitive people",
             "Sensitive groups affected", "Health effects for everyone",
             "Serious health effects", "Emergency — health warnings"],
        )
    ])
    st.dataframe(ref_df, use_container_width=True, hide_index=True)

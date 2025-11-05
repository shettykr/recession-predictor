import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier

# ----------------------------
# Load trained model
# ----------------------------
with open("recession_xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📉 U.S. Recession Early-Warning Predictor (Full Model)")
st.caption("Built by Manohar — powered by 50 years of FRED macro data 🎁")

st.write("""
This predictor estimates the probability of a **U.S. recession within the next 3 months**  
using GDP, unemployment, inflation, industrial production, yield spread, and 6-month lag trends.
""")

# ----------------------------
# Collect user inputs
# ----------------------------
gdp = st.number_input("Real GDP (Billions, chained 2012 USD):", value=23700.0, step=50.0)
unemp = st.number_input("Unemployment Rate (%):", value=4.2, step=0.1)
inflation = st.number_input("CPI (1982-84=100):", value=320.0, step=1.0)
spread = st.number_input("10-Year minus 2-Year Treasury Yield Spread (%):", value=-0.3, step=0.1)
indprod = st.number_input("Industrial Production Index:", value=103.5, step=0.2)

if st.button("🔮 Predict Recession Probability"):
    # ----------------------------
    # Create synthetic 6-month history
    # (simulate previous months slightly varying around entered values)
    # ----------------------------
    np.random.seed(42)
    history = pd.DataFrame({
        'gdp': [gdp * (1 - 0.001 * i) for i in range(6, -1, -1)],
        'unemployment': [unemp * (1 + 0.01 * i) for i in range(6, -1, -1)],
        'inflation': [inflation * (1 - 0.002 * i) for i in range(6, -1, -1)],
        'yield_spread': [spread + 0.05 * np.random.randn() for _ in range(7)],
        'industrial_production': [indprod * (1 - 0.0015 * i) for i in range(6, -1, -1)]
    })

    # ----------------------------
    # Compute derived features
    # ----------------------------
    data = history.copy()
    data['gdp_change'] = data['gdp'].pct_change()
    data['unemployment_change'] = data['unemployment'].diff()
    data['inflation_change'] = data['inflation'].diff()
    data['indprod_change'] = data['industrial_production'].pct_change()
    data['yield_inverted'] = (data['yield_spread'] < 0).astype(int)
    data = data.fillna(method='bfill')

    # ----------------------------
    # Create lag features (1–6 months)
    # ----------------------------
    for col in ['gdp_change', 'unemployment_change', 'inflation_change', 'indprod_change', 'yield_spread', 'yield_inverted']:
        for lag in range(1, 7):
            data[f"{col}_lag{lag}"] = data[col].shift(lag)

    data = data.dropna().tail(1)

    # ----------------------------
    # Predict using model
    # ----------------------------
    proba = model.predict_proba(data)[:, 1][0]
    st.metric("Recession Probability (next 3 months)", f"{proba*100:.1f}%")

    if proba >= 0.5:
        st.error("⚠️ High Recession Risk — Conditions critical!")
    elif proba >= 0.3:
        st.warning("🟠 Moderate Risk — Economic slowdown likely.")
    else:
        st.success("🟢 Economy stable — No major signs of contraction.")

    with st.expander("See Computed Input Features"):
        st.dataframe(data.T.style.format("{:.4f}"))

st.caption("Model: XGBoost | Features: 6-month lagged macro indicators | Author: Manohar Shetty")

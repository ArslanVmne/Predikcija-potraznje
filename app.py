import streamlit as st

st.set_page_config(
    page_title="ForecastIQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.markdown("## 📈 ForecastIQ")
st.sidebar.caption("Store Sales Forecasting & Inventory Optimization")

st.title("ForecastIQ")
st.subheader("Demand Forecasting & Inventory Optimization System")
st.caption("Built on the Favorita Grocery Sales dataset — 54 stores, 33 product families, 4 years of daily sales data")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### What this app does")
    st.markdown("""
- **Forecasts daily demand** per product family and store using a hybrid ensemble model (LightGBM + Prophet + LSTM)
- **Detects demand anomalies** — unexpected spikes or drops using IQR + Isolation Forest
- **Explains predictions** via SHAP feature contributions
- **Optimizes inventory** using EOQ and safety stock formulas
- **Simulates scenarios** — what happens if a promotion runs or oil prices change
- **Generates purchase orders** based on forecast and configurable lead time / service level
    """)

with col2:
    st.markdown("### Models")
    st.markdown("""
| Model | Role |
|---|---|
| **LightGBM + Prophet features** | Primary forecasting model |
| **LSTM** | Captures long-range temporal patterns |
| **Ensemble** | Weighted combination of LightGBM + LSTM |
| **SHAP** | Feature importance & explainability |
| **Isolation Forest** | Anomaly detection |
| **EOQ + Safety Stock** | Inventory optimization |
    """)

st.divider()

st.markdown("### Navigate")
c1, c2, c3, c4 = st.columns(4)
c1.page_link("pages/1_📊_Forecast.py", label="Forecast", icon="📊")
c2.page_link("pages/2_🎛️_What_If.py", label="What-If Simulator", icon="🎛️")
c3.page_link("pages/3_📋_Orders.py", label="Purchase Orders", icon="📋")
c4.page_link("pages/4_📤_Upload.py", label="Upload Data", icon="📤")

_, c5, _ = st.columns([2, 1, 2])
c5.page_link("pages/5_🔲_ABC_XYZ.py", label="ABC-XYZ Matrix", icon="🔲")

st.divider()

st.markdown("### Dataset")
st.markdown("""
[Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) — Kaggle competition dataset.
3M+ daily sales records across 54 Ecuadorian grocery stores and 33 product categories (2013–2017).
External factors included: oil prices, holidays, promotions, transactions.
""")

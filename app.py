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

nav_items = [
    ("pages/1_📊_Forecast.py",  "📊", "Forecast",       "Daily demand · CI · anomaly detection"),
    ("pages/2_🎛️_What_If.py", "🎛️", "What-If",         "Simulate promotions, holidays & oil"),
    ("pages/3_📋_Orders.py",    "📋", "Orders",          "ML-based purchase order generator"),
    ("pages/5_🔲_ABC_XYZ.py",  "🔲", "ABC-XYZ Matrix",  "Inventory risk segmentation"),
    ("pages/4_📤_Upload.py",    "📤", "Upload Data",     "Upload your own sales CSV"),
]

cols = st.columns(5)
for col, (path, icon, label, desc) in zip(cols, nav_items):
    with col:
        st.page_link(path, label=label, icon=icon)
        st.caption(desc)

st.divider()

st.markdown("### Dataset")
st.markdown("""
[Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) — Kaggle competition dataset.
3M+ daily sales records across 54 Ecuadorian grocery stores and 33 product categories (2013–2017).
External factors included: oil prices, holidays, promotions, transactions.
""")

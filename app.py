import streamlit as st

from src.data_loader import get_stores, load_current_stock, load_inventory_params, load_val_preds
from src.ui import render_sidebar

st.set_page_config(
    page_title="ForecastIQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()


@st.cache_data
def get_live_kpis():
    inv = load_inventory_params()
    stock = load_current_stock()
    val = load_val_preds()
    stores = get_stores()

    merged = inv.merge(stock, on=["store_nbr", "family"], how="left")
    merged["current_stock"] = merged["current_stock"].fillna(0)
    critical_skus = int(((merged["current_stock"] < merged["safety_stock"]) & (merged["ABC"] == "A")).sum())

    forecast_total = int(val["ensemble_pred"].sum())

    return {
        "critical_skus": critical_skus,
        "forecast_total": forecast_total,
        "stores": len(stores),
        "families": int(inv["family"].nunique()),
    }


with st.spinner("Loading live data..."):
    kpis = get_live_kpis()

st.title("ForecastIQ")
st.subheader("Demand Forecasting & Inventory Optimization System")
st.caption("Built on the Favorita Grocery Sales dataset — 54 stores, 33 product families, 4 years of daily sales data")

st.divider()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Stores monitored", kpis["stores"])
k2.metric("Product families", kpis["families"])
k3.metric("15-day forecast total", f"{kpis['forecast_total']:,} units")
k4.metric("A-class SKUs below safety stock", kpis["critical_skus"],
          delta="Immediate action needed" if kpis["critical_skus"] > 0 else "All stocked",
          delta_color="inverse" if kpis["critical_skus"] > 0 else "off")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### What this app does")
    st.markdown("""
- **Forecasts daily demand** per product family and store using a multi-model AI system
- **Detects demand anomalies** — automatically flags unexpected spikes or drops in sales
- **Explains every prediction** in plain English — shows which factors are driving demand up or down
- **Optimizes inventory** using industry-standard EOQ and safety stock formulas
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
    ("pages/1_📊_Forecast.py",  "📊", "Forecast",       "Daily demand · confidence intervals · anomaly alerts"),
    ("pages/2_🎛️_What_If.py", "🎛️", "What-If",         "Simulate promotions, holidays & oil prices"),
    ("pages/3_📋_Orders.py",    "📋", "Orders",          "AI-powered purchase order generator"),
    ("pages/4_🔲_ABC_XYZ.py",  "🔲", "ABC-XYZ Matrix",  "Inventory risk & priority segmentation"),
    ("pages/5_📤_Upload.py",    "📤", "Upload Data",     "Upload your own sales CSV"),
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
External factors included: oil prices, public holidays, and promotional campaigns.
""")

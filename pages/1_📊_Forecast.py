import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.anomaly import detect_anomalies
from src.data_loader import get_families, get_stores, load_inventory_params, load_shap_by_family
from src.model_inference import get_forecast, get_history, get_mape

st.set_page_config(page_title="Forecast — ForecastIQ", page_icon="📊", layout="wide")
st.sidebar.markdown("## 📈 ForecastIQ")

SHAP_LABELS = {
    "roll_mean_7": "7-day rolling avg",
    "onpromotion": "Promotion",
    "lag_7": "Sales 7 days ago",
    "roll_mean_14": "14-day rolling avg",
    "transactions": "Transactions",
    "lag_56": "Sales 56 days ago",
    "prophet_trend": "Prophet trend",
    "lag_14": "Sales 14 days ago",
    "dayofmonth": "Day of month",
    "roll_mean_28": "28-day rolling avg",
}


@st.cache_data
def cached_families():
    return get_families()


@st.cache_data
def cached_stores():
    return get_stores()


def make_forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame, anomalies: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history["date"], y=history["sales"],
        name="History", mode="lines",
        line=dict(color="#94a3b8", width=2),
    ))

    # Anomaly markers
    anom = anomalies[anomalies["is_anomaly"]]
    if not anom.empty:
        fig.add_trace(go.Scatter(
            x=anom["date"], y=anom["sales"],
            name="Anomaly", mode="markers",
            marker=dict(color="#ef4444", size=10, symbol="circle",
                        line=dict(color="#fff", width=1.5)),
            hovertemplate="<b>%{x}</b><br>Sales: %{y}<extra>Anomaly</extra>",
        ))

    if not forecast.empty:
        dates = forecast["date"].tolist()
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=forecast["ci_upper"].tolist() + forecast["ci_lower"].tolist()[::-1],
            fill="toself", fillcolor="rgba(37,99,235,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=forecast["yhat"],
            name="Forecast", mode="lines",
            line=dict(color="#2563eb", width=2, dash="dash"),
        ))

    fig.update_layout(
        height=320,
        margin=dict(t=20, r=20, b=50, l=60),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", size=13),
        xaxis=dict(showgrid=False, tickfont=dict(size=12)),
        yaxis=dict(gridcolor="#334155", title="Units sold", tickfont=dict(size=12)),
        legend=dict(orientation="h", y=-0.3, font=dict(size=12)),
    )
    return fig


def make_shap_chart(shap_data: list) -> go.Figure:
    top = sorted(shap_data, key=lambda x: abs(x["value"]), reverse=True)[:8]
    top = sorted(top, key=lambda x: x["value"])
    colors = ["#16a34a" if d["value"] >= 0 else "#ef4444" for d in top]

    fig = go.Figure(go.Bar(
        x=[d["value"] for d in top],
        y=[d["label"] for d in top],
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        height=300,
        margin=dict(t=10, r=20, b=40, l=160),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", size=13),
        xaxis=dict(title="SHAP contribution", gridcolor="#334155", tickfont=dict(size=12)),
        yaxis=dict(showgrid=False, tickfont=dict(size=13)),
    )
    return fig


# ── Sidebar controls ──────────────────────────────────────────────────────────
families = cached_families()
stores = cached_stores()

with st.sidebar:
    st.divider()
    family = st.selectbox("Product family", families, index=families.index("PRODUCE") if "PRODUCE" in families else 0)
    store = st.selectbox("Store", stores, index=0)

# ── Load data ─────────────────────────────────────────────────────────────────
history = get_history(store, family, days=90)
forecast = get_forecast(store, family)
anomalies = detect_anomalies(history)
mape = get_mape(store, family)

forecast_total = int(forecast["yhat"].sum()) if not forecast.empty else 0
last_sales = history["sales"].tail(len(forecast)).sum() if len(history) >= len(forecast) else history["sales"].sum()
trend_pct = round((forecast_total / max(last_sales, 1) - 1) * 100, 1) if last_sales > 0 else 0.0

# SHAP
shap_df = load_shap_by_family()
family_row = shap_df[shap_df["family"] == family]
if family_row.empty:
    family_row = shap_df.head(1)
shap_features = [c for c in shap_df.columns if c != "family"]
shap_values = family_row[shap_features].iloc[0] if not family_row.empty else {}
shap_data = [
    {"feature": f, "value": float(shap_values[f]), "label": SHAP_LABELS.get(f, f)}
    for f in shap_features
]

# Store breakdown
params = load_inventory_params()
store_params = params[params["family"] == family].sort_values("store_nbr")
breakdown = []
for _, row in store_params.iterrows():
    fc_val = int(row["mean_daily"] * 15)
    stock = int(row["safety_stock"])
    order = max(fc_val - stock, 0)
    status = "🔴 Urgent" if order > fc_val * 0.5 else ("🟡 Planned" if order > 0 else "🟢 OK")
    breakdown.append({"Store": int(row["store_nbr"]), "Forecast": fc_val, "Stock": stock, "Order qty": order, "Status": status})

# ── Page ──────────────────────────────────────────────────────────────────────
st.title(f"Sales Forecast — {family}")
st.caption(f"Store {store}  ·  Validation period: Aug 1–15, 2017  ·  Ensemble model")

# KPIs
n_anomalies = int(anomalies["is_anomaly"].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("15-day Forecast", f"{forecast_total:,} units")
k2.metric("MAPE", f"{mape}%", delta=f"{'Excellent' if mape < 10 else 'Acceptable'}", delta_color="off")
k3.metric("Trend vs prior period", f"{'+' if trend_pct >= 0 else ''}{trend_pct}%",
          delta_color="normal" if trend_pct >= 0 else "inverse")
k4.metric("Anomalies (90d)", n_anomalies, delta="Detected" if n_anomalies > 0 else "None",
          delta_color="inverse" if n_anomalies > 0 else "off")

st.divider()

# Forecast chart
st.subheader("Forecast with Confidence Interval — 95% CI")
st.plotly_chart(make_forecast_chart(history, forecast, anomalies), use_container_width=True)

# Anomaly table
if n_anomalies > 0:
    with st.expander(f"Detected anomalies — {n_anomalies} event(s) in last 90 days"):
        anom_df = anomalies[anomalies["is_anomaly"]][["date", "sales", "direction", "zscore"]].copy()
        anom_df.columns = ["Date", "Sales", "Type", "Z-score"]
        anom_df["Type"] = anom_df["Type"].map({"spike": "Demand spike", "drop": "Demand drop"})
        anom_df = anom_df.sort_values("Date", ascending=False)
        st.dataframe(anom_df, use_container_width=True, hide_index=True)

st.divider()

# Breakdown + SHAP
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Store Breakdown")
    st.dataframe(pd.DataFrame(breakdown), use_container_width=True, hide_index=True)

with col_right:
    st.subheader("SHAP — Feature Contributions")
    st.plotly_chart(make_shap_chart(shap_data), use_container_width=True)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.anomaly import detect_anomalies
from src.data_loader import get_families, get_store_labels, get_stores, load_inventory_params, load_shap_by_family, load_val_preds
from src.inventory import get_critical_a_count
from src.model_inference import get_forecast, get_history, get_mape
from src.ui import HOVERLABEL, render_sidebar

st.set_page_config(page_title="Forecast | ForecastIQ", page_icon="📊", layout="wide")
render_sidebar()

SHAP_LABELS = {
    "roll_mean_7":  "Sales trend, last 7 days",
    "onpromotion":  "Active promotion",
    "lag_7":        "Sales last week",
    "roll_mean_14": "Sales trend, last 2 weeks",
    "transactions": "Customer footfall",
    "lag_56":       "Sales 2 months ago",
    "prophet_trend": "Long-term category trend",
    "lag_14":       "Sales 2 weeks ago",
    "dayofmonth":   "Day of the month",
    "roll_mean_28": "Sales trend, last month",
}


@st.cache_data
def cached_forecast(store: int, family: str) -> pd.DataFrame:
    return get_forecast(store, family)


@st.cache_data
def cached_history(store: int, family: str) -> pd.DataFrame:
    return get_history(store, family, days=90)


@st.cache_data
def cached_anomalies(store: int, family: str) -> pd.DataFrame:
    return detect_anomalies(cached_history(store, family))


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
            name="Expected range", showlegend=True,
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
        hoverlabel=HOVERLABEL,
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
        xaxis=dict(title="Impact on forecast", gridcolor="#334155", tickfont=dict(size=12)),
        yaxis=dict(showgrid=False, tickfont=dict(size=13)),
        hoverlabel=HOVERLABEL,
    )
    return fig


# ── Sidebar controls ──────────────────────────────────────────────────────────
families = get_families()
stores = get_stores()
store_labels = get_store_labels()

with st.sidebar:
    st.divider()
    family = st.selectbox("Product family", families,
                          index=families.index("PRODUCE") if "PRODUCE" in families else 0,
                          help="Select a product category to view its demand forecast, anomalies, and store breakdown.")

    uploaded_stores = st.session_state.get("uploaded_stores", [])
    if uploaded_stores:
        store_options = [s for s in stores if s in uploaded_stores]
        st.caption(f"Showing uploaded stores: {uploaded_stores}")
    else:
        store_options = stores
    store = st.selectbox("Store", store_options, index=0,
                         format_func=lambda s: store_labels.get(s, f"Store {s}"),
                         help="Select a store location to view its specific forecast and historical sales patterns.")

# ── Load data ─────────────────────────────────────────────────────────────────
history = cached_history(store, family)
forecast = cached_forecast(store, family)

if history.empty and forecast.empty:
    st.info(
        f"No data available for **{family}** in **Store {store}**. "
        "Try selecting a different store or product family."
    )
    st.stop()

anomalies = cached_anomalies(store, family)
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

# Store breakdown — use actual model forecast per store, fall back to mean_daily if unavailable
@st.cache_data
def cached_store_breakdown(family: str) -> list[dict]:
    params = load_inventory_params()
    store_params = params[params["family"] == family].sort_values("store_nbr")
    rows = []
    for _, row in store_params.iterrows():
        s = int(row["store_nbr"])
        fc = get_forecast(s, family)
        fc_val = int(fc["yhat"].sum()) if not fc.empty else int(row["mean_daily"] * 15)
        stock = int(row["safety_stock"])
        order = max(fc_val - stock, 0)
        status = "🔴 Urgent" if order > fc_val * 0.5 else ("🟡 Planned" if order > 0 else "🟢 OK")
        rows.append({"Store": s, "Forecast": fc_val, "Stock": stock, "Order qty": order, "Status": status})
    return rows

with st.spinner("Computing store breakdown..."):
    breakdown = cached_store_breakdown(family)

# ── Cross-page alert: critical stock for this store ───────────────────────────
@st.cache_data
def cached_critical_count(store_nbr: int) -> int:
    return get_critical_a_count(store_nbr)

critical_count = cached_critical_count(store)
if critical_count > 0:
    col_alert, col_link = st.columns([5, 1])
    with col_alert:
        st.warning(f"⚠️ {critical_count} A-class product{'s' if critical_count > 1 else ''} in Store {store} {'are' if critical_count > 1 else 'is'} below safety stock.")
    with col_link:
        st.page_link("pages/3_📋_Orders.py", label="Go to Orders →", icon="📋")

# ── Page ──────────────────────────────────────────────────────────────────────
st.title(f"Sales Forecast: {family}")
st.caption(f"Store {store}  ·  Forecast period: Aug 1-15, 2017  ·  AI ensemble model")

# KPIs
n_anomalies = int(anomalies["is_anomaly"].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("15-day Forecast", f"{forecast_total:,} units",
          help="Total units forecasted for Aug 1-15, 2017 for the selected store and product family, using the LightGBM + LSTM ensemble model.")
k2.metric("Forecast accuracy", f"{mape}% avg error",
          delta=f"{'Excellent' if mape < 10 else 'Acceptable'}",
          delta_color="off",
          help="Average % error vs. actual sales. Under 15% is excellent for retail demand forecasting.")
k3.metric("Trend vs prior period", f"{'+' if trend_pct >= 0 else ''}{trend_pct}%",
          delta_color="normal" if trend_pct >= 0 else "inverse",
          help="Percentage change comparing the 15-day forecast against actual sales from the equivalent prior period. Positive means demand is growing.")
k4.metric("Anomalies (90d)", n_anomalies, delta="Detected" if n_anomalies > 0 else "None",
          delta_color="inverse" if n_anomalies > 0 else "off",
          help="Unusual demand spikes or drops detected in the last 90 days using IQR + Isolation Forest. A value is flagged when it falls outside 2.5 times the interquartile range.")

st.divider()

# Forecast chart
st.subheader("Forecast with expected range")
st.plotly_chart(make_forecast_chart(history, forecast, anomalies), use_container_width=True)

# Anomaly table
if n_anomalies > 0:
    with st.expander(f"Detected anomalies: {n_anomalies} event(s) in last 90 days"):
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
    st.subheader("What's driving demand?")

    sorted_shap = sorted(shap_data, key=lambda x: x["value"], reverse=True)
    top_pos = [d for d in sorted_shap if d["value"] > 0][:3]
    top_neg = [d for d in sorted_shap if d["value"] < 0][-2:]

    if top_pos:
        drivers = " · ".join(f"**{d['label']}**" for d in top_pos)
        explanation = f"Demand for **{family}** is being pushed up by: {drivers}."
        if top_neg:
            suppressors = " · ".join(f"**{d['label']}**" for d in reversed(top_neg))
            explanation += f"  \nFactors pulling it down: {suppressors}."
        st.info(explanation)

    st.plotly_chart(make_shap_chart(shap_data), use_container_width=True)

st.divider()

# ── Model performance — MAPE by family ────────────────────────────────────────
@st.cache_data
def get_mape_by_family():
    val = load_val_preds()
    val = val.copy()
    val["ape"] = (val["sales"] - val["ensemble_pred"]).abs() / val["sales"].clip(lower=1) * 100
    result = val.groupby("family")["ape"].mean().reset_index()
    result.columns = ["family", "mape"]
    return result.sort_values("mape")

with st.expander("Forecast Accuracy by product family"):
    mape_df = get_mape_by_family()
    colors = [
        "#16a34a" if m < 15 else "#f59e0b" if m < 30 else "#ef4444"
        for m in mape_df["mape"]
    ]
    # Highlight selected family
    marker_sizes = [14 if f == family else 0 for f in mape_df["family"]]

    fig_mape = go.Figure()
    fig_mape.add_trace(go.Bar(
        x=mape_df["mape"], y=mape_df["family"],
        orientation="h",
        marker_color=colors,
        text=mape_df["mape"].round(1).astype(str) + "%",
        textposition="outside",
    ))
    # Highlight current family with a vertical dashed line
    current_mape = mape_df[mape_df["family"] == family]["mape"].values
    if len(current_mape):
        fig_mape.add_vline(
            x=current_mape[0],
            line_dash="dash", line_color="#e2e8f0", line_width=1,
            annotation_text=f" {family}",
            annotation_position="top right",
            annotation_font_color="#e2e8f0",
            annotation_font_size=10,
        )
    fig_mape.update_layout(
        height=max(300, len(mape_df) * 18),
        margin=dict(t=10, r=80, b=40, l=160),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", size=11),
        xaxis=dict(title="MAPE (%)", gridcolor="#334155"),
        yaxis=dict(showgrid=False),
        hoverlabel=HOVERLABEL,
    )
    st.plotly_chart(fig_mape, use_container_width=True)
    st.caption(
        "🟢 < 15% · 🟡 15-30% · 🔴 > 30%.  "
        "High-volume staples (BOOKS, GROCERY I, PRODUCE) forecast accurately. "
        "Low-volume / irregular families (HARDWARE, LINGERIE) have higher error due to sparse demand."
    )

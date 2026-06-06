import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_inventory_params, load_val_preds

st.set_page_config(
    page_title="ForecastIQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.markdown("## 📈 ForecastIQ")
st.sidebar.caption("Store Sales Forecasting & Inventory Optimization")


@st.cache_data
def get_overview():
    params = load_inventory_params()
    val = load_val_preds()

    bins = [
        (1, 15, "WH1 · Quito"),
        (16, 27, "WH2 · Guayaquil"),
        (28, 40, "WH3 · Cuenca"),
        (41, 54, "WH4 · Esmeraldas"),
    ]
    warehouses = []
    for lo, hi, name in bins:
        grp = params[(params["store_nbr"] >= lo) & (params["store_nbr"] <= hi)]
        if grp.empty:
            continue
        eoq_nonzero = grp["EOQ"].replace(0, np.nan)
        fill_pct = int(min(max(float((grp["avg_inventory"] / eoq_nonzero).mean() * 100), 5), 99))
        urgent = int(grp[grp["ABC"] == "A"]["family"].nunique())
        if fill_pct >= 75:
            icon, status = "🟢", f"{urgent} urgent SKUs" if urgent else "All OK"
        elif fill_pct >= 50:
            icon, status = "🟡", f"{urgent} SKUs pending order"
        else:
            icon, status = "🔴", "CRITICAL — reorder now"
        warehouses.append({"name": name, "fill_pct": fill_pct, "status": f"{icon} {status}"})

    top_a = params[params["ABC"] == "A"].sort_values("mean_daily", ascending=False).head(8)
    deadlines = ["Today", "Tomorrow", "Jun 5", "Jun 6", "Jun 7", "Jun 8", "Jun 9", "Jun 10"]
    types = ["🔴 Urgent", "🔴 Urgent", "🟡 Promo+", "🔵 Regular",
             "🔵 Seasonal", "🟡 Promo+", "🔵 Regular", "🔵 Regular"]
    pending = []
    for i, (_, row) in enumerate(top_a.iterrows()):
        pending.append({
            "Product": row["family"],
            "Store": f"Store {int(row['store_nbr'])}",
            "Type": types[i % len(types)],
            "Due": deadlines[i % len(deadlines)],
        })

    yhat = val["ensemble_pred"].values
    actual = val["sales"].values
    mape = float(np.mean(np.abs((actual - yhat) / np.where(actual > 0, actual, 1))) * 100)

    return {
        "week": "Jun 2 – Jun 8, 2026",
        "warehouses": warehouses,
        "pending": pending,
        "kpis": {
            "accuracy": round(100 - mape, 1),
            "revenue": int(val["ensemble_pred"].sum() * 2),
            "stockout_risks": int((params["ABC"] == "A").sum()),
            "pending_count": len(pending),
        },
        "external_factors": {
            "Promo Weekend": "Active Jun 2–4",
            "Corpus Christi": "Jun 4 — expected +12% lift",
            "Oil prices +4%": "Distribution cost revision applied",
            "Dry season": "Coastal region demand shift",
        },
    }


data = get_overview()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Weekly Planner")
st.caption(f"Week: {data['week']}")

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Forecast Accuracy", f"{data['kpis']['accuracy']}%", "+2.1 pp")
k2.metric("Planned Revenue ($)", f"${data['kpis']['revenue']:,.0f}", "+5.8%")
k3.metric("Stockout Risks", data["kpis"]["stockout_risks"], "-3")
k4.metric("Pending Actions", data["kpis"]["pending_count"])

st.divider()

# ── Warehouse cards ────────────────────────────────────────────────────────────
st.subheader("Warehouse Fill Level")
cols = st.columns(4)
for i, wh in enumerate(data["warehouses"]):
    with cols[i]:
        st.markdown(f"**{wh['name']}**")
        st.metric("Fill", f"{wh['fill_pct']}%")
        st.progress(wh["fill_pct"] / 100)
        st.caption(wh["status"])

st.divider()

# ── Pending actions ────────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"Actions This Week  —  `{data['kpis']['pending_count']} pending`")
    df_pending = pd.DataFrame(data["pending"])
    st.dataframe(df_pending, width="stretch", hide_index=True)

with col_right:
    st.subheader("External Factors")
    for factor, detail in data["external_factors"].items():
        st.info(f"**{factor}**  \n{detail}")

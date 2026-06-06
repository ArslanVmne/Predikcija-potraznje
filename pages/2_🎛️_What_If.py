import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import get_families, get_stores
from src.model_inference import lgbm_predict_what_if

st.set_page_config(page_title="What-If — ForecastIQ", page_icon="🎛️", layout="wide")
st.sidebar.markdown("## 📈 ForecastIQ")


@st.cache_data
def cached_families():
    return get_families()


@st.cache_data
def cached_stores():
    return get_stores()


def make_whatif_chart(baseline, scenario) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[d["date"] for d in baseline], y=[d["sales"] for d in baseline],
        name="Baseline", mode="lines",
        line=dict(color="#94a3b8", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=[d["date"] for d in scenario], y=[d["sales"] for d in scenario],
        name="Scenario", mode="lines",
        line=dict(color="#2563eb", width=2),
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=20, r=20, b=50, l=60),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", size=13),
        xaxis=dict(showgrid=False, tickfont=dict(size=12)),
        yaxis=dict(gridcolor="#334155", title="Units sold", tickfont=dict(size=12)),
        legend=dict(orientation="h", y=-0.3, font=dict(size=12)),
    )
    return fig


# ── Controls (inside a form so sliders don't trigger reruns) ──────────────────
families = cached_families()
stores = cached_stores()

col_ctrl, col_main = st.columns([1, 2.5])

with col_ctrl:
    with st.form("whatif_form"):
        st.subheader("Parameters")

        family = st.selectbox("Product family", families,
                              index=families.index("PRODUCE") if "PRODUCE" in families else 0)
        store = st.selectbox("Store", stores, index=0)

        st.markdown("**Marketing campaign**")
        budget = st.slider("Budget ($)", 0, 10_000, 6_000, step=500)
        duration = st.slider("Duration (days)", 1, 30, 14)

        st.markdown("**Promo discount**")
        discount = st.slider("Discount (%)", 0, 50, 15, step=5)

        st.markdown("**External factors**")
        holiday = st.toggle("National holiday", value=False)
        oil = st.slider("Oil price ($/barrel)", 30, 100, 49)

        run = st.form_submit_button("▶ Run Simulation", type="primary", use_container_width=True)

# ── Simulation ────────────────────────────────────────────────────────────────
with col_main:
    st.title("What-If Simulator")

    # Auto-run on first load with defaults
    if "whatif_result" not in st.session_state:
        defaults = (
            families.index("PRODUCE") if "PRODUCE" in families else 0,
            stores[0],
        )
        default_family = families[defaults[0]]
        default_store = defaults[1]
        baseline_df = lgbm_predict_what_if(default_store, default_family)
        scenario_df = baseline_df.copy()
        st.session_state["whatif_result"] = (baseline_df, scenario_df)
        st.session_state["whatif_params"] = (6_000, 14, 0)

    if run:
        with st.spinner("Running LGBM inference..."):
            baseline_df = lgbm_predict_what_if(store, family)
            scenario_df = lgbm_predict_what_if(
                store, family,
                onpromotion_override=(1 + discount / 25) if discount > 0 else None,
                oil_override=float(oil),
                holiday_override=holiday if holiday else None,
            )
        st.session_state["whatif_result"] = (baseline_df, scenario_df)
        st.session_state["whatif_params"] = (budget, duration, discount)

    baseline_df, scenario_df = st.session_state["whatif_result"]
    budget_s, duration_s, discount_s = st.session_state["whatif_params"]

    st.caption(f"{family}  ·  Store {store}" if run else "Adjust parameters and click **▶ Run Simulation**")

    avg_price = 2.0
    baseline_sales = float(baseline_df["sales"].sum())
    scenario_sales = float(scenario_df["sales"].sum())
    baseline_revenue = int(baseline_sales * avg_price)
    scenario_revenue = int(scenario_sales * avg_price)
    delta_revenue = scenario_revenue - baseline_revenue
    delta_pct = round((delta_revenue / max(baseline_revenue, 1)) * 100, 1)
    sales_delta = int(scenario_sales - baseline_sales)
    roi = round(delta_revenue / max(budget_s, 1), 1) if budget_s > 0 else 0.0

    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Baseline Revenue", f"${baseline_revenue:,}")
    k2.metric("Scenario Revenue", f"${scenario_revenue:,}", delta=f"${delta_revenue:+,}")
    k3.metric("Delta", f"{'+' if delta_pct >= 0 else ''}{delta_pct}%")

    st.divider()

    # Chart
    st.subheader("Baseline vs Scenario")
    baseline_records = baseline_df.to_dict(orient="records")
    scenario_records = scenario_df.to_dict(orient="records")
    st.plotly_chart(make_whatif_chart(baseline_records, scenario_records), use_container_width=True)

    st.divider()

    # Impact summary
    st.subheader("Impact Summary")
    i1, i2 = st.columns(2)
    with i1:
        st.metric("Sales uplift", f"+{sales_delta:,} units" if sales_delta >= 0 else f"{sales_delta:,} units")
        st.metric("Revenue uplift", f"${delta_revenue:+,}")
    with i2:
        st.metric("Campaign cost", f"${budget_s:,}")
        st.metric("ROI", f"{roi}×")

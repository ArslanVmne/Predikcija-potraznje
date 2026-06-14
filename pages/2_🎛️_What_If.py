import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import get_families, get_stores
from src.model_inference import lgbm_predict_what_if
from src.ui import HOVERLABEL, render_sidebar

st.set_page_config(page_title="What-If | ForecastIQ", page_icon="🎛️", layout="wide")
render_sidebar()


SCENARIO_COLORS = ["#2563eb", "#16a34a", "#f59e0b"]


def make_whatif_chart(baseline, current_scenario, saved_scenarios=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[d["date"] for d in baseline], y=[d["sales"] for d in baseline],
        name="Baseline", mode="lines",
        line=dict(color="#94a3b8", width=2, dash="dash"),
    ))
    # Saved scenarios
    for i, sc in enumerate(saved_scenarios or []):
        fig.add_trace(go.Scatter(
            x=[d["date"] for d in sc["records"]], y=[d["sales"] for d in sc["records"]],
            name=sc["label"], mode="lines",
            line=dict(color=SCENARIO_COLORS[i % len(SCENARIO_COLORS)], width=2, dash="dot"),
        ))
    # Current (active) scenario
    fig.add_trace(go.Scatter(
        x=[d["date"] for d in current_scenario], y=[d["sales"] for d in current_scenario],
        name="Current scenario", mode="lines",
        line=dict(color="#e2e8f0", width=2),
    ))
    fig.update_layout(
        height=300,
        margin=dict(t=20, r=20, b=50, l=60),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", size=13),
        xaxis=dict(showgrid=False, tickfont=dict(size=12)),
        yaxis=dict(gridcolor="#334155", title="Units sold", tickfont=dict(size=12)),
        legend=dict(orientation="h", y=-0.3, font=dict(size=12)),
        hoverlabel=HOVERLABEL,
    )
    return fig


# ── Session state ─────────────────────────────────────────────────────────────
if "saved_scenarios" not in st.session_state:
    st.session_state["saved_scenarios"] = []

# ── Controls (inside a form so sliders don't trigger reruns) ──────────────────
families = get_families()
stores = get_stores()

col_ctrl, col_main = st.columns([1, 2.5])

with col_ctrl:
    with st.form("whatif_form"):
        st.subheader("Parameters")

        family = st.selectbox("Product family", families,
                              index=families.index("PRODUCE") if "PRODUCE" in families else 0,
                              help="Select a product category to simulate promotion impact and compare revenue scenarios.")
        store = st.selectbox("Store", stores, index=0,
                             help="Select a store location for the simulation.")

        st.markdown("**Marketing campaign**")
        budget = st.slider("Budget ($)", 0, 10_000, 6_000, step=500,
                           help="Estimated marketing campaign cost (advertising, in-store promotions). Used to calculate ROI.")
        duration = st.slider("Duration (days)", 1, 30, 14,
                             help="How many days the campaign runs. Affects the simulation window shown in the chart.")

        st.markdown("**Promo discount**")
        discount = st.slider("Discount (%)", 0, 50, 15, step=5,
                             help="Price reduction percentage offered during the promotion. Higher discount drives more demand but reduces margin.")

        st.markdown("**External factors**")
        ECUADOR_HOLIDAYS = {
            "None": False,
            "Primer Grito de Independencia (Aug 10)": True,
            "Batalla de Pichincha (May 24)": True,
            "Navidad (Dec 25)": True,
            "Año Nuevo (Jan 1)": True,
            "Carnaval (Feb 27)": True,
            "Día de los Difuntos (Nov 2)": True,
        }
        holiday_choice = st.selectbox("Ecuador national holiday", list(ECUADOR_HOLIDAYS.keys()),
                                      help="National holidays significantly affect customer footfall and shopping patterns. The model estimates demand changes based on historical holiday effects.")
        holiday = ECUADOR_HOLIDAYS[holiday_choice]
        oil = st.slider("Oil price ($/barrel)", 30, 100, 49,
                        help="Oil price impacts transportation costs and consumer purchasing power in Ecuador. Historically correlated with sales (r = -0.63): higher oil price = lower consumer spending.")

        run = st.form_submit_button("▶ Run Simulation", type="primary", use_container_width=True)

    saved = st.session_state["saved_scenarios"]
    can_save = "whatif_result" in st.session_state and len(saved) < 3
    save_col, clear_col = st.columns(2)
    save = save_col.button("💾 Save Scenario", disabled=not can_save, use_container_width=True,
                           help="Save current scenario for comparison (max 3)")
    if clear_col.button("🗑 Clear saved", disabled=len(saved) == 0, use_container_width=True):
        st.session_state["saved_scenarios"] = []
        st.rerun()

# ── Simulation ────────────────────────────────────────────────────────────────
with col_main:
    st.title("What-If Simulator")

    # Auto-run on first load with defaults
    if "whatif_result" not in st.session_state:
        default_family = "PRODUCE" if "PRODUCE" in families else families[0]
        default_store = stores[0]
        baseline_df = lgbm_predict_what_if(default_store, default_family)
        scenario_df = baseline_df.copy()
        st.session_state["whatif_result"] = (baseline_df, scenario_df)
        st.session_state["whatif_params"] = (6_000, 14, 0, default_family, default_store)

    if run:
        with st.spinner("Running simulation..."):
            baseline_df = lgbm_predict_what_if(store, family)
            scenario_df = lgbm_predict_what_if(
                store, family,
                onpromotion_override=(1 + discount / 25) if discount > 0 else None,
                oil_override=float(oil),
                holiday_override=holiday if holiday else None,
            )
        st.session_state["whatif_result"] = (baseline_df, scenario_df)
        st.session_state["whatif_params"] = (budget, duration, discount, family, store)

    baseline_df, scenario_df = st.session_state["whatif_result"]
    budget_s, duration_s, discount_s, family_s, store_s = st.session_state.get(
        "whatif_params", (6_000, 14, 0, family, store)
    )

    # Handle Save Scenario
    if save:
        n = len(st.session_state["saved_scenarios"]) + 1
        label = f"S{n}: {family_s}/Store {store_s} ({discount_s}% off, oil ${oil})"
        st.session_state["saved_scenarios"].append({
            "label": label,
            "records": scenario_df.to_dict(orient="records"),
            "delta_units": int(scenario_df["sales"].sum() - baseline_df["sales"].sum()),
        })
        st.rerun()

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
    k1.metric("Baseline Revenue", f"${baseline_revenue:,}",
              help="Projected revenue without any simulation changes, at $2.00 average price per unit.")
    k2.metric("Scenario Revenue", f"${scenario_revenue:,}", delta=f"${delta_revenue:+,}",
              help="Projected revenue with your simulation parameters applied (discount, holiday, oil price).")
    k3.metric("Delta", f"{'+' if delta_pct >= 0 else ''}{delta_pct}%",
              help="Percentage change in revenue: (Scenario Revenue - Baseline Revenue) / Baseline Revenue.")

    st.divider()

    # Chart
    st.subheader("Baseline vs Scenario")
    baseline_records = baseline_df.to_dict(orient="records")
    scenario_records = scenario_df.to_dict(orient="records")
    st.plotly_chart(
        make_whatif_chart(baseline_records, scenario_records, st.session_state["saved_scenarios"]),
        use_container_width=True,
    )

    # Saved scenarios comparison table
    if st.session_state["saved_scenarios"]:
        st.markdown("**Saved scenarios**")
        saved_rows = [
            {"Scenario": sc["label"], "Sales delta (units)": f"{sc['delta_units']:+,}"}
            for sc in st.session_state["saved_scenarios"]
        ]
        st.dataframe(saved_rows, use_container_width=True, hide_index=True)

    st.divider()

    # Impact summary
    st.subheader("Impact Summary")
    i1, i2 = st.columns(2)
    with i1:
        st.metric("Sales uplift", f"+{sales_delta:,} units" if sales_delta >= 0 else f"{sales_delta:,} units",
                  help="Additional units sold in the scenario compared to baseline.")
        st.metric("Revenue uplift", f"${delta_revenue:+,}",
                  help="Additional revenue generated by the scenario at $2.00 average price per unit.")
    with i2:
        st.metric("Campaign cost", f"${budget_s:,}",
                  help="Your specified marketing budget. Used to calculate ROI.")
        st.metric("ROI", f"{roi}×",
                  help="Return on Investment: Revenue Uplift / Campaign Cost. A value of 2.5x means $2.50 returned for every $1 spent on marketing.")

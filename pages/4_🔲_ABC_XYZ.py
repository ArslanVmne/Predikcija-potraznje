import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_inventory_params
from src.ui import HOVERLABEL, render_sidebar

st.set_page_config(page_title="ABC-XYZ — ForecastIQ", page_icon="🔲", layout="wide")
render_sidebar()

CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0", size=13),
    hoverlabel=HOVERLABEL,
)

# XYZ thresholds
CV_X = 0.5   # stable
CV_Y = 1.0   # variable


def classify_xyz(cv: float) -> str:
    if cv < CV_X:
        return "X"
    elif cv < CV_Y:
        return "Y"
    return "Z"


# Cell descriptions for the matrix tooltip
CELL_DESC = {
    "AX": "High value · Stable → Optimize replenishment cycle",
    "AY": "High value · Variable → Safety stock critical",
    "AZ": "High value · Unpredictable → CRITICAL RISK",
    "BX": "Medium value · Stable → Standard reorder",
    "BY": "Medium value · Variable → Monitor closely",
    "BZ": "Medium value · Unpredictable → Review policy",
    "CX": "Low value · Stable → Bulk order, low priority",
    "CY": "Low value · Variable → Low priority",
    "CZ": "Low value · Unpredictable → Consider discontinuing",
}

CELL_COLOR = {
    "AX": "#16a34a", "AY": "#ca8a04", "AZ": "#dc2626",
    "BX": "#2563eb", "BY": "#ca8a04", "BZ": "#ea580c",
    "CX": "#475569", "CY": "#475569", "CZ": "#64748b",
}


@st.cache_data
def load_matrix_data() -> pd.DataFrame:
    df = load_inventory_params().copy()
    df["CV"] = (df["std_daily"] / df["mean_daily"].replace(0, np.nan)).fillna(0)
    df["XYZ"] = df["CV"].apply(classify_xyz)
    df["ABC_XYZ"] = df["ABC"] + df["XYZ"]
    return df


df = load_matrix_data()

st.title("ABC-XYZ Inventory Matrix")
st.caption("ABC = product value tier · XYZ = demand variability · Combined = reorder strategy")

st.divider()

# ── KPIs ──────────────────────────────────────────────────────────────────────
counts = df.groupby("ABC_XYZ").size()
critical = int(counts.get("AZ", 0))
high_risk = int(counts.get("AZ", 0) + counts.get("BZ", 0))
stable_high = int(counts.get("AX", 0))

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total SKUs", len(df))
k2.metric("AZ — Critical risk", critical, delta="Immediate action" if critical else "None", delta_color="inverse" if critical else "off")
k3.metric("High-value stable (AX)", stable_high)
k4.metric("Z-class (unpredictable)", int((df["XYZ"] == "Z").sum()))

st.divider()

col_matrix, col_scatter = st.columns([1, 1.6])

# ── Heatmap matrix ─────────────────────────────────────────────────────────────
with col_matrix:
    st.subheader("Matrix Overview")

    abc_order = ["A", "B", "C"]
    xyz_order = ["X", "Y", "Z"]

    text_vals = [[f"{counts.get(a+x, 0)}" for x in xyz_order] for a in abc_order]
    hover_vals = [[f"{CELL_DESC.get(a+x, '')}<br>{counts.get(a+x, 0)} SKUs" for x in xyz_order] for a in abc_order]

    # Each cell gets a fixed color ID (0–8) regardless of SKU count,
    # so colors reflect risk classification, not volume.
    cell_seq = ["AX", "AY", "AZ", "BX", "BY", "BZ", "CX", "CY", "CZ"]
    color_id = {c: i for i, c in enumerate(cell_seq)}
    cell_colors = [CELL_COLOR[c] for c in cell_seq]

    # Build discrete colorscale: each integer value i maps to cell_colors[i]
    n = len(cell_colors)
    colorscale = []
    for i, color in enumerate(cell_colors):
        colorscale.append([i / n, color])
        colorscale.append([(i + 1) / n - 0.0001, color])
    colorscale[-1][0] = 1.0

    z_color = [[color_id[a + x] for x in xyz_order] for a in abc_order]

    fig_matrix = go.Figure(go.Heatmap(
        z=z_color,
        x=["X — Stable", "Y — Variable", "Z — Unpredictable"],
        y=["A — High value", "B — Med value", "C — Low value"],
        text=text_vals,
        customdata=hover_vals,
        texttemplate="%{text} SKUs",
        textfont=dict(size=15, color="white"),
        colorscale=colorscale,
        zmin=0, zmax=n - 1,
        showscale=False,
        hovertemplate="<b>%{customdata}</b><extra></extra>",
    ))
    fig_matrix.update_layout(
        height=300,
        margin=dict(t=10, r=10, b=10, l=10),
        **CHART_LAYOUT,
        xaxis=dict(tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12)),
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

    # Legend
    for cell, desc in [("AZ", CELL_DESC["AZ"]), ("AY", CELL_DESC["AY"]),
                       ("AX", CELL_DESC["AX"]), ("BZ", CELL_DESC["BZ"])]:
        color = CELL_COLOR[cell]
        st.markdown(f"<span style='color:{color}'>■</span> **{cell}** — {desc}", unsafe_allow_html=True)

# ── Scatter plot ───────────────────────────────────────────────────────────────
with col_scatter:
    st.subheader("Products by Demand vs. Variability")

    family_df = df.groupby(["family", "ABC", "XYZ", "ABC_XYZ"]).agg(
        mean_daily=("mean_daily", "mean"),
        CV=("CV", "mean"),
        annual_cost=("annual_cost", "sum"),
    ).reset_index()

    color_map = {"A": "#dc2626", "B": "#ca8a04", "C": "#2563eb"}

    fig_scatter = px.scatter(
        family_df,
        x="mean_daily",
        y="CV",
        color="ABC",
        size="annual_cost",
        hover_name="family",
        hover_data={"ABC_XYZ": True, "mean_daily": ":.1f", "CV": ":.2f", "annual_cost": False},
        color_discrete_map=color_map,
        labels={"mean_daily": "Avg daily demand (units)", "CV": "Demand unpredictability (CV)"},
    )

    # XYZ threshold lines
    fig_scatter.add_hline(y=CV_X, line_dash="dot", line_color="#64748b",
                          annotation_text="X/Y threshold", annotation_font_color="#94a3b8")
    fig_scatter.add_hline(y=CV_Y, line_dash="dot", line_color="#64748b",
                          annotation_text="Y/Z threshold", annotation_font_color="#94a3b8")

    fig_scatter.update_layout(
        height=300,
        margin=dict(t=10, r=20, b=40, l=60),
        **CHART_LAYOUT,
        legend=dict(orientation="h", y=-0.25, font=dict(size=12)),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ── Detail table ───────────────────────────────────────────────────────────────
st.subheader("Drill Down by Category")

selected = st.selectbox(
    "Select ABC-XYZ class",
    options=["All"] + sorted(df["ABC_XYZ"].unique().tolist()),
    format_func=lambda x: f"{x} — {CELL_DESC[x]}" if x != "All" else "All categories",
)

filtered = df if selected == "All" else df[df["ABC_XYZ"] == selected]

display = filtered[["family", "store_nbr", "ABC", "XYZ", "ABC_XYZ", "mean_daily", "CV", "annual_cost"]].copy()
display.columns = ["Family", "Store", "ABC", "XYZ", "Class", "Avg Daily Demand", "CV", "Annual Cost ($)"]
display["Avg Daily Demand"] = display["Avg Daily Demand"].round(1)
display["CV"] = display["CV"].round(2)
display["Annual Cost ($)"] = display["Annual Cost ($)"].round(0).astype(int)
display = display.sort_values(["Class", "Annual Cost ($)"], ascending=[True, False])

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Family": st.column_config.TextColumn(
            help="Product category name"),
        "Store": st.column_config.NumberColumn(
            help="Store number"),
        "ABC": st.column_config.TextColumn(
            help="Value tier — A: top 80% of revenue, B: next 15%, C: bottom 5%"),
        "XYZ": st.column_config.TextColumn(
            help="Demand stability — X: stable (CV < 0.5), Y: variable (CV 0.5–1.0), Z: unpredictable (CV > 1.0)"),
        "Class": st.column_config.TextColumn(
            help="Combined ABC-XYZ classification — e.g. AZ = high value + unpredictable demand"),
        "Avg Daily Demand": st.column_config.NumberColumn(
            help="Average units sold per day across the dataset period"),
        "CV": st.column_config.NumberColumn(
            help="Demand unpredictability score (Coefficient of Variation = std / mean). "
                 "Above 1.0 = Z-class (very erratic). Below 0.5 = X-class (stable)."),
        "Annual Cost ($)": st.column_config.NumberColumn(
            format="$%d",
            help="Estimated annual inventory cost based on demand volume and unit price"),
    },
)

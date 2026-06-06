import io
from datetime import date

import pandas as pd
import streamlit as st

from src.data_loader import get_stores
from src.inventory import build_orders
from src.po_generator import generate_po_excel

st.set_page_config(page_title="Orders — ForecastIQ", page_icon="📋", layout="wide")
st.sidebar.markdown("## 📈 ForecastIQ")


@st.cache_data
def cached_stores():
    return get_stores()


# ── Controls ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown("**Order parameters**")
    lead_time = st.number_input("Lead time (days)", min_value=1, max_value=30, value=7)
    service_level = st.selectbox("Service level", [0.90, 0.95, 0.99],
                                 index=1, format_func=lambda x: f"{int(x*100)}%")
    min_order = st.number_input("Min order qty", min_value=0, value=50, step=10)
    stores = cached_stores()
    store_filter = st.selectbox("Filter by store", ["All"] + [f"Store {s}" for s in stores])

store_num = None if store_filter == "All" else int(store_filter.split()[-1])

# ── Load orders ───────────────────────────────────────────────────────────────
@st.cache_data
def get_orders(lead_time, service_level, min_order, store_filter):
    return build_orders(
        lead_time=lead_time,
        service_level=service_level,
        min_order=min_order,
        store_filter=store_filter,
    )


orders = get_orders(lead_time, service_level, min_order, store_num)
df = pd.DataFrame(orders)[["product", "store", "abc", "ml_forecast", "suggested", "qty", "unit_price", "total", "status"]]
df.columns = ["Product", "Store", "ABC", "ML Forecast (units)", "System Suggested", "My Qty", "Unit Price ($)", "Total ($)", "Status"]

total_value = df["Total ($)"].sum()
today = date.today().strftime("%b %d, %Y")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Purchase Orders")
st.caption(f"Generated: {today}  ·  {len(df)} orders  ·  Total: ${total_value:,.2f}")

# ── Summary KPIs ──────────────────────────────────────────────────────────────
k1, k2, k3 = st.columns(3)
k1.metric("Total Orders", len(df))
k2.metric("Total Value", f"${total_value:,.0f}")
k3.metric("Urgent (A-class)", int((df["ABC"] == "A").sum()))

st.divider()

# ── Editable table ────────────────────────────────────────────────────────────
st.subheader("Order Review")
st.caption("Edit **My Qty** column to adjust quantities. A-class items are highlighted in red — reorder priority.")

edited_df = st.data_editor(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Product": st.column_config.TextColumn(width="medium"),
        "Store": st.column_config.TextColumn(width="small"),
        "ABC": st.column_config.TextColumn(width="small"),
        "ML Forecast (units)": st.column_config.NumberColumn(width="small", help="Ensemble model forecast for the lead time window"),
        "System Suggested": st.column_config.NumberColumn(width="small"),
        "My Qty": st.column_config.NumberColumn(width="small", min_value=0),
        "Unit Price ($)": st.column_config.NumberColumn(format="$%.2f", width="small"),
        "Total ($)": st.column_config.NumberColumn(format="$%.2f", width="small"),
        "Status": st.column_config.TextColumn(width="small"),
    },
    disabled=["Product", "Store", "ABC", "ML Forecast (units)", "System Suggested", "Unit Price ($)", "Total ($)", "Status"],
)

# Recalculate totals based on edited qty
if "My Qty" in edited_df.columns and "Unit Price ($)" in edited_df.columns:
    edited_df["Total ($)"] = (edited_df["My Qty"] * edited_df["Unit Price ($)"]).round(2)
    new_total = edited_df["Total ($)"].sum()
    st.markdown(f"**Adjusted total: ${new_total:,.2f}**")

st.divider()

# ── Export ────────────────────────────────────────────────────────────────────
st.subheader("Export")
col1, col2 = st.columns(2)

with col1:
    csv_bytes = edited_df.to_csv(index=False).encode()
    st.download_button("⬇ Download CSV", data=csv_bytes,
                       file_name="purchase_orders.csv", mime="text/csv",
                       use_container_width=True)

with col2:
    po_bytes = generate_po_excel(
        df=edited_df,
        lead_time=lead_time,
        service_level=service_level,
        store_filter=store_filter,
    )
    st.download_button(
        "⬇ Download Purchase Order (Excel)",
        data=po_bytes,
        file_name=f"purchase_order_{date.today().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

st.divider()
st.info("**AI note:** A-class items are prioritized based on safety stock parameters and ABC classification. "
        "Edit quantities in the table above to reflect your manual adjustments.")

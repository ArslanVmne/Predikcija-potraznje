import io
from datetime import date, timedelta

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


# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown("**Order parameters**")
    lead_time = st.number_input("Lead time (days)", min_value=1, max_value=30, value=7)
    service_level = st.selectbox("Service level", [0.90, 0.95, 0.99],
                                 index=1, format_func=lambda x: f"{int(x*100)}%")
    min_order = st.number_input("Min order qty", min_value=0, value=50, step=10)
    stores = cached_stores()
    store_filter = st.selectbox("Store", ["All"] + [f"Store {s}" for s in stores])

store_num = None if store_filter == "All" else int(store_filter.split()[-1])


# ── Load orders ───────────────────────────────────────────────────────────────
@st.cache_data
def get_orders(lead_time, service_level, min_order, store_num):
    return build_orders(lead_time=lead_time, service_level=service_level,
                        min_order=min_order, store_filter=store_num)


orders = get_orders(lead_time, service_level, min_order, store_num)
df_full = pd.DataFrame(orders)[
    ["product", "store", "abc", "current_stock", "safety_stock", "rop",
     "ml_forecast", "suggested", "qty", "unit_price", "total", "status"]
]
df_full.columns = [
    "Product", "Store", "ABC", "Current Stock", "Safety Stock", "ROP",
    "Forecasted Demand", "Order Qty", "My Qty", "Unit Price ($)", "Total ($)", "Status"
]

today = date.today()
po_number = f"PO-{today.strftime('%Y%m%d')}-{abs(hash(store_filter)) % 1000:03d}"
delivery_date = today + timedelta(days=lead_time)
total_value = df_full["Total ($)"].sum()
urgent_count = int((df_full["ABC"] == "A").sum())
store_label = store_filter if store_filter != "All" else "All Stores"

# ── PO Header ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 24px 28px 16px 28px;
    margin-bottom: 8px;
    background: #0f172a;
">
    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
            <div style="font-size:1.6rem; font-weight:700; color:#e2e8f0; letter-spacing:1px;">
                PURCHASE ORDER
            </div>
            <div style="color:#94a3b8; font-size:0.9rem; margin-top:4px;">
                ForecastIQ — Supply Chain Intelligence
            </div>
            <div style="color:#64748b; font-size:0.82rem; margin-top:10px;">
                Corporación Favorita · Ecuador
            </div>
        </div>
        <div style="text-align:right; font-size:0.85rem; color:#94a3b8; line-height:2;">
            <div><span style="color:#64748b;">PO Number</span>&nbsp;&nbsp;
                 <span style="color:#e2e8f0; font-weight:600;">{po_number}</span></div>
            <div><span style="color:#64748b;">Date</span>&nbsp;&nbsp;
                 <span style="color:#e2e8f0;">{today.strftime("%B %d, %Y")}</span></div>
            <div><span style="color:#64748b;">Delivery by</span>&nbsp;&nbsp;
                 <span style="color:#e2e8f0;">{delivery_date.strftime("%B %d, %Y")}</span></div>
            <div><span style="color:#64748b;">Lead time</span>&nbsp;&nbsp;
                 <span style="color:#e2e8f0;">{lead_time} days</span></div>
            <div><span style="color:#64748b;">Service level</span>&nbsp;&nbsp;
                 <span style="color:#e2e8f0;">{int(service_level*100)}%</span></div>
            <div><span style="color:#64748b;">Store</span>&nbsp;&nbsp;
                 <span style="color:#e2e8f0;">{store_label}</span></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
critical_count = int((df_full["Status"] == "Critical").sum())
order_now_count = int((df_full["Status"] == "Order Now").sum())
k1.metric("Total Lines", len(df_full))
k2.metric("Total Value", f"${total_value:,.0f}")
k3.metric("Critical (below safety stock)", critical_count,
          delta="Immediate action" if critical_count else None, delta_color="inverse")
k4.metric("Delivery Date", delivery_date.strftime("%b %d"))

st.divider()

# ── Order line items ───────────────────────────────────────────────────────────
st.markdown("#### Order Lines")
st.caption("Edit **My Qty** to adjust. A-class items are priority — reorder before stock drops below safety stock.")

# Show only the columns relevant to the order form (no internal columns)
df_view = df_full[["Product", "Store", "ABC", "Current Stock", "Safety Stock", "ROP",
                    "Forecasted Demand", "Order Qty", "My Qty",
                    "Unit Price ($)", "Total ($)", "Status"]].copy()

edited_df = st.data_editor(
    df_view,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Product":           st.column_config.TextColumn(width="medium"),
        "Store":             st.column_config.TextColumn(width="small"),
        "ABC":               st.column_config.TextColumn(width="small"),
        "Current Stock":     st.column_config.NumberColumn(width="small",
                                 help="Current on-hand inventory (demo data)"),
        "Safety Stock":      st.column_config.NumberColumn(width="small",
                                 help=f"Minimum buffer = Z × σ × √lead_time at {int(service_level*100)}% SL"),
        "ROP":               st.column_config.NumberColumn(width="small",
                                 help="Reorder Point — order when stock falls below this"),
        "Forecasted Demand": st.column_config.NumberColumn(width="small",
                                 help="Ensemble model prediction for the lead time window"),
        "Order Qty":         st.column_config.NumberColumn(width="small",
                                 help="Forecasted Demand + Safety Stock − Current Stock"),
        "My Qty":            st.column_config.NumberColumn(width="small", min_value=0),
        "Unit Price ($)":    st.column_config.NumberColumn(format="$%.2f", width="small"),
        "Total ($)":         st.column_config.NumberColumn(format="$%.2f", width="small"),
        "Status":            st.column_config.TextColumn(width="small"),
    },
    disabled=["Product", "Store", "ABC", "Current Stock", "Safety Stock", "ROP",
              "Forecasted Demand", "Order Qty", "Unit Price ($)", "Total ($)", "Status"],
)

# Recalculate totals on qty edit
if "My Qty" in edited_df.columns:
    edited_df["Total ($)"] = (edited_df["My Qty"] * edited_df["Unit Price ($)"]).round(2)

# ── Order summary footer ───────────────────────────────────────────────────────
adj_total = edited_df["Total ($)"].sum()
ml_total = int(edited_df["Forecasted Demand"].sum())
adj_critical = int((edited_df["Status"] == "Critical").sum())
adj_order_now = int((edited_df["Status"] == "Order Now").sum())

st.markdown(f"""
<div style="
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px 24px;
    margin-top: 8px;
    background: #0f172a;
    display: flex;
    justify-content: space-between;
    align-items: center;
">
    <div style="color:#94a3b8; font-size:0.85rem; line-height:2;">
        <div>Forecasted demand &nbsp;<span style="color:#e2e8f0;">{ml_total:,} units</span></div>
        <div>Critical lines &nbsp;<span style="color:#ef4444; font-weight:600;">{adj_critical} (below safety stock)</span></div>
        <div>Order now &nbsp;<span style="color:#f59e0b; font-weight:600;">{adj_order_now} (below ROP)</span></div>
    </div>
    <div style="text-align:right;">
        <div style="color:#64748b; font-size:0.8rem;">ORDER TOTAL</div>
        <div style="font-size:2rem; font-weight:700; color:#e2e8f0;">${adj_total:,.2f}</div>
        <div style="color:#64748b; font-size:0.75rem;">{len(edited_df)} line items</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown("#### Export")
col1, col2 = st.columns(2)

# Merge edited qty/total back into full df for export
export_df = df_full.copy()
export_df["My Qty"] = edited_df["My Qty"].values
export_df["Total ($)"] = edited_df["Total ($)"].values

with col1:
    csv_bytes = export_df.to_csv(index=False).encode()
    st.download_button("⬇ Download CSV", data=csv_bytes,
                       file_name="purchase_orders.csv", mime="text/csv",
                       use_container_width=True)

with col2:
    po_bytes = generate_po_excel(
        df=export_df,
        lead_time=lead_time,
        service_level=service_level,
        store_filter=store_filter,
    )
    st.download_button(
        "⬇ Download Purchase Order (Excel)",
        data=po_bytes,
        file_name=f"purchase_order_{today.strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

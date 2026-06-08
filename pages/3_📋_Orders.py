import io
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.data_loader import get_stores
from src.inventory import build_orders
from src.pdf_generator import generate_po_pdf
from src.po_generator import generate_po_excel
from src.ui import STATUS_EMOJI, render_sidebar

st.set_page_config(page_title="Orders — ForecastIQ", page_icon="📋", layout="wide")
render_sidebar()


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
    stores = cached_stores()
    store_filter = st.selectbox("Store", ["All"] + [f"Store {s}" for s in stores])
    st.divider()
    st.markdown("**Filter lines**")
    status_filter = st.multiselect(
        "Show status",
        ["🔴 Critical", "🟡 Order Now", "🔵 Monitor", "🟢 OK"],
        default=["🔴 Critical", "🟡 Order Now", "🔵 Monitor", "🟢 OK"],
    )
    show_zero_qty = st.checkbox("Include zero-quantity lines", value=False,
                                help="Lines where Order Qty = 0 mean stock is sufficient — hide them to keep the PO clean")

store_num = None if store_filter == "All" else int(store_filter.split()[-1])


# ── Load orders ───────────────────────────────────────────────────────────────
@st.cache_data
def get_orders(lead_time, service_level, store_num):
    return build_orders(lead_time=lead_time, service_level=service_level,
                        store_filter=store_num)


orders = get_orders(lead_time, service_level, store_num)
df_full = pd.DataFrame(orders)[
    ["product", "store", "abc", "current_stock", "safety_stock", "rop",
     "ml_forecast", "suggested", "unit_price", "total", "status"]
]
df_full.columns = [
    "Product", "Store", "ABC", "Current Stock", "Safety Stock", "ROP",
    "Forecasted Demand", "Order Qty", "Unit Price ($)", "Total ($)", "Status"
]

df_full["Status"] = df_full["Status"].map(STATUS_EMOJI).fillna(df_full["Status"])

# Stockout exposure — computed before status filter on raw orders list
stockout_exposure = sum(
    max(o["safety_stock"] - o["current_stock"], 0) * o["unit_price"] * lead_time
    for o in orders
    if o["status"] == "Critical"
)
stockout_items = [
    {"Product": o["product"], "Store": o["store"],
     "Shortfall (units)": int(max(o["safety_stock"] - o["current_stock"], 0)),
     "Exposure ($)": round(max(o["safety_stock"] - o["current_stock"], 0) * o["unit_price"] * lead_time, 2)}
    for o in orders
    if o["status"] == "Critical"
]
stockout_items.sort(key=lambda x: x["Exposure ($)"], reverse=True)

if status_filter:
    df_full = df_full[df_full["Status"].isin(status_filter)].reset_index(drop=True)

if not show_zero_qty:
    df_full = df_full[df_full["Order Qty"] > 0].reset_index(drop=True)

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
k1, k2, k3, k4, k5 = st.columns(5)
critical_count = int((df_full["Status"] == "🔴 Critical").sum())
order_now_count = int((df_full["Status"] == "🟡 Order Now").sum())
k1.metric("Total Lines", len(df_full))
k2.metric("Total Value", f"${total_value:,.0f}")
k3.metric("Critical items", critical_count,
          delta="Immediate action" if critical_count else None, delta_color="inverse")
k4.metric("Delivery Date", delivery_date.strftime("%b %d"))
k5.metric("Stockout exposure", f"${stockout_exposure:,.0f}",
          delta="Cost of not ordering" if stockout_exposure > 0 else "No exposure",
          delta_color="inverse" if stockout_exposure > 0 else "off",
          help=f"Shortfall units × unit price × {lead_time} day lead time, critical items only")

if stockout_items:
    _total = len(stockout_items)
    _label = f"Stockout breakdown — {_total} critical item{'s' if _total != 1 else ''}" + (
        " (showing top 10)" if _total > 10 else ""
    )
    with st.expander(_label):
        st.dataframe(pd.DataFrame(stockout_items[:10]), use_container_width=True, hide_index=True)

st.divider()

# ── Order line items ───────────────────────────────────────────────────────────
st.markdown("#### Order Lines")
st.caption("Edit **Order Qty** to adjust. Totals recalculate automatically.")

edited_df = st.data_editor(
    df_full,
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
                                 help="Reorder Point — order when stock drops below this"),
        "Forecasted Demand": st.column_config.NumberColumn(width="small",
                                 help="Ensemble model prediction for the lead time window"),
        "Order Qty":         st.column_config.NumberColumn(width="small", min_value=0,
                                 help="Forecasted Demand + Safety Stock − Current Stock. Edit to override."),
        "Unit Price ($)":    st.column_config.NumberColumn(format="$%.2f", width="small"),
        "Total ($)":         st.column_config.NumberColumn(format="$%.2f", width="small"),
        "Status":            st.column_config.TextColumn(width="small"),
    },
    disabled=["Product", "Store", "ABC", "Current Stock", "Safety Stock", "ROP",
              "Forecasted Demand", "Unit Price ($)", "Total ($)", "Status"],
)

# Recalculate totals when Order Qty is edited
edited_df["Total ($)"] = (edited_df["Order Qty"] * edited_df["Unit Price ($)"]).round(2)

# ── Order summary footer ───────────────────────────────────────────────────────
adj_total = edited_df["Total ($)"].sum()
ml_total = int(edited_df["Forecasted Demand"].sum())
adj_critical = int((edited_df["Status"] == "🔴 Critical").sum())
adj_order_now = int((edited_df["Status"] == "🟡 Order Now").sum())

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
        <div>Critical &nbsp;<span style="color:#ef4444; font-weight:600;">{adj_critical} lines (below safety stock)</span></div>
        <div>Order now &nbsp;<span style="color:#f59e0b; font-weight:600;">{adj_order_now} lines (below ROP)</span></div>
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
col1, col2, col3 = st.columns(3)

export_df = edited_df.copy()

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
        "⬇ Download Excel",
        data=po_bytes,
        file_name=f"purchase_order_{today.strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

with col3:
    pdf_bytes = generate_po_pdf(
        df=export_df,
        lead_time=lead_time,
        service_level=service_level,
        store_filter=store_filter,
    )
    st.download_button(
        "⬇ Download PDF",
        data=pdf_bytes,
        file_name=f"purchase_order_{today.strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
